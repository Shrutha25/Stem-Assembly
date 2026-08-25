"""Data capture tooling (build plan §10, build order item 3) — the tool that
unblocks labeling, called out in the plan as "the actual long pole".

Captures frames from the configured camera into `data/sessions/<session_id>/`
either on a fixed interval or on keypress, plus a manifest.jsonl. Every run
gets its own timestamped, uniquely-named session folder automatically, so
later train/val splits can be done *by session* (whole sittings held out
together) rather than randomly (§10) — important even for a single worker,
since frames a fraction of a second apart within one sitting are near-
duplicates and would leak between train/val if split randomly.

The --keypress preview window shows the current step's name and required
parts as an on-screen prompt, so you know what should be in frame before
each capture — build through the configured steps in order, pressing N to
advance the prompt as you go (it does not auto-advance; you're capturing
training data, not running the trained pipeline).

Press O to toggle orientation-negative mode for the current step (if it has
a `step<N>_wrong_orientation` class — see config.yaml) — stage that step's
part deliberately wrong (flipped/rotated) and capture. The mode is shown in
bold red on the panel and is recorded per-frame in the manifest
(`orientation_negative_class`), so labeling later knows which images are the
deliberate negatives without guessing from filenames. It resets to normal
automatically on N/P, so it can't silently carry over into the next step's
otherwise-normal captures.

Usage:
    python -m assembly_trainer.data_capture --keypress   # SPACE=capture, N/P=next/prev step, O=orientation-negative, ESC=quit
    python -m assembly_trainer.data_capture --interval-seconds 1.0
    python -m assembly_trainer.data_capture --keypress --session "morning-lighting"
    python -m assembly_trainer.data_capture --keypress --resume-session "20260821T125004Z_session_548e39"
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .camera import ThreadedCamera
from .config import load_config
from .gating import step_orientation_variant


def _new_session_dir(root: Path, session_label: str) -> tuple[str, Path]:
    session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{session_label}_{uuid.uuid4().hex[:6]}"
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir


def _next_frame_index(session_dir: Path) -> int:
    """Resuming a session must continue numbering after what's already
    there, or new captures would overwrite existing frames."""
    existing = [int(p.stem) for p in session_dir.glob("*.jpg") if p.stem.isdigit()]
    return max(existing) + 1 if existing else 0


_OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
_OVERLAY_SCALE = 0.6
_OVERLAY_THICKNESS = 1


def _wrap_to_width(text: str, max_width: int) -> list[str]:
    """Word-wrap so a long requirements list (later steps need many classes)
    never runs off the right edge — cv2.putText does not wrap on its own."""
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (w, _), _ = cv2.getTextSize(candidate, _OVERLAY_FONT, _OVERLAY_SCALE, _OVERLAY_THICKNESS)
        if w > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_overlay(frame, step, step_idx: int, total_steps: int, saved_count: int, orientation_class: str | None, orientation_mode: bool):
    """Compose the step prompt as its own panel above the camera feed, not
    drawn on top of it — the frame you're actually shooting stays untouched
    (it's also the version that gets saved to disk; see _save_frame).

    The capture-mode line is deliberately the most visually distinct part of
    the panel (its own bright color, not just text) — a mode you can miss
    at a glance is exactly what caused mislabeled captures before this was
    added.
    """
    frame_h, frame_w = frame.shape[:2]
    margin = 12
    max_text_width = frame_w - 2 * margin

    requires_text = ", ".join(f"{cls} x{n}" for cls, n in step.requires.items())
    raw_lines = [
        f"Step {step_idx + 1}/{total_steps}: {step.name}",
        f"should be visible: {requires_text}",
        f"captured so far: {saved_count}  |  SPACE=capture  N=next  P=prev  O=toggle orientation-negative  ESC=quit",
    ]

    if orientation_mode and orientation_class:
        mode_text = f"** CAPTURING WRONG-ORIENTATION NEGATIVE -> will be labeled '{orientation_class}' **"
        mode_color = (0, 0, 255)  # red — deliberately alarming, this is not the normal mode
    elif orientation_mode and not orientation_class:
        mode_text = "** O pressed, but this step has no wrong-orientation class — capturing as NORMAL **"
        mode_color = (0, 140, 255)  # orange warning
    else:
        mode_text = "mode: normal capture"
        mode_color = (0, 255, 255)

    mode_lines = _wrap_to_width(mode_text, max_text_width)
    body_lines: list[str] = []
    for raw in raw_lines:
        body_lines.extend(_wrap_to_width(raw, max_text_width))
    lines = mode_lines + body_lines

    # Measure actual glyph metrics instead of guessing a fixed pixel gap —
    # a guessed gap is exactly what let an earlier version's lines overlap.
    (_, text_h), baseline = cv2.getTextSize("Ag", _OVERLAY_FONT, _OVERLAY_SCALE, _OVERLAY_THICKNESS)
    line_step = text_h + baseline + 10
    panel_h = line_step * len(lines) + 10  # grows automatically for a long requirements list

    panel = np.full((panel_h, frame_w, 3), (30, 30, 30), dtype=frame.dtype)
    y = line_step
    for line in mode_lines:
        cv2.putText(panel, line, (margin, y), _OVERLAY_FONT, _OVERLAY_SCALE, mode_color, 2, cv2.LINE_AA)
        y += line_step
    for line in body_lines:
        cv2.putText(panel, line, (margin, y), _OVERLAY_FONT, _OVERLAY_SCALE, (0, 255, 255), _OVERLAY_THICKNESS, cv2.LINE_AA)
        y += line_step

    return cv2.vconcat([panel, frame])


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_root = Path(args.data_root)

    if args.resume_session:
        session_dir = data_root / args.resume_session
        if not session_dir.is_dir():
            raise SystemExit(f"--resume-session '{args.resume_session}' not found under {data_root}")
        session_id = args.resume_session
    else:
        session_id, session_dir = _new_session_dir(data_root, args.session)
    manifest_path = session_dir / "manifest.jsonl"

    camera = ThreadedCamera(cfg.camera).start()
    print(f"[data_capture] session={session_id} dir={session_dir}")
    print(f"[data_capture] camera resolution={camera.actual_resolution}")

    step_idx = max(0, min(len(cfg.steps) - 1, (args.step or 1) - 1))
    frame_index = _next_frame_index(session_dir)
    if frame_index > 0:
        print(f"[data_capture] resuming — {frame_index} frame(s) already in this session, continuing from there")
    known_classes = set(cfg.classes)
    orientation_mode = False
    try:
        if args.keypress:
            print("[data_capture] SPACE = capture frame, N/P = next/prev step, O = toggle orientation-negative, ESC = quit")
            print(f"[data_capture] build through the {len(cfg.steps)} configured steps in order and capture "
                  "as you go — see the on-screen step prompt for what should be visible in frame.")
            while True:
                frame = camera.get_latest_frame()
                step = cfg.steps[step_idx]
                orientation_class = step_orientation_variant(step.id, known_classes)
                if frame is not None:
                    annotated = _draw_overlay(
                        frame, step, step_idx, len(cfg.steps), frame_index, orientation_class, orientation_mode
                    )
                    cv2.imshow("assembly_trainer data_capture", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32 and frame is not None:  # SPACE
                    label = orientation_class if orientation_mode else None
                    frame_index = _save_frame(session_dir, manifest_path, frame_index, frame, args, session_id, step, label)
                elif key in (ord("n"), ord("N")):
                    step_idx = min(len(cfg.steps) - 1, step_idx + 1)
                    orientation_mode = False  # don't carry a special mode across a step change — same mistake as before
                elif key in (ord("p"), ord("P")):
                    step_idx = max(0, step_idx - 1)
                    orientation_mode = False
                elif key in (ord("o"), ord("O")):
                    orientation_mode = not orientation_mode
        else:
            step = cfg.steps[step_idx]
            print(f"[data_capture] capturing every {args.interval_seconds}s for step {step_idx + 1}/{len(cfg.steps)} "
                  f"({step.name}) — Ctrl+C to stop. Use --keypress instead if you want to switch steps live "
                  "or capture orientation-negatives.")
            while True:
                frame = camera.get_latest_frame()
                if frame is not None:
                    frame_index = _save_frame(session_dir, manifest_path, frame_index, frame, args, session_id, step, None)
                time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print(f"[data_capture] wrote {frame_index} frames to {session_dir}")


def _save_frame(session_dir: Path, manifest_path: Path, frame_index: int, frame, args, session_id: str, step, orientation_negative_class: str | None) -> int:
    filename = f"{frame_index:06d}.jpg"
    cv2.imwrite(str(session_dir / filename), frame)
    record = {
        "session_id": session_id,
        "session": args.session,
        "file": filename,
        "step_id": step.id,
        "step_name": step.name,
        # None for a normal capture; set to e.g. "step3_wrong_orientation"
        # when captured in orientation-negative mode (O key) — tells whoever
        # labels this image which class to draw the bbox as, instead of the
        # step's normal required class. Only steps with a real orientation
        # class have this available (see step_orientation_variant) — not
        # steps 1/2 anymore, since their "wrong orientation" turned out to
        # be a position problem, now handled by target_hole instead.
        "orientation_negative_class": orientation_negative_class,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return frame_index + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--data-root", default="data/sessions")
    parser.add_argument("--session", default="session", help="label for this sitting, e.g. 'morning-lighting' (each run gets its own folder regardless)")
    parser.add_argument("--resume-session", default=None, help="existing session folder name (under --data-root) to add more frames to, instead of starting a new session")
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--keypress", action="store_true", help="capture on SPACE instead of a fixed interval")
    parser.add_argument("--step", type=int, default=1, help="step number to start on; N/P keys change it live in --keypress mode")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
