"""Minimal local bounding-box labeling tool — same cv2 window style as
data_capture.py (panel above the image, keyboard-driven), instead of an
external app. Left-drag to draw a box, keys 0-9 then a,b,d,e,... (see
_class_key_label) pick the class in cfg.classes order, right-click a box to
delete it.

Saves standard YOLO-format labels directly to `data/dataset/labels/<split>/`
— the same place `verify_labels.py` and `training/train.py` expect — one
`.txt` per image, saved automatically on every edit and on leaving an image
(no separate save step, nothing lost if you close mid-image).

Auto-resumes at the first not-yet-labeled image by default — just re-run
the same command to continue where you left off, no index-tracking needed.

Also has a second mode (`--mode keypoints`) for the block_7x7 corner-pose
dataset used by steps 1/2's hole-position check (see hole_position.py) —
click the block's 4 corners in a fixed order on images that already have a
block_7x7 box in the main dataset; see `run_keypoints()` below for details.

Usage:
    python -m assembly_trainer.label_tool --split train
    python -m assembly_trainer.label_tool --split val --start-at 50   # manual override, if ever needed
    python -m assembly_trainer.label_tool --split train --indices 20,27,45,102   # only these, in this order
    python -m assembly_trainer.label_tool --split train --mode keypoints         # block_7x7 corner pose dataset
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .data_capture import _wrap_to_width
from .hole_position import CORNER_NAMES

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SCALE = 0.55
_THICKNESS = 1
Box = tuple[int, int, int, int, int]  # (class_id, x1, y1, x2, y2) in pixel coords


# n/p/u/c are already bound to actions (next/prev/undo/clear) below --
# never assign a class to one of these letters.
_RESERVED_KEYS = {"n", "p", "u", "c"}


def _class_key_label(class_id: int) -> str:
    """0-9 for the first 10 classes (digit keys); beyond that, letters
    a,b,d,e,... (skipping _RESERVED_KEYS) since there's no single keystroke
    for a two-digit number."""
    if class_id < 10:
        return str(class_id)
    letters = (c for c in "abcdefghijklmnopqrstuvwxyz" if c not in _RESERVED_KEYS)
    for _ in range(class_id - 10):
        next(letters)
    return next(letters)


def _class_color(class_id: int, n: int) -> tuple[int, int, int]:
    hue = int(179 * class_id / max(1, n))
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _load_boxes(label_path: Path, img_w: int, img_h: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, w, h = (float(v) for v in parts[1:])
        x1, y1 = int((xc - w / 2) * img_w), int((yc - h / 2) * img_h)
        x2, y2 = int((xc + w / 2) * img_w), int((yc + h / 2) * img_h)
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def _save_boxes(label_path: Path, boxes: list[Box], img_w: int, img_h: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        xc, yc = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        w, h = abs(x2 - x1) / img_w, abs(y2 - y1) / img_h
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    text = "\n".join(lines) + ("\n" if lines else "")
    label_path.write_text(text, encoding="utf-8")


class _MouseState:
    def __init__(self):
        self.boxes: list[Box] = []
        self.current_class = 0
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.dirty = False


def _make_mouse_callback(state: _MouseState, panel_h: int, img_w: int, img_h: int):
    def _clamp(x, y):
        return max(0, min(img_w, x)), max(0, min(img_h, y - panel_h))

    def _on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < panel_h:
                return
            state.drag_start = _clamp(x, y)
            state.drag_end = state.drag_start
        elif event == cv2.EVENT_MOUSEMOVE:
            if state.drag_start is not None:
                state.drag_end = _clamp(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if state.drag_start is not None:
                x1, y1 = state.drag_start
                x2, y2 = _clamp(x, y)
                if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                    state.boxes.append((state.current_class, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                    state.dirty = True
                state.drag_start = None
                state.drag_end = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if y < panel_h:
                return
            px, py = _clamp(x, y)
            for i in reversed(range(len(state.boxes))):
                _, x1, y1, x2, y2 = state.boxes[i]
                if x1 <= px <= x2 and y1 <= py <= y2:
                    del state.boxes[i]
                    state.dirty = True
                    break

    return _on_mouse


def _render_panel(
    frame, raw_lines: list[str], colored_line_indices: set[int], highlight_color: tuple[int, int, int]
):
    """Shared panel-above-image chrome for both label modes: word-wraps
    `raw_lines` to the frame width, measures actual glyph metrics for line
    spacing (not a guessed constant -- overlapping text was a real bug here
    once), and vconcats a dark panel above the frame. `colored_line_indices`
    picks which *pre-wrap* lines render in `highlight_color` instead of the
    default yellow."""
    margin = 10
    max_text_width = frame.shape[1] - 2 * margin
    lines: list[tuple[str, bool]] = []  # (text, is_colored)
    for i, raw in enumerate(raw_lines):
        for wrapped in _wrap_to_width(raw, max_text_width):
            lines.append((wrapped, i in colored_line_indices))

    (_, text_h), baseline = cv2.getTextSize("Ag", _FONT, _SCALE, _THICKNESS)
    line_step = text_h + baseline + 8
    panel_h = line_step * len(lines) + 10
    panel = np.full((panel_h, frame.shape[1], 3), (25, 25, 25), dtype=frame.dtype)
    y = line_step
    for text, is_colored in lines:
        color = highlight_color if is_colored else (0, 255, 255)
        cv2.putText(panel, text, (margin, y), _FONT, _SCALE, color, _THICKNESS, cv2.LINE_AA)
        y += line_step

    return cv2.vconcat([panel, frame]), panel_h


def _draw(frame, state: _MouseState, classes: list[str], image_idx: int, total: int, split: str, filename: str):
    out = frame.copy()
    n = len(classes)
    for cls, x1, y1, x2, y2 in state.boxes:
        color = _class_color(cls, n)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = classes[cls] if 0 <= cls < n else str(cls)
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, max(th, y1 - 4)), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if state.drag_start is not None and state.drag_end is not None:
        color = _class_color(state.current_class, n)
        cv2.rectangle(out, state.drag_start, state.drag_end, color, 1, cv2.LINE_AA)

    current_name = classes[state.current_class] if 0 <= state.current_class < n else "?"
    raw_lines = [
        f"[{split}] image {image_idx + 1}/{total}: {filename}   ({len(state.boxes)} box(es))",
        f"armed class: {state.current_class} = {current_name}",
        "  ".join(f"{_class_key_label(i)}={c}" for i, c in enumerate(classes)),
        "LMB-drag=draw box   RMB=delete box under cursor   U=undo last   C=clear all",
        "N/SPACE=next image   P=prev image   ESC=quit (auto-saves on every edit)",
    ]
    return _render_panel(out, raw_lines, {1}, _class_color(state.current_class, n))


# ---------------------------------------------------------------------------
# Keypoint mode (--mode keypoints): block_7x7 corner-pose dataset for
# steps 1/2's hole-position check (see hole_position.py).
# ---------------------------------------------------------------------------

Corners = list[tuple[float, float] | None]  # exactly 4, pixel coords, None = marked not-visible


def _find_block_box(main_labels_dir: Path, stem: str, block_class_id: int) -> tuple[float, float, float, float] | None:
    """The block_7x7 box (normalized xc,yc,w,h), reused as-is from the MAIN
    dataset's label for this image rather than redrawn -- returns None if
    this image has no block_7x7 box there at all."""
    label_path = main_labels_dir / f"{stem}.txt"
    if not label_path.exists():
        return None
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != block_class_id:
            continue
        xc, yc, w, h = (float(v) for v in parts[1:])
        return (xc, yc, w, h)
    return None


class _KeypointState:
    def __init__(self):
        self.corners: Corners = [None, None, None, None]
        self.decided: list[bool] = [False, False, False, False]
        self.current_idx = 0
        self.dirty = False

    def all_decided(self) -> bool:
        return all(self.decided)

    def advance_to_next_undecided(self) -> None:
        for offset in range(1, 5):
            idx = (self.current_idx + offset) % 4
            if not self.decided[idx]:
                self.current_idx = idx
                return


def _pose_label_line(block_box_norm: tuple[float, float, float, float], corners: Corners, img_w: int, img_h: int) -> str:
    # Single-class dataset (just block_7x7) -- always class 0 here,
    # independent of block_7x7's index in the main 11-class list.
    xc, yc, w, h = block_box_norm
    parts = ["0", f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for pt in corners:
        if pt is None:
            parts += ["0.000000", "0.000000", "0"]  # v=0: not labeled (ultralytics convention)
        else:
            px, py = pt
            parts += [f"{px / img_w:.6f}", f"{py / img_h:.6f}", "2"]  # v=2: labeled and visible
    return " ".join(parts)


def _save_pose_label(
    pose_label_path: Path, block_box_norm: tuple[float, float, float, float], state: _KeypointState, img_w: int, img_h: int
) -> None:
    pose_label_path.parent.mkdir(parents=True, exist_ok=True)
    pose_label_path.write_text(_pose_label_line(block_box_norm, state.corners, img_w, img_h) + "\n", encoding="utf-8")


def _load_pose_label(pose_label_path: Path, img_w: int, img_h: int) -> tuple[Corners, list[bool]] | None:
    if not pose_label_path.exists():
        return None
    line = pose_label_path.read_text(encoding="utf-8").strip()
    parts = line.split()
    if len(parts) != 5 + 4 * 3:
        return None
    kp = parts[5:]
    corners: Corners = []
    for i in range(4):
        x, y, v = float(kp[i * 3]), float(kp[i * 3 + 1]), float(kp[i * 3 + 2])
        corners.append(None if v == 0 else (x * img_w, y * img_h))
    return corners, [True, True, True, True]


def _make_keypoint_mouse_callback(state: _KeypointState, panel_h: int, img_w: int, img_h: int):
    def _clamp(x, y):
        return max(0, min(img_w, x)), max(0, min(img_h, y - panel_h))

    def _on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < panel_h:
                return
            px, py = _clamp(x, y)
            state.corners[state.current_idx] = (float(px), float(py))
            state.decided[state.current_idx] = True
            state.dirty = True
            state.advance_to_next_undecided()

    return _on_mouse


def _draw_keypoints(frame, state: _KeypointState, block_box_norm, image_idx: int, total: int, split: str, filename: str):
    out = frame.copy()
    img_h, img_w = frame.shape[:2]

    if block_box_norm is not None:
        xc, yc, w, h = block_box_norm
        x1, y1 = int((xc - w / 2) * img_w), int((yc - h / 2) * img_h)
        x2, y2 = int((xc + w / 2) * img_w), int((yc + h / 2) * img_h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (120, 120, 120), 1)

    for i, pt in enumerate(state.corners):
        color = _class_color(i, 4)
        if pt is not None:
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(out, (px, py), 8, color, -1)
            cv2.putText(out, str(i + 1), (px + 10, py - 10), _FONT, 0.6, color, 2, cv2.LINE_AA)
        elif state.decided[i]:
            # Skipped/occluded corners have no click position to draw a dot
            # at -- that's the point of skipping them -- so confirm it some
            # other way: a small "N not visible" badge, top-left of the
            # frame, one per skipped corner. Without this, pressing S looks
            # like nothing happened (the earlier bug this fixes).
            slot = sum(1 for j in range(i) if state.decided[j] and state.corners[j] is None)
            bx, by = 10, 40 + slot * 26
            cv2.putText(out, f"{i + 1} not visible", (bx, by), _FONT, 0.55, color, 2, cv2.LINE_AA)

    def _status(i: int) -> str:
        if not state.decided[i]:
            return "?"
        return "set" if state.corners[i] is not None else "SKIPPED"

    current_name = CORNER_NAMES[state.current_idx]
    n_done = sum(state.decided)
    raw_lines = [
        f"[{split}] image {image_idx + 1}/{total}: {filename}   ({n_done}/4 corners decided)",
        f"click corner: {state.current_idx + 1} = {current_name}",
        "  ".join(f"{i + 1}={name}[{_status(i)}]" for i, name in enumerate(CORNER_NAMES)),
        "LMB=set current corner   1-4=jump to corner   S=mark not visible/occluded   U=undo current   C=clear all",
        "N/SPACE=next image (needs all 4 decided)   P=prev image   ESC=quit",
    ]
    return _render_panel(out, raw_lines, {1}, _class_color(state.current_idx, 4))


def run_keypoints(args: argparse.Namespace) -> None:
    """Unlike run() (box mode), a pose label is only ever written to disk
    once all 4 corners are decided (set or explicitly skipped) -- the
    ultralytics pose row format has no way to represent "corner not yet
    decided," so there's no safe partial state to persist mid-edit the way
    box mode persists "zero boxes so far." Leaving an image incomplete
    (P or ESC before finishing) simply doesn't save that visit; a
    previously-completed file on disk is never overwritten with partial data.
    """
    cfg = load_config(args.app_config)
    block_class_id = cfg.classes.index("block_7x7")

    main_images_dir = Path(args.dataset_root) / "images" / args.split
    main_labels_dir = Path(args.dataset_root) / "labels" / args.split
    pose_images_dir = Path(args.pose_dataset_root) / "images" / args.split
    pose_labels_dir = Path(args.pose_dataset_root) / "labels" / args.split

    all_images = sorted(main_images_dir.glob("*.jpg"))
    if not all_images:
        raise SystemExit(f"no images found under {main_images_dir} — run split_sessions.py first")

    # Only images that already have a block_7x7 box in the main dataset --
    # that box is reused as-is for the pose label, and there's nothing to
    # corner-label on an image that doesn't contain the block at all.
    candidates = [p for p in all_images if _find_block_box(main_labels_dir, p.stem, block_class_id) is not None]
    if not candidates:
        raise SystemExit(f"no images in split '{args.split}' have a block_7x7 box in the main dataset labels")

    if args.indices:
        requested = [int(tok.strip()) for tok in args.indices.split(",") if tok.strip()]
        valid_max = len(candidates) - 1
        image_paths = []
        for i in requested:
            if 0 <= i <= valid_max:
                image_paths.append(candidates[i])
            else:
                print(f"[label_tool] WARNING: index {i} out of range (0-{valid_max}) for "
                      f"{len(candidates)} block_7x7-containing images in split '{args.split}' — skipping it")
        if not image_paths:
            raise SystemExit("no valid indices left after filtering — nothing to show")
        idx = max(0, min(len(image_paths) - 1, args.start_at)) if args.start_at is not None else 0
    else:
        image_paths = candidates
        if args.start_at is not None:
            idx = max(0, min(len(image_paths) - 1, args.start_at))
        else:
            unlabeled = [i for i, p in enumerate(image_paths) if not (pose_labels_dir / f"{p.stem}.txt").exists()]
            idx = unlabeled[0] if unlabeled else 0
            if unlabeled:
                print(f"[label_tool] resuming at image {idx + 1}/{len(image_paths)} "
                      f"({len(image_paths) - len(unlabeled)} already visited)")
            else:
                print("[label_tool] every block_7x7 image in this split already has a pose label — starting from the top for review")

    print(f"[label_tool] keypoint mode: {len(image_paths)} images with block_7x7 in split '{args.split}'")

    window = "assembly_trainer label_tool (keypoints)"
    cv2.namedWindow(window)

    while True:
        image_path = image_paths[idx]
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[label_tool] WARNING: could not read {image_path}, skipping")
            idx = min(len(image_paths) - 1, idx + 1)
            continue
        img_h, img_w = frame.shape[:2]

        block_box_norm = _find_block_box(main_labels_dir, image_path.stem, block_class_id)
        pose_label_path = pose_labels_dir / f"{image_path.stem}.txt"

        state = _KeypointState()
        loaded = _load_pose_label(pose_label_path, img_w, img_h)
        if loaded is not None:
            state.corners, state.decided = loaded

        panel_h_ref = [0]
        cv2.setMouseCallback(window, _make_keypoint_mouse_callback(state, 0, img_w, img_h))

        advance: str | None = None
        while advance is None:
            composite, panel_h = _draw_keypoints(frame, state, block_box_norm, idx, len(image_paths), args.split, image_path.name)
            if panel_h != panel_h_ref[0]:
                panel_h_ref[0] = panel_h
                cv2.setMouseCallback(window, _make_keypoint_mouse_callback(state, panel_h, img_w, img_h))
            cv2.imshow(window, composite)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return
            elif key in (ord("n"), ord("N"), 32):  # N or SPACE
                if state.all_decided():
                    advance = "next"
                else:
                    print(f"[label_tool] {4 - sum(state.decided)} corner(s) still undecided — "
                          "click them or press S to mark not-visible before moving on")
            elif key in (ord("p"), ord("P")):
                advance = "prev"
            elif key in (ord("s"), ord("S")):
                state.corners[state.current_idx] = None
                state.decided[state.current_idx] = True
                state.dirty = True
                state.advance_to_next_undecided()
            elif key in (ord("u"), ord("U")):
                state.corners[state.current_idx] = None
                state.decided[state.current_idx] = False
                state.dirty = True
            elif key in (ord("c"), ord("C")):
                state.corners = [None, None, None, None]
                state.decided = [False, False, False, False]
                state.current_idx = 0
                state.dirty = True
            elif ord("1") <= key <= ord("4"):
                state.current_idx = key - ord("1")

        if state.all_decided():
            pose_images_dir.mkdir(parents=True, exist_ok=True)
            dest_img = pose_images_dir / image_path.name
            if not dest_img.exists():
                shutil.copy2(image_path, dest_img)
            _save_pose_label(pose_label_path, block_box_norm, state, img_w, img_h)

        idx = min(len(image_paths) - 1, idx + 1) if advance == "next" else max(0, idx - 1)

    cv2.destroyAllWindows()


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.app_config)
    key_to_class = {_class_key_label(i): i for i in range(len(cfg.classes))}
    images_dir = Path(args.dataset_root) / "images" / args.split
    labels_dir = Path(args.dataset_root) / "labels" / args.split

    image_paths_all = sorted(images_dir.glob("*.jpg"))
    if not image_paths_all:
        raise SystemExit(f"no images found under {images_dir} — run split_sessions.py first")

    if args.indices:
        requested = [int(tok.strip()) for tok in args.indices.split(",") if tok.strip()]
        valid_max = len(image_paths_all) - 1
        image_paths = []
        for i in requested:
            if 0 <= i <= valid_max:
                image_paths.append(image_paths_all[i])
            else:
                print(f"[label_tool] WARNING: index {i} is out of range (0-{valid_max}) "
                      f"for split '{args.split}' ({len(image_paths_all)} images) — skipping it")
        if not image_paths:
            raise SystemExit("no valid indices left after filtering — nothing to show")
        print(f"[label_tool] custom list: {len(image_paths)}/{len(requested)} requested indices are valid, "
              "navigating N/P through just this list in the order given")
        idx = max(0, min(len(image_paths) - 1, args.start_at)) if args.start_at is not None else 0
    else:
        image_paths = image_paths_all
        if args.start_at is not None:
            idx = max(0, min(len(image_paths) - 1, args.start_at))
        else:
            # Auto-resume: first image with no label file yet (a visited image
            # always gets one written, even if empty — see the advance handling
            # below), so this naturally picks up where you left off.
            unlabeled = [
                i for i, p in enumerate(image_paths) if not (labels_dir / f"{p.stem}.txt").exists()
            ]
            idx = unlabeled[0] if unlabeled else 0
            if unlabeled:
                print(f"[label_tool] resuming at image {idx + 1}/{len(image_paths)} "
                      f"({len(image_paths) - len(unlabeled)} already visited)")
            else:
                print("[label_tool] every image in this split already has a label file — starting from the top for review")

    window = "assembly_trainer label_tool"
    cv2.namedWindow(window)

    while True:
        image_path = image_paths[idx]
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[label_tool] WARNING: could not read {image_path}, skipping")
            idx = min(len(image_paths) - 1, idx + 1)
            continue
        img_h, img_w = frame.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"

        state = _MouseState()
        state.boxes = _load_boxes(label_path, img_w, img_h)

        panel_h_ref = [0]
        cv2.setMouseCallback(window, _make_mouse_callback(state, 0, img_w, img_h))

        advance: str | None = None
        while advance is None:
            composite, panel_h = _draw(frame, state, cfg.classes, idx, len(image_paths), args.split, image_path.name)
            if panel_h != panel_h_ref[0]:
                panel_h_ref[0] = panel_h
                cv2.setMouseCallback(window, _make_mouse_callback(state, panel_h, img_w, img_h))
            cv2.imshow(window, composite)

            if state.dirty:
                _save_boxes(label_path, state.boxes, img_w, img_h)
                state.dirty = False

            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return
            elif key in (ord("n"), ord("N"), 32):  # N or SPACE
                advance = "next"
            elif key in (ord("p"), ord("P")):
                advance = "prev"
            elif key in (ord("u"), ord("U")):
                if state.boxes:
                    state.boxes.pop()
                    state.dirty = True
            elif key in (ord("c"), ord("C")):
                if state.boxes:
                    state.boxes = []
                    state.dirty = True
            elif 32 <= key < 127 and chr(key) in key_to_class:
                state.current_class = key_to_class[chr(key)]

        # Always persist on leaving an image, even with zero boxes and even
        # if nothing changed this visit — a missing label file means
        # "not visited yet" for auto-resume, so a genuinely-empty image
        # still needs its (empty) file written the first time you pass it.
        if not label_path.exists() or state.dirty:
            _save_boxes(label_path, state.boxes, img_w, img_h)
            state.dirty = False

        idx = min(len(image_paths) - 1, idx + 1) if advance == "next" else max(0, idx - 1)

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default="data/dataset")
    parser.add_argument("--app-config", default="config/default_config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--mode", choices=("boxes", "keypoints"), default="boxes",
                         help="'boxes' (default): main multi-class dataset. 'keypoints': block_7x7 "
                              "corner-pose dataset for steps 1/2's hole-position check (see hole_position.py)")
    parser.add_argument("--pose-dataset-root", default="data/dataset_pose",
                         help="--mode keypoints only: where the pose dataset's images/labels are written "
                              "(separate tree from --dataset-root -- the pose label format is incompatible "
                              "with the main dataset's plain-box format)")
    parser.add_argument("--start-at", type=int, default=None,
                         help="0-based image index to start at (default: auto-resume at the first "
                              "not-yet-labeled image); with --indices, an index into that list instead")
    parser.add_argument("--indices", default=None,
                         help="comma-separated list of 0-based image indices to visit, in that exact "
                              "order, instead of the whole split (e.g. '20,27,45') — N/P navigate "
                              "through just this list")
    args = parser.parse_args()
    if args.mode == "keypoints":
        run_keypoints(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
