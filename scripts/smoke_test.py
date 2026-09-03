"""Headless smoke test of the full pipeline (build plan build order item 11).

No camera, no trained model, no network — pure Python, driven by
`MockDetector` through an injectable fake clock so a 90-second escalation
timeout runs in milliseconds of wall time. Exercises:

  1. A full walkthrough of every configured step (config -> gating ->
     stability -> state machine -> escalation), asserting steps advance in
     strict order and the run completes.
  2. The motor-wiring step's anti-gaming reset-between-confirmations rule
     in isolation.
  3. Escalation tier progression/stacking and the one-shot trainer-alert
     flag, plus the timer resetting when a step actually advances.
  4. Per-class diagnosis (not_present / wrong_orientation / wrong_part /
     satisfied) used for the trainee-facing "what's wrong" UI hints.
  5. "Start again" reset (pipeline.reset()).
  6. Hole-position gating for steps with target_hole set (steps 1/2) --
     correct-hole advancement, wrong-hole blocking advancement even when
     count matches, and uncertain-on-missing-corners not advancing either.

Run: python scripts/smoke_test.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from assembly_trainer.config import Config, load_config
from assembly_trainer.detector import Detection
from assembly_trainer.escalation import Tier
from assembly_trainer.gating import step_orientation_variant
from assembly_trainer.mock_detector import MockDetector, Script, build_full_walkthrough_script
from assembly_trainer.pipeline import Pipeline, TrainerFlag
from assembly_trainer.state_machine import StepStateMachine

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL: {message}")
    else:
        print(f"  ok:   {message}")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_frame_source(width: int, height: int):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    return lambda: frame


def tick_n(pipeline: Pipeline, clock: FakeClock, cfg: Config, n: int):
    interval_s = cfg.app.inference_interval_ms / 1000.0
    last = None
    for _ in range(n):
        last = pipeline.step()
        clock.advance(interval_s)
    return last


def single_step_config(cfg: Config, step_id: int) -> Config:
    """Isolate one step as a 1-step config, id renumbered to 1, so a test
    can target it directly without walking through every earlier step."""
    step = copy.deepcopy(cfg.get_step(step_id))
    step.id = 1
    new_cfg = copy.deepcopy(cfg)
    new_cfg.steps = [step]
    return new_cfg


def test_full_walkthrough() -> None:
    cfg = load_config("config/default_config.yaml")
    print(f"\n[1] full {len(cfg.steps)}-step walkthrough")
    clock = FakeClock()
    frame_w, frame_h = cfg.camera.width, cfg.camera.height

    script = build_full_walkthrough_script(cfg, frame_w, frame_h, hold_seconds=8.0)
    detector = MockDetector(script, clock=clock.now)
    pipeline = Pipeline(cfg, detector, make_frame_source(frame_w, frame_h), clock=clock.now)

    total_seconds = sum(hold for hold, _ in script)
    n_ticks = int(total_seconds / (cfg.app.inference_interval_ms / 1000.0))

    seen_step_ids: list[int] = []
    max_step_seen = 0
    for _ in range(n_ticks):
        result = pipeline.step()
        clock.advance(cfg.app.inference_interval_ms / 1000.0)
        sid = result.frame_result.step_id
        if sid is not None and sid != (seen_step_ids[-1] if seen_step_ids else None):
            seen_step_ids.append(sid)
        if sid is not None:
            max_step_seen = max(max_step_seen, sid)
        if result.frame_result.completed:
            break

    last_step_id = cfg.steps[-1].id
    check(pipeline.state_machine.completed, "state machine reaches 'completed' by end of walkthrough")
    check(seen_step_ids == sorted(set(seen_step_ids)), f"step ids advanced in strictly increasing order: {seen_step_ids}")
    check(max_step_seen == last_step_id, f"walkthrough reached the last step ({last_step_id}) (saw up to step {max_step_seen})")
    check(len(pipeline.flags) == 0, "no trainer-alert flags fired during a smoothly-progressing run")


def test_escalation_tiers() -> None:
    print("\n[2] escalation tier progression, stacking, and reset-on-advance")
    base_cfg = load_config("config/default_config.yaml")
    cfg = single_step_config(base_cfg, 1)  # step 1's actual requires, whatever the config currently says
    step1_requires = cfg.steps[0].requires
    esc = cfg.escalation
    clock = FakeClock()
    frame_w, frame_h = cfg.camera.width, cfg.camera.height

    empty_script: Script = [(1e9, [])]
    detector = MockDetector(empty_script, clock=clock.now)
    pipeline = Pipeline(cfg, detector, make_frame_source(frame_w, frame_h), clock=clock.now)

    seen_tiers: list[Tier] = []
    n_ticks = int((esc.tier_3_trainer_alert_after_seconds + 2) / (cfg.app.inference_interval_ms / 1000.0))
    for _ in range(n_ticks):
        result = pipeline.step()
        clock.advance(cfg.app.inference_interval_ms / 1000.0)
        if not seen_tiers or result.tier != seen_tiers[-1]:
            seen_tiers.append(result.tier)

    check(
        seen_tiers == [Tier.NONE, Tier.REFERENCE_IMAGE, Tier.REFERENCE_VIDEO, Tier.TRAINER_ALERT],
        f"tiers escalate in strict stacking order over a stalled step: {[t.name for t in seen_tiers]}",
    )
    check(len(pipeline.flags) == 1, f"trainer-alert flag fires exactly once (one-shot), got {len(pipeline.flags)}")

    # Now satisfy the step (using whatever step 1 currently requires) and
    # confirm the timer resets on the *next* step.
    satisfying_detections = [
        Detection(cls, 0.9, (500 + i * 60, 500, 550 + i * 60, 550))
        for cls, count in step1_requires.items()
        for i in range(count)
    ]
    detector._script = [(1e9, satisfying_detections)]
    detector._start = clock.now()
    window = cfg.resolve(cfg.steps[0], "stability_window_frames")
    result = tick_n(pipeline, clock, cfg, window + 2)

    check(pipeline.state_machine.completed, "single-step config completes once requirements are stably met")
    # Well below tier_1 (20s) is enough to prove the timer actually reset on
    # advance rather than continuing to climb from the pre-advance stall —
    # not an exact tick count, since the window may stabilize a few ticks
    # early if it still held stale entries from the stall phase.
    check(
        pipeline.escalation.elapsed_seconds < esc.tier_1_reference_image_after_seconds / 2,
        f"escalation timer resets when the step actually advances (elapsed={pipeline.escalation.elapsed_seconds:.2f}s)",
    )


def test_diagnosis() -> None:
    print("\n[3] wrong-part / wrong-orientation diagnosis")
    base_cfg = load_config("config/default_config.yaml")
    cfg = single_step_config(base_cfg, 1)  # {block_7x7: 1, peg: 3}
    frame_w, frame_h = cfg.camera.width, cfg.camera.height
    block = Detection("block_7x7", 0.9, (700, 400, 1000, 700))
    # Steps 1/2 gate on peg COUNT: 2 long pegs are permanently fixtured on
    # the 7x7 block, so step 1 needs 3 and step 2 needs 4.
    fixtures = [
        Detection("peg", 0.9, (720, 420, 750, 470)),
        Detection("peg", 0.9, (960, 420, 990, 470)),
    ]

    result = StepStateMachine(cfg).process_frame([], frame_w, frame_h)
    check(
        result.class_diagnosis["peg"].reason.value == "not_present",
        "empty frame -> peg diagnosed as not_present",
    )

    result = StepStateMachine(cfg).process_frame([block] + fixtures, frame_w, frame_h)
    check(
        not result.class_status["peg"],
        "bare block (2 fixture pegs, none placed) does NOT satisfy step 1's count of 3",
    )

    # ALL step<N>_wrong_orientation classes have been removed from the config
    # class list AND from the training labels, so no step resolves an
    # orientation variant and the WRONG_ORIENTATION diagnosis is unreachable.
    # steps 1/2's version was really about position (now
    # hole_template/target_hole); steps 3/4's was dropped afterwards.
    step3_cfg = copy.deepcopy(base_cfg)
    step3_cfg.steps = [copy.deepcopy(base_cfg.get_step(3))]
    check(
        step_orientation_variant(3, set(step3_cfg.classes)) is None,
        "step3_wrong_orientation no longer in the class list -> no orientation variant resolves",
    )

    unexpected_peecee = Detection("peecee", 0.9, (750, 450, 800, 500))
    result = StepStateMachine(cfg).process_frame([block, unexpected_peecee], frame_w, frame_h)
    diag = result.class_diagnosis["peg"]
    check(
        diag.reason.value == "wrong_part" and diag.detail == "peecee",
        f"unrelated class detected in the peg's spot -> diagnosed as wrong_part (detail={diag.detail!r})",
    )

    placed = Detection("peg", 0.9, (750, 450, 800, 500))
    result = StepStateMachine(cfg).process_frame([block] + fixtures + [placed], frame_w, frame_h)
    check(
        result.class_diagnosis["peg"].reason.value == "satisfied",
        "2 fixture pegs + 1 placed = 3 -> step 1 satisfied",
    )


def test_reset() -> None:
    print("\n[4] 'Start again' reset (pipeline.reset())")
    cfg = load_config("config/default_config.yaml")
    clock = FakeClock()
    frame_w, frame_h = cfg.camera.width, cfg.camera.height

    script = build_full_walkthrough_script(cfg, frame_w, frame_h, hold_seconds=8.0)
    detector = MockDetector(script, clock=clock.now)
    pipeline = Pipeline(cfg, detector, make_frame_source(frame_w, frame_h), clock=clock.now)

    total_seconds = sum(hold for hold, _ in script)
    n_ticks = int(total_seconds / (cfg.app.inference_interval_ms / 1000.0))
    for _ in range(n_ticks):
        result = pipeline.step()
        clock.advance(cfg.app.inference_interval_ms / 1000.0)
        if result.frame_result.completed:
            break
    check(pipeline.state_machine.completed, "walkthrough reaches completed before reset is tested")

    # A stalled-step trainer alert should also be cleared by reset, not just step id.
    pipeline.flags.append(TrainerFlag(step_id=1, step_name="x", at=0.0))

    pipeline.reset()
    check(pipeline.state_machine.current_step_id == cfg.first_step_id(), "reset returns to the first step")
    check(not pipeline.state_machine.completed, "reset clears the completed flag")
    check(pipeline.escalation.elapsed_seconds < 0.01, "reset restarts the escalation timer at zero")
    check(len(pipeline.flags) == 0, "reset clears prior trainer-alert flags")

    # A fresh walkthrough after reset should complete again, proving the
    # internal stability windows/assembly bbox were actually cleared too
    # (not just the step id) — a stale window could otherwise let step 1
    # "complete" on its very first frame post-reset.
    detector._script = script
    detector._start = clock.now()
    for _ in range(n_ticks):
        result = pipeline.step()
        clock.advance(cfg.app.inference_interval_ms / 1000.0)
        if result.frame_result.completed:
            break
    check(pipeline.state_machine.completed, "walkthrough completes again cleanly after reset")


def _hole_position_test_config() -> Config:
    """A single-step config (id renumbered to 1) with target_hole/hole_template
    wired up, isolated from the real dataset/model so this is pure geometry."""
    base_cfg = load_config("config/default_config.yaml")
    cfg = single_step_config(base_cfg, 1)
    # Pure geometry: require a SINGLE peg rather than step 1's real count of
    # 3 (2 block fixtures + 1 placed), so this exercises hole projection
    # alone and not the fixture-baseline counting.
    cfg.steps[0].requires = {"block_7x7": 1, "peg": 1}
    cfg.steps[0].target_hole = "a1"
    cfg.hole_template = {
        "edge_a": [
            {"name": "a1", "x": 0.1, "y": 0.0},
            {"name": "a2", "x": 0.5, "y": 0.0},
            {"name": "a3", "x": 0.9, "y": 0.0},
        ]
    }
    return cfg


def test_hole_position() -> None:
    print("\n[5] hole-position gating (steps with target_hole set)")
    cfg = _hole_position_test_config()
    frame_w, frame_h = cfg.camera.width, cfg.camera.height
    window = cfg.resolve(cfg.steps[0], "stability_window_frames")

    # Axis-aligned square in pixel space, corners in CANONICAL_CORNERS order
    # (top-left, top-right, bottom-right, bottom-left) -- hole a1 (canonical
    # 0.1, 0.0) lands at pixel (730, 400), a3 (0.9, 0.0) at (970, 400).
    corners = [(700.0, 400.0), (1000.0, 400.0), (1000.0, 700.0), (700.0, 700.0)]
    block = Detection("block_7x7", 0.9, (650, 380, 1050, 720))

    def peg_at(px: float, py: float) -> Detection:
        return Detection("peg", 0.9, (px - 15, py - 15, px + 15, py + 15))

    # -- correct hole: peg near a1's projected pixel position --
    sm = StepStateMachine(cfg)
    result = None
    for _ in range(window + 2):
        result = sm.process_frame([block, peg_at(730, 400)], frame_w, frame_h, pose_corners=corners)
    check(sm.completed, "peg confidently in the target hole (a1) -> step completes")

    # -- wrong hole: peg near a3's position instead, count still matches --
    sm = StepStateMachine(cfg)
    result = None
    for _ in range(window + 2):
        result = sm.process_frame([block, peg_at(970, 400)], frame_w, frame_h, pose_corners=corners)
    check(
        not sm.completed and result.class_status["peg"] is False,
        "peg in the wrong hole (a3, not a1) does NOT complete the step even though the count matches",
    )
    check(
        result.class_diagnosis["peg"].reason.value == "wrong_hole"
        and result.class_diagnosis["peg"].detail == "a3",
        f"diagnosis correctly reports wrong_hole with detail='a3' (got {result.class_diagnosis['peg']!r})",
    )

    # -- uncertain: peg count matches but no pose corners this run at all,
    # so there's never a homography to judge position against --
    sm = StepStateMachine(cfg)
    result = None
    for _ in range(window + 2):
        result = sm.process_frame([block, peg_at(730, 400)], frame_w, frame_h, pose_corners=None)
    check(
        not sm.completed and result.class_diagnosis["peg"].reason.value == "uncertain",
        "no pose corners available -> stays uncertain, never falsely completes on count alone",
    )


def test_seated_gate() -> None:
    """Step 4's require_seated check: a PeeCee merely PRESENT in the ROI must
    not complete the step -- it has to be seated on the blocks. Without this,
    the PeeCee already lying beside the assembly during step 3 satisfies step
    4's `peecee: 1` the instant step 3 ends, and seating is never tested."""
    print("\n[6] seated gating (steps with require_seated set)")
    base_cfg = load_config("config/default_config.yaml")
    step4 = next(s for s in base_cfg.steps if s.require_seated)
    cfg = single_step_config(base_cfg, step4.id)
    frame_w, frame_h = cfg.camera.width, cfg.camera.height
    window = cfg.resolve(cfg.steps[0], "stability_window_frames")

    # blocks span x 700..1000, y 400..700
    blocks = [
        Detection("block_7x7", 0.95, (700, 400, 1000, 700)),
        Detection("block_7x14", 0.95, (700, 400, 1000, 500)),
    ]
    # seated: centre x=850 over the blocks, base y=650 inside them, tall box
    seated = Detection("peecee", 0.95, (820, 450, 880, 650))
    # lying flat ON the blocks -- right place, wrong posture
    flat = Detection("peecee", 0.95, (750, 600, 950, 660))
    # upright but OFF to the side -- the future "face-up in step 3" case
    beside = Detection("peecee", 0.95, (1200, 450, 1260, 650))

    def run(dets):
        sm = StepStateMachine(cfg)
        result = None
        for _ in range(window + 2):
            result = sm.process_frame(dets, frame_w, frame_h)
        return sm, result

    sm, _ = run(blocks + [seated])
    check(sm.completed, "PeeCee seated on the blocks -> step completes")

    sm, result = run(blocks + [flat])
    check(
        not sm.completed and result.class_diagnosis["peecee"].reason.value == "not_seated",
        "PeeCee lying flat on the blocks -> NOT complete, diagnosed not_seated",
    )

    sm, result = run(blocks + [beside])
    check(
        not sm.completed and result.class_diagnosis["peecee"].reason.value == "not_seated",
        "PeeCee upright but beside the blocks -> NOT complete (the upright-but-unseated case)",
    )


def main() -> None:
    test_full_walkthrough()
    test_escalation_tiers()
    test_diagnosis()
    test_reset()
    test_hole_position()
    test_seated_gate()

    print()
    if FAILURES:
        print(f"SMOKE TEST: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST: ALL PASSED")


if __name__ == "__main__":
    main()
