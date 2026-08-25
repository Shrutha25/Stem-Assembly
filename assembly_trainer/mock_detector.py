"""Scripted/synthetic `Detector` implementation.

Used by the headless smoke test (deterministic, injectable clock, no camera
or model needed) and as a live-demo fallback when no trained model exists
yet but a real camera does — frame content is ignored, so it can sit behind
a real `ThreadedCamera` and still prove the server/UI plumbing end-to-end.

Also builds the motor-wiring step's connected -> reset -> connected script
(the one step type this can't handle via generic per-class layout, since
its completion condition is a loose/connected *state* read within a fixed
ROI, not a class count).
"""

from __future__ import annotations

import time
from typing import Callable

from .config import Config
from .detector import Detection

Script = list[tuple[float, list[Detection]]]  # (hold_seconds, detections-to-emit)


class MockDetector:
    def __init__(self, script: Script, clock: Callable[[], float] = time.monotonic, loop: bool = False):
        self._script = script
        self._clock = clock
        self._loop = loop
        self._start = clock()

    def infer(self, frame) -> list[Detection]:
        if not self._script:
            return []
        elapsed = self._clock() - self._start
        total = sum(hold for hold, _ in self._script)
        if self._loop and total > 0:
            elapsed = elapsed % total

        t = 0.0
        for hold, detections in self._script:
            t += hold
            if elapsed < t:
                return detections
        return self._script[-1][1]


def _at(class_name: str, x: float, y: float, w: float = 24, h: float | None = None, confidence: float = 0.92) -> Detection:
    h = w if h is None else h
    return Detection(class_name=class_name, confidence=confidence, bbox=(x, y, x + w, y + h))


# Fixed offsets (from the anchor plate's top-left) for every class other than
# block_7x7 itself. block_7x7 is given a bbox big enough to span this whole
# envelope, since physically it's the base plate the pegs attach to and
# everything else stacks around — so the growing-assembly ROI (§4), anchored
# on the plate from step 1, already covers every later step's new-part
# positions without needing a real fixture layout.
_CLASS_OFFSETS: dict[str, tuple[float, float]] = {
    "short_peg": (20, 20),
    "block_7x14": (20, 100),
    "peecee": (20, 160),
}
_ENVELOPE_W, _ENVELOPE_H = 320, 240
_ITEM_SPACING = 26


def _layout_step_detections(requires: dict[str, int], cx: float, cy: float) -> list[Detection]:
    detections = []
    for class_name, count in requires.items():
        if class_name == "block_7x7":
            detections.append(
                Detection("block_7x7", 0.92, (cx - 10, cy - 10, cx + _ENVELOPE_W + 10, cy + _ENVELOPE_H + 10))
            )
            continue
        if class_name == "motor_lead_connected":
            continue  # the motor-wiring step handles this separately, see below
        base_x, base_y = _CLASS_OFFSETS[class_name]
        if class_name == "block_7x14":
            detections.append(_at(class_name, cx + base_x, cy + base_y, w=250, h=30))
        elif class_name == "peecee":
            detections.append(_at(class_name, cx + base_x, cy + base_y, w=60, h=40))
        else:
            for i in range(count):
                detections.append(_at(class_name, cx + base_x + i * _ITEM_SPACING, cy + base_y))
    return detections


def build_full_walkthrough_script(cfg: Config, frame_w: int, frame_h: int, hold_seconds: float = 1.0) -> Script:
    """A synthetic session that satisfies every step in order, including the
    motor-wiring step's lead_1 -> reset -> lead_2 flow. Used by the smoke
    test and as the default `--mock` live-demo script.
    """
    cx, cy = frame_w * 0.4, frame_h * 0.4
    script: Script = []

    for step in cfg.steps:
        if step.type == "sequential_roi":
            continue  # handled separately below
        script.append((hold_seconds, _layout_step_detections(step.requires, cx, cy)))

    wiring_step = next((s for s in cfg.steps if s.type == "sequential_roi"), None)
    if wiring_step is not None:
        base_requires = {k: v for k, v in wiring_step.requires.items() if k != "motor_lead_connected"}
        base_parts = _layout_step_detections(base_requires, cx, cy)
        peecee = next(d for d in base_parts if d.class_name == "peecee")
        px1, py1, px2, py2 = peecee.bbox

        # lead 1: connected
        script.append((hold_seconds, base_parts + [_at("motor_lead_connected", px1, py1, w=(px2 - px1), h=(py2 - py1))]))
        # reset: zone reads empty (motor_lead_loose was removed -- see config
        # comment; "not confidently connected" is all the reset check needs)
        script.append((hold_seconds, base_parts))
        # lead 2: connected
        script.append((hold_seconds, base_parts + [_at("motor_lead_connected", px1, py1, w=(px2 - px1), h=(py2 - py1))]))

    return script
