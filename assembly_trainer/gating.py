"""Per-step class gate + 3-way per-frame classification (build plan pipeline
diagram, §1 "Three-way classification outcome").

This is the stage between the detector/tracker and the stability window: for
one inference cycle, decide whether a required class's presence (or, for the
motor-wiring step, a connector's loose/connected state) reads as CORRECT /
INCORRECT / UNCERTAIN, restricted to detections that fall inside the current
placement ROI (§4 closing note) so a part still sitting in a parts bin
doesn't count.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .detector import Detection
from .hole_position import Homography, nearest_hole
from .roi import Bbox, bbox_center, center_inside


class Outcome(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"


@dataclass
class FrameGateReading:
    outcome: Outcome
    confident_count: int
    ambiguous_count: int
    empty: bool  # nothing of the watched class(es) seen in the ROI at all


def _in_roi(detections: list[Detection], class_name: str, roi: Bbox) -> list[Detection]:
    return [d for d in detections if d.class_name == class_name and center_inside(d.bbox, roi)]


def evaluate_class_count(
    detections: list[Detection],
    class_name: str,
    roi: Bbox,
    confidence_threshold: float,
    required_count: int,
) -> FrameGateReading:
    """3-way read for "is `required_count` of `class_name` present in `roi`?".

    - CORRECT: enough confident (>= threshold) detections to meet the count.
    - UNCERTAIN: not enough confident detections alone, but low-confidence
      detections exist that *could* close the gap — don't let a borderline
      detection count for or against progress.
    - INCORRECT: confidently not enough, even generously counting the
      low-confidence detections.
    """
    dets = _in_roi(detections, class_name, roi)
    confident = [d for d in dets if d.confidence >= confidence_threshold]
    ambiguous = [d for d in dets if d.confidence < confidence_threshold]

    if len(confident) >= required_count:
        outcome = Outcome.CORRECT
    elif ambiguous and (len(confident) + len(ambiguous)) >= required_count:
        outcome = Outcome.UNCERTAIN
    else:
        outcome = Outcome.INCORRECT

    return FrameGateReading(
        outcome=outcome,
        confident_count=len(confident),
        ambiguous_count=len(ambiguous),
        empty=not dets,
    )


def evaluate_hole_position(
    detections: list[Detection],
    class_name: str,
    roi: Bbox,
    confidence_threshold: float,
    target_hole: str,
    hole_template: dict[str, list[dict]],
    homography: Homography | None,
) -> FrameGateReading:
    """3-way read of "is a confident `class_name` detection sitting in the
    `target_hole` slot?" -- steps 1/2 only (see hole_position.py).

    UNCERTAIN whenever the block's corners weren't confidently visible this
    frame (no homography) or no confident peg is in the ROI at all -- same
    "don't count an ambiguous frame for or against" philosophy as
    evaluate_class_count. Only a confidently-placed peg that projects to a
    *different* known hole reads as INCORRECT (a real, positive "wrong
    hole" signal, not just absence).
    """
    dets = [d for d in _in_roi(detections, class_name, roi) if d.confidence >= confidence_threshold]
    if homography is None or not dets:
        return FrameGateReading(outcome=Outcome.UNCERTAIN, confident_count=0, ambiguous_count=0, empty=not dets)

    for d in dets:
        canonical_point = homography.project(bbox_center(d.bbox))
        hole = nearest_hole(canonical_point, hole_template)
        if hole == target_hole:
            return FrameGateReading(outcome=Outcome.CORRECT, confident_count=1, ambiguous_count=0, empty=False)

    return FrameGateReading(outcome=Outcome.INCORRECT, confident_count=0, ambiguous_count=0, empty=False)


class Reason(str, Enum):
    """Why a required class isn't satisfied yet, for trainee-facing UI
    feedback. NOT_PRESENT/WRONG_ORIENTATION/WRONG_PART/UNCERTAIN are purely
    diagnostic (advancement stays on evaluate_class_count + the stability
    window, unaffected) -- WRONG_HOLE is the one exception: for steps with
    target_hole set (1/2 only), it reflects evaluate_hole_position's own
    stability-windowed outcome, which DOES gate advancement (see
    state_machine.py) -- a peg in the wrong hole must not let the step
    complete just because the count matched."""

    SATISFIED = "satisfied"
    NOT_PRESENT = "not_present"
    WRONG_ORIENTATION = "wrong_orientation"
    WRONG_PART = "wrong_part"
    WRONG_HOLE = "wrong_hole"
    UNCERTAIN = "uncertain"


@dataclass
class ClassDiagnosis:
    reason: Reason
    # the unexpected class name found (WRONG_PART), or the hole the part was
    # actually detected in (WRONG_HOLE) -- None otherwise
    detail: str | None = None


def step_orientation_variant(step_id: int, known_classes: set[str]) -> str | None:
    """`step<N>_wrong_orientation`, by naming convention — only "real" if
    it's actually a trained class (only some steps have a part with a
    meaningful wrong orientation; see config.yaml's classes list). Scoped to
    the step rather than a specific part: each step in this system only ever
    introduces one orientation-sensitive part, so a step-level class is
    equivalent in practice, but note it means e.g. step 1 and step 2 (which
    both place a `short_peg`) get separately-labeled classes rather than
    sharing one — a deliberate choice, not a data-efficiency win."""
    variant = f"step{step_id}_wrong_orientation"
    return variant if variant in known_classes else None


# Classes that are physically always around regardless of build progress
# (e.g. long_peg sits near/on the block whether or not the trainee has
# placed the real short_peg yet) -- purely training negatives so the model
# learns to name them instead of guessing among the peg-family classes, not
# a signal that anything's actually wrong. Never surfaced as WRONG_PART.
_IGNORED_FOR_WRONG_PART = {"long_peg"}


def diagnose_class(
    detections: list[Detection],
    class_name: str,
    roi: Bbox,
    confidence_threshold: float,
    required_count: int,
    expected_classes: set[str],
    orientation_variant: str | None,
    target_hole: str | None = None,
    hole_template: dict[str, list[dict]] | None = None,
    homography: Homography | None = None,
) -> ClassDiagnosis:
    """Explain *why* `class_name` isn't satisfied yet: missing entirely, the
    step's wrong-oriented part detected instead, some other unexpected class
    sitting where it should be, in the wrong hole, or just too uncertain to
    call.

    `expected_classes` should be every class legitimately expected in this
    ROI for the current step (i.e. the step's own `requires` keys plus its
    orientation variant, if any) — anything else confidently detected there
    counts as "wrong part", except _IGNORED_FOR_WRONG_PART (classes that are
    just always physically present regardless of progress). `orientation_variant` is this step's
    `step<N>_wrong_orientation` class name (see `step_orientation_variant`),
    or None if this step has no orientation-sensitive part. `target_hole` is
    only set for steps 1/2 (see hole_position.py) — when set, a count-correct
    peg that's confidently in a *different* known hole is diagnosed as
    WRONG_HOLE instead of SATISFIED (and this is the one diagnosis reason
    that also gates advancement — see state_machine.py).
    """
    reading = evaluate_class_count(detections, class_name, roi, confidence_threshold, required_count)
    if reading.outcome == Outcome.CORRECT:
        if target_hole is not None:
            hole_reading = evaluate_hole_position(
                detections, class_name, roi, confidence_threshold, target_hole, hole_template or {}, homography
            )
            if hole_reading.outcome == Outcome.INCORRECT:
                dets = [d for d in _in_roi(detections, class_name, roi) if d.confidence >= confidence_threshold]
                detected_hole = None
                if homography is not None and dets:
                    detected_hole = nearest_hole(homography.project(bbox_center(dets[0].bbox)), hole_template or {})
                return ClassDiagnosis(Reason.WRONG_HOLE, detail=detected_hole)
            if hole_reading.outcome == Outcome.UNCERTAIN:
                return ClassDiagnosis(Reason.UNCERTAIN)
        return ClassDiagnosis(Reason.SATISFIED)

    if orientation_variant:
        variant_dets = [
            d for d in detections
            if d.class_name == orientation_variant and d.confidence >= confidence_threshold and center_inside(d.bbox, roi)
        ]
        if variant_dets:
            return ClassDiagnosis(Reason.WRONG_ORIENTATION)

    for d in detections:
        if d.confidence < confidence_threshold or not center_inside(d.bbox, roi):
            continue
        if d.class_name == class_name or d.class_name in expected_classes or d.class_name in _IGNORED_FOR_WRONG_PART:
            continue
        return ClassDiagnosis(Reason.WRONG_PART, detail=d.class_name)

    if reading.outcome == Outcome.UNCERTAIN:
        return ClassDiagnosis(Reason.UNCERTAIN)

    return ClassDiagnosis(Reason.NOT_PRESENT)
