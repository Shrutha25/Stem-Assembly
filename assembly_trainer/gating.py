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
from .roi import Bbox, bbox_center, center_inside, union_bbox


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


# Which classes count as "the thing something can be seated ON" for
# evaluate_seated. The two structural blocks -- their union is the platform.
_SEAT_BASE_CLASSES = ("block_7x7", "block_7x14")

# A seated part stands upright ON the blocks, so its box is taller relative to
# its width. Detection bboxes are in PIXELS, so this ratio is in pixel space --
# do NOT derive it from normalized YOLO label w/h, where a 16:9 frame distorts
# it by H/W (0.5625 at 1920x1080) and a threshold looks ~1.8x larger than it is.
#
# Swept over 193 labelled frames (88 seated / 105 unseated), ground truth taken
# from the capture session. Thresholds 0.80-0.95 all give 100% recall with ZERO
# false positives; 0.875 is the midpoint of that plateau, chosen for margin on
# both sides. Recall falls away above 1.00 (88.6% at 1.05) and false positives
# appear below 0.80 (2.9% at 0.70).
#
# This is a SECONDARY filter only, and deliberately so. Aspect ratio is really
# orientation in disguise, and orientation is a proxy for seating that breaks:
# a PeeCee stood upright on the table but NOT on the blocks has a seated-like
# aspect. The positional tests below are what actually reject that case, which
# is why all three are ANDed rather than scored together.
#
# CAVEAT: these frames contain only FACE-DOWN unseated examples. The
# upright-but-unseated case is not represented yet -- capture it as its own
# session and re-run the sweep before trusting this against it. The positional
# terms are what should catch it; aspect alone would not.
_SEATED_MIN_ASPECT = 0.875


def evaluate_seated(
    detections: list[Detection],
    class_name: str,
    roi: Bbox,
    confidence_threshold: float,
) -> FrameGateReading:
    """3-way read of "is a confident `class_name` detection SEATED on the
    blocks?" -- i.e. resting on the assembly rather than lying beside it.

    Seating is a spatial relation, not an appearance: a tight crop of the
    part looks the same on the block or next to it, so no amount of training
    data can make it a learnable class (the same reason steps 1/2's
    "wrong orientation" classes were replaced by hole geometry -- see
    hole_position.py and the classes comment in default_config.yaml). It is
    therefore decided here, from the detected boxes.

    A detection counts as seated when all three hold:
      - its centre is horizontally within the blocks' span
      - its base (bottom edge) falls within the blocks' vertical span
      - its box is at least `_SEATED_MIN_ASPECT` times taller than wide

    UNCERTAIN when there is nothing to judge -- no confident detection in the
    ROI, or no block visible to be seated on -- so an ambiguous frame counts
    neither for nor against, matching evaluate_class_count/hole_position.
    """
    dets = [d for d in _in_roi(detections, class_name, roi) if d.confidence >= confidence_threshold]
    blocks = [
        d.bbox for d in detections
        if d.class_name in _SEAT_BASE_CLASSES and d.confidence >= confidence_threshold
    ]
    if not dets or not blocks:
        return FrameGateReading(outcome=Outcome.UNCERTAIN, confident_count=0, ambiguous_count=0, empty=not dets)

    base = union_bbox(blocks)
    if base is None:
        return FrameGateReading(outcome=Outcome.UNCERTAIN, confident_count=0, ambiguous_count=0, empty=False)
    bx1, by1, bx2, by2 = base

    for d in dets:
        x1, y1, x2, y2 = d.bbox
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        centre_x = (x1 + x2) / 2
        if not (bx1 <= centre_x <= bx2):
            continue          # sitting off to one side of the blocks
        if not (by1 <= y2 <= by2):
            continue          # base isn't resting within the blocks' span
        if (height / width) < _SEATED_MIN_ASPECT:
            continue          # lying flat rather than standing on them
        return FrameGateReading(outcome=Outcome.CORRECT, confident_count=1, ambiguous_count=0, empty=False)

    # Something is confidently there and the blocks are visible, but nothing
    # is actually seated -- a real "not seated yet" signal, not mere absence.
    return FrameGateReading(outcome=Outcome.INCORRECT, confident_count=0, ambiguous_count=0, empty=False)


# Max distance between two parts' box centres for them to count as ATTACHED,
# in units of the smaller box's diagonal. Measured over 478 labelled frames
# containing both blocks -- all of them attached, since the 7x14 only ever
# appears in the build once it is on the 7x7: centre distance ranged
# 0.214-0.980 (median 0.443). 1.20 clears the observed maximum with headroom,
# so every genuine attachment passes.
#
# Centre distance rather than box overlap: overlap is unreliable here. Among
# those same attached frames it ranged 0.000-1.000 (p05 0.148) -- from some
# camera angles two genuinely-attached blocks have boxes that do not intersect
# at all, so an overlap threshold would stall on real assemblies.
#
# CAVEAT: the dataset contains no NEGATIVE examples (a 7x14 present but not
# attached), so this threshold is calibrated to pass all known-attached frames
# and has NOT been validated on how reliably it rejects a detached one. A
# 7x14 laid immediately alongside the 7x7 sits near the limit. Capture a
# "blocks apart" session and re-measure before relying on it.
_ATTACHED_MAX_CENTRE_DIST = 1.20


def evaluate_attached(
    detections: list[Detection],
    class_name: str,
    other_class: str,
    roi: Bbox,
    confidence_threshold: float,
) -> FrameGateReading:
    """3-way read of "is `class_name` ATTACHED to `other_class`?" -- i.e. put
    together with it rather than merely both lying in the placement zone.

    Like seating, attachment is a spatial relation and not an appearance: a
    crop of the 7x14 block looks identical whether it is fixed onto the pegs
    or sitting beside them, so it can only be decided from the detected boxes
    (same reasoning as evaluate_seated and hole_position.py).

    UNCERTAIN when there is nothing to judge -- either part missing from the
    ROI this frame -- so an ambiguous frame counts neither for nor against.
    """
    mine = [d for d in _in_roi(detections, class_name, roi) if d.confidence >= confidence_threshold]
    theirs = [d for d in _in_roi(detections, other_class, roi) if d.confidence >= confidence_threshold]
    if not mine or not theirs:
        return FrameGateReading(outcome=Outcome.UNCERTAIN, confident_count=0, ambiguous_count=0, empty=not mine)

    for a in mine:
        for b in theirs:
            if _centre_distance_ratio(a.bbox, b.bbox) <= _ATTACHED_MAX_CENTRE_DIST:
                return FrameGateReading(outcome=Outcome.CORRECT, confident_count=1, ambiguous_count=0, empty=False)

    # Both parts confidently present but too far apart to be attached -- a
    # real "not attached yet" signal rather than mere absence.
    return FrameGateReading(outcome=Outcome.INCORRECT, confident_count=0, ambiguous_count=0, empty=False)


def _centre_distance_ratio(a: Bbox, b: Bbox) -> float:
    """Distance between two boxes' centres, in units of the SMALLER box's
    diagonal -- scale-free, so it holds as the assembly moves nearer or
    further from the camera."""
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    diagonal = min(((x[2] - x[0]) ** 2 + (x[3] - x[1]) ** 2) ** 0.5 for x in (a, b))
    return distance / diagonal if diagonal > 0 else float("inf")


class Reason(str, Enum):
    """Why a required class isn't satisfied yet, for trainee-facing UI
    feedback. NOT_PRESENT/WRONG_ORIENTATION/WRONG_PART/UNCERTAIN are purely
    diagnostic (advancement stays on evaluate_class_count + the stability
    window, unaffected) -- WRONG_HOLE and NOT_SEATED are the exceptions:
    for steps with target_hole set (1/2 only) WRONG_HOLE reflects
    evaluate_hole_position's own stability-windowed outcome, and for steps
    with require_seated set (4 only) NOT_SEATED reflects evaluate_seated's.
    Both DO gate advancement (see state_machine.py) -- a peg in the wrong
    hole, or a PeeCee lying beside the blocks rather than seated on them,
    must not let the step complete just because the count matched."""

    SATISFIED = "satisfied"
    NOT_PRESENT = "not_present"
    NOT_SEATED = "not_seated"
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


# Classes that are physically always around regardless of build progress.
# `peg` is the merged short_peg/long_peg class (see default_config.yaml): the
# 7x7 block permanently carries 2 long pegs, so pegs are in frame at every
# stage of the build. Steps 1/2 require `peg` explicitly and gate on its
# count, so this entry never affects them -- it only stops steps 3/4/5, which
# deliberately do not require pegs (the 7x14 block occludes them), from
# reporting the block's own fixture pegs as an unexpected "wrong part".
_IGNORED_FOR_WRONG_PART = {"peg"}


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
    require_seated: bool = False,
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

    `require_seated` is only set for step 4 — when set, a count-correct part
    that is lying beside the blocks rather than standing on them is diagnosed
    as NOT_SEATED instead of SATISFIED, and like WRONG_HOLE this one also
    gates advancement.
    """
    reading = evaluate_class_count(detections, class_name, roi, confidence_threshold, required_count)
    if reading.outcome == Outcome.CORRECT:
        if require_seated:
            seated_reading = evaluate_seated(detections, class_name, roi, confidence_threshold)
            if seated_reading.outcome == Outcome.INCORRECT:
                return ClassDiagnosis(Reason.NOT_SEATED)
            if seated_reading.outcome == Outcome.UNCERTAIN:
                return ClassDiagnosis(Reason.UNCERTAIN)
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
        return ClassDiagnosis(Reason.UNCERTAIN, detail=f"{reading.confident_count}/{required_count}")

    # Carry "found/required" so the UI can say "2 of 4 detected" instead of a
    # flat "not detected yet". That distinction matters for count-gated steps
    # (steps 1/2 gate on peg COUNT -- see config): with 3 of the 4 required
    # pegs clearly detected and boxed on screen, "not detected yet" reads as a
    # broken detector rather than "place one more peg".
    return ClassDiagnosis(Reason.NOT_PRESENT, detail=f"{reading.confident_count}/{required_count}")
