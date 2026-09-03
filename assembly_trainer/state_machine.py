"""Rule engine / step state machine (build plan §4/§5, build order item 8).

Strictly linear sequence, cumulative per-step requirements. The final
motor-wiring step is special-cased per §5 as a two-part sequential ROI flow
with an anti-gaming reset-between-confirmations rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config, StepConfig
from .detector import Detection
from .gating import (
    ClassDiagnosis,
    diagnose_class,
    evaluate_class_count,
    evaluate_hole_position,
    evaluate_attached,
    evaluate_seated,
    step_orientation_variant,
)
from .hole_position import Point, compute_homography
from .roi import Bbox, center_inside, connector_zone_roi, normalized_roi_to_pixels, placement_roi, union_bbox
from .stability import StabilityWindow

# Only this class gets hole-position checking (steps 1/2 both place a peg --
# see hole_position.py / config's target_hole). Hardcoded rather than a new
# per-class config field since no other class in this system has hole-level
# semantics.
#
# NOTE for whoever enables hole-position checking (it is inert today -- no
# pose_model_path exists): `peg` is now the MERGED short_peg/long_peg class,
# so peg detections include the 2 long pegs permanently fixtured on the 7x7
# block, which are never in the trainee's target hole. A target_hole check
# against this class must therefore ignore pegs sitting in the two fixture
# positions rather than treating them as a wrong-hole placement.
_HOLE_POSITION_CLASS = "peg"

# Only this class gets the seated check on steps with require_seated set
# (step 4). Hardcoded for the same reason as _HOLE_POSITION_CLASS: no
# other class in this system has seated-on-the-assembly semantics.
_SEATED_CLASS = "peecee"


@dataclass
class FrameResult:
    step_id: int | None
    step_name: str | None
    completed: bool
    advanced: bool  # this frame caused a step (or sub-step) transition
    class_status: dict[str, bool] = field(default_factory=dict)  # class -> currently stably satisfied
    class_diagnosis: dict[str, ClassDiagnosis] = field(default_factory=dict)  # class -> why it isn't satisfied (UI hint only)
    progress_ratio: float = 0.0  # satisfied / required, for the current step
    sub_step_progress: tuple[int, int] | None = None  # (completed_sub_steps, total) for sequential_roi steps
    waiting_for_reset: bool = False
    roi: Bbox | None = None  # the placement zone detections are being gated against this frame (UI viz only)


class StepStateMachine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_step_id: int | None = cfg.first_step_id()
        self.completed = False

        self._windows: dict[str, StabilityWindow] = {}
        self._assembly_bbox: Bbox | None = None
        self._known_classes = set(cfg.classes)

        # sequential_roi (motor-wiring step) state
        self._sub_step_index = 0
        self._sub_step_awaiting_reset = False
        self._last_peecee_bbox: Bbox | None = None

    def current_step(self) -> StepConfig:
        return self.cfg.get_step(self.current_step_id)

    def reset(self) -> None:
        """Return to a fresh start-of-step-1 state — same as a brand new
        instance. Used by the "Start Again" flow so a trainee (or a trainer
        clearing the station between trainees) doesn't need to restart the
        whole server to retry."""
        self.current_step_id = self.cfg.first_step_id()
        self.completed = False
        self._windows.clear()
        self._assembly_bbox = None
        self._sub_step_index = 0
        self._sub_step_awaiting_reset = False
        self._last_peecee_bbox = None

    def go_to_previous_step(self) -> bool:
        """Step back one, for a trainee who advanced by mistake or wants to
        redo the last action. Returns False (and changes nothing) when already
        on the first step.

        Clears the stability windows so the earlier step is judged on fresh
        frames rather than inheriting reads taken while the later step was
        active. Also drops the grown assembly bbox: it only ever expands as
        steps complete, so after stepping back it describes an assembly that
        no longer exists -- dropping it makes the placement ROI fall back to
        the configured workstation zone and re-grow naturally.
        """
        previous_id = self.cfg.previous_step_id(self.current_step_id)
        if previous_id is None:
            return False
        self.current_step_id = previous_id
        self.completed = False
        self._windows.clear()
        self._assembly_bbox = None
        self._sub_step_index = 0
        self._sub_step_awaiting_reset = False
        self._last_peecee_bbox = None
        return True

    def _window(self, key: str, window_frames: int) -> StabilityWindow:
        win = self._windows.get(key)
        if win is None or win._window_frames != window_frames:
            win = StabilityWindow(window_frames)
            self._windows[key] = win
        return win

    def _placement_roi(self, step: StepConfig, frame_w: int, frame_h: int) -> Bbox:
        margin = self.cfg.resolve(step, "assembly_roi_margin_ratio")
        workstation_px = None
        if self.cfg.camera.workstation_roi:
            workstation_px = normalized_roi_to_pixels(self.cfg.camera.workstation_roi, frame_w, frame_h)
        return placement_roi(self._assembly_bbox, workstation_px, margin, frame_w, frame_h)

    def process_frame(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        pose_corners: list[Point | None] | None = None,
    ) -> FrameResult:
        """`pose_corners`, when given, is exactly 4 entries (block_7x7's
        corners in hole_position.CANONICAL_CORNERS order, None for any not
        confidently visible this frame) from the separate pose model —
        only meaningful for steps with `target_hole` set (1/2); ignored
        otherwise, including for the sequential_roi (wiring) step."""
        if self.completed:
            return FrameResult(step_id=None, step_name=None, completed=True, advanced=False)

        step = self.current_step()
        if step.type == "sequential_roi":
            return self._process_sequential_roi(step, detections, frame_w, frame_h)
        return self._process_normal(step, detections, frame_w, frame_h, pose_corners)

    def _process_normal(
        self,
        step: StepConfig,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        pose_corners: list[Point | None] | None = None,
    ) -> FrameResult:
        threshold = self.cfg.resolve(step, "confidence_threshold")
        window_frames = self.cfg.resolve(step, "stability_window_frames")
        pass_ratio = self.cfg.resolve(step, "stability_pass_ratio")
        roi = self._placement_roi(step, frame_w, frame_h)

        orientation_variant = step_orientation_variant(step.id, self._known_classes)
        expected_classes = set(step.requires) | ({orientation_variant} if orientation_variant else set())
        homography = compute_homography(pose_corners) if step.target_hole and pose_corners else None

        class_status: dict[str, bool] = {}
        class_diagnosis: dict[str, ClassDiagnosis] = {}
        for class_name, required_count in step.requires.items():
            reading = evaluate_class_count(detections, class_name, roi, threshold, required_count)
            win = self._window(f"{step.id}:{class_name}", window_frames)
            win.push(reading.outcome)
            ok = win.is_stable_correct(pass_ratio)

            target_hole = step.target_hole if class_name == _HOLE_POSITION_CLASS else None
            if target_hole is not None:
                # A peg in the wrong hole must not let the step complete just
                # because the count matched -- ANDed with its own stability
                # window so a momentary bad homography reading can't flip it
                # either way (same "exclude uncertain frames" philosophy as
                # every other check here).
                hole_reading = evaluate_hole_position(
                    detections, class_name, roi, threshold, target_hole, self.cfg.hole_template, homography
                )
                hole_win = self._window(f"{step.id}:{class_name}:hole", window_frames)
                hole_win.push(hole_reading.outcome)
                ok = ok and hole_win.is_stable_correct(pass_ratio)

            if step.require_seated and class_name == _SEATED_CLASS:
                # Same shape as the hole check above: a part merely PRESENT in
                # the ROI must not complete the step -- it has to be seated on
                # the blocks. ANDed with its own stability window so a single
                # frame where the blocks aren't visible (evaluate_seated ->
                # UNCERTAIN) can't flip it either way.
                seated_reading = evaluate_seated(detections, class_name, roi, threshold)
                seated_win = self._window(f"{step.id}:{class_name}:seated", window_frames)
                seated_win.push(seated_reading.outcome)
                ok = ok and seated_win.is_stable_correct(pass_ratio)

            attach_to = step.require_attached_to.get(class_name)
            if attach_to:
                # Same shape again: both parts merely being in the ROI must not
                # complete the step -- they have to be put together. Its own
                # stability window so a frame where either part isn't visible
                # (evaluate_attached -> UNCERTAIN) can't flip it either way.
                attached_reading = evaluate_attached(detections, class_name, attach_to, roi, threshold)
                attached_win = self._window(f"{step.id}:{class_name}:attached", window_frames)
                attached_win.push(attached_reading.outcome)
                ok = ok and attached_win.is_stable_correct(pass_ratio)

            class_status[class_name] = ok
            # Diagnosis is purely a UI hint, computed fresh each frame (not
            # stability-windowed like advancement) — cheap, and reacting
            # immediately to what's in frame is more useful for a trainee
            # trying to fix something than waiting for a window to settle.
            class_diagnosis[class_name] = diagnose_class(
                detections, class_name, roi, threshold, required_count, expected_classes, orientation_variant,
                target_hole, self.cfg.hole_template, homography,
                require_seated=step.require_seated and class_name == _SEATED_CLASS,
            )

        satisfied = sum(1 for ok in class_status.values() if ok)
        progress_ratio = satisfied / len(class_status) if class_status else 0.0

        if all(class_status.values()):
            self._grow_assembly_bbox(step, detections, roi)
            advanced = self._advance_step()
            return FrameResult(
                step_id=self.current_step_id,
                step_name=None if self.completed else self.current_step().name,
                completed=self.completed,
                advanced=advanced,
                class_status=class_status,
                class_diagnosis=class_diagnosis,
                progress_ratio=1.0,
                roi=roi,
            )

        return FrameResult(
            step_id=step.id,
            step_name=step.name,
            completed=False,
            advanced=False,
            class_status=class_status,
            class_diagnosis=class_diagnosis,
            progress_ratio=progress_ratio,
            roi=roi,
        )

    def _process_sequential_roi(self, step: StepConfig, detections: list[Detection], frame_w: int, frame_h: int) -> FrameResult:
        threshold = self.cfg.resolve(step, "confidence_threshold")
        window_frames = self.cfg.resolve(step, "stability_window_frames")
        pass_ratio = self.cfg.resolve(step, "stability_pass_ratio")

        peecee = next((d for d in detections if d.class_name == "peecee"), None)
        if peecee is not None:
            self._last_peecee_bbox = peecee.bbox
        if self._last_peecee_bbox is None:
            # Can't locate the connector zone yet — no data this frame.
            return FrameResult(
                step_id=step.id, step_name=step.name, completed=False, advanced=False,
                sub_step_progress=(self._sub_step_index, len(step.sub_steps)),
            )

        roi = connector_zone_roi(self._last_peecee_bbox, frame_w, frame_h)
        total = len(step.sub_steps)
        sub_step = step.sub_steps[self._sub_step_index]

        if sub_step.require_roi_empty_first and self._sub_step_awaiting_reset:
            reading = evaluate_class_count(detections, "motor_lead_connected", roi, threshold, 1)
            win = self._window(f"{step.id}:reset:{sub_step.key}", window_frames)
            win.push(reading.outcome)
            if win.is_confidently_reset(pass_ratio):
                self._sub_step_awaiting_reset = False
                win.reset()
            return FrameResult(
                step_id=step.id, step_name=step.name, completed=False, advanced=False,
                sub_step_progress=(self._sub_step_index, total), waiting_for_reset=True,
            )

        reading = evaluate_class_count(detections, "motor_lead_connected", roi, threshold, 1)
        win = self._window(f"{step.id}:confirm:{sub_step.key}", window_frames)
        win.push(reading.outcome)

        if win.is_stable_correct(pass_ratio):
            self._sub_step_index += 1
            if self._sub_step_index >= total:
                self._grow_assembly_bbox(step, detections, roi)
                advanced = self._advance_step()
                return FrameResult(
                    step_id=self.current_step_id,
                    step_name=None if self.completed else self.current_step().name,
                    completed=self.completed,
                    advanced=advanced,
                    sub_step_progress=(total, total),
                    roi=roi,
                )
            next_sub_step = step.sub_steps[self._sub_step_index]
            self._sub_step_awaiting_reset = next_sub_step.require_roi_empty_first
            return FrameResult(
                step_id=step.id, step_name=step.name, completed=False, advanced=True,
                sub_step_progress=(self._sub_step_index, total), roi=roi,
            )

        return FrameResult(
            step_id=step.id, step_name=step.name, completed=False, advanced=False,
            sub_step_progress=(self._sub_step_index, total), roi=roi,
        )

    def _grow_assembly_bbox(self, step: StepConfig, detections: list[Detection], roi: Bbox) -> None:
        boxes = [d.bbox for d in detections if d.class_name in step.requires and center_inside(d.bbox, roi)]
        new_union = union_bbox(boxes)
        if new_union is None:
            return
        self._assembly_bbox = union_bbox([self._assembly_bbox, new_union]) if self._assembly_bbox else new_union

    def _advance_step(self) -> bool:
        next_id = self.cfg.next_step_id(self.current_step_id)
        self._windows.clear()
        self._sub_step_index = 0
        self._sub_step_awaiting_reset = False
        if next_id is None:
            self.completed = True
            self.current_step_id = None
        else:
            self.current_step_id = next_id
        return True
