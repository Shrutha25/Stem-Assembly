"""Orchestration glue (build order item 10): frame source -> detector ->
state machine -> escalation timer -> shared, UI-consumable result.

One `Pipeline.step()` call is one inference cycle. The caller (the FastAPI
server for live use, or the smoke test for headless verification) is
responsible for calling it on a schedule (`app.inference_interval_ms`) and
reading `pipeline.last_result` / `pipeline.flags` afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .config import Config
from .detector import Detection, Detector
from .escalation import EscalationTimer, Tier
from .pose_detector import PoseCornerDetector
from .state_machine import FrameResult, StepStateMachine


@dataclass
class TrainerFlag:
    step_id: int
    step_name: str
    at: float  # wall-clock time.time() (for display), independent of the pipeline's injectable clock


@dataclass
class PipelineResult:
    frame_result: FrameResult
    tier: Tier
    elapsed_on_step: float
    detections: list[Detection] = field(default_factory=list)
    frame: object = None
    flags: list[TrainerFlag] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        detector: Detector,
        frame_source: Callable[[], object | None],
        clock: Callable[[], float] = time.monotonic,
        pose_detector: PoseCornerDetector | None = None,
    ):
        self.cfg = cfg
        self.detector = detector
        self.frame_source = frame_source
        self.clock = clock
        # Optional second model (see pose_detector.py) -- only invoked when
        # the current step has target_hole set, so steps without a hole
        # check (3-5, and 1/2 until pose_model_path is configured) pay
        # nothing extra per frame.
        self.pose_detector = pose_detector

        self.state_machine = StepStateMachine(cfg)
        self.escalation = EscalationTimer(clock=clock)
        self.escalation.on_step_started()

        self.flags: list[TrainerFlag] = []
        self.last_result: PipelineResult | None = None

    def reset(self) -> None:
        """Start a fresh attempt: back to step 1, escalation timer restarted,
        past trainer flags cleared (they belonged to the attempt being
        discarded)."""
        self.state_machine.reset()
        self.escalation.on_step_started()
        self.flags.clear()
        self.last_result = None

    def go_to_previous_step(self) -> bool:
        """Step back one. Restarts the escalation timer so the trainee gets a
        full grace period on the step they returned to rather than inheriting
        the elapsed time that had already escalated. Past trainer flags are
        kept -- unlike reset(), this isn't a new attempt, and a trainer should
        still see that help was requested earlier in this one."""
        if not self.state_machine.go_to_previous_step():
            return False
        self.escalation.on_step_started()
        self.last_result = None
        return True

    def step(self) -> PipelineResult | None:
        frame = self.frame_source()
        if frame is None:
            return None

        detections = self.detector.infer(frame)
        frame_h, frame_w = frame.shape[0], frame.shape[1]

        pose_corners = None
        if self.pose_detector is not None and not self.state_machine.completed:
            current_step = self.state_machine.current_step()
            if current_step.target_hole is not None:
                pose_corners = self.pose_detector.infer_corners(frame)

        prev_step_id = self.state_machine.current_step_id
        result = self.state_machine.process_frame(detections, frame_w, frame_h, pose_corners=pose_corners)

        # Reset the escalation timer whenever the *current step* changes
        # (not on step-10 sub-step transitions — see escalation.py docstring).
        if result.step_id != prev_step_id:
            self.escalation.on_step_started()

        if result.completed:
            tier = Tier.NONE
        else:
            step_cfg = self.state_machine.current_step()
            esc_cfg = self.cfg.resolve_escalation(step_cfg)
            tier = self.escalation.tick(esc_cfg)

        new_flags: list[TrainerFlag] = []
        if not result.completed and self.escalation.consume_new_trainer_alert():
            flag = TrainerFlag(step_id=result.step_id, step_name=result.step_name, at=time.time())
            self.flags.append(flag)
            new_flags.append(flag)

        pipeline_result = PipelineResult(
            frame_result=result,
            tier=tier,
            elapsed_on_step=self.escalation.elapsed_seconds,
            detections=detections,
            frame=frame,
            flags=new_flags,
        )
        self.last_result = pipeline_result
        return pipeline_result
