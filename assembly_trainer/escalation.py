"""Tiered escalation timer (build plan §6, build order item 9).

No Check button — timer-based, per-step, resets when a step becomes
current. Tiers stack (reaching tier 3 doesn't hide tier 1/2's reference
material); the trainer alert is an addition to the UI, not a takeover.

v1 is the simple elapsed-time-only version the plan explicitly allows;
pause-on-partial-progress is left as a documented, unused config field
(`escalation.pause_on_partial_progress`) rather than implemented now.
"""

from __future__ import annotations

import time
from enum import IntEnum

from .config import EscalationDefaults


class Tier(IntEnum):
    NONE = 0
    REFERENCE_IMAGE = 1
    REFERENCE_VIDEO = 2
    TRAINER_ALERT = 3


class EscalationTimer:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._started_at: float | None = None
        self._current_tier = Tier.NONE
        self._trainer_alert_pending = False

    def on_step_started(self) -> None:
        self._started_at = self._clock()
        self._current_tier = Tier.NONE
        self._trainer_alert_pending = False

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return self._clock() - self._started_at

    def tick(self, escalation_cfg: EscalationDefaults) -> Tier:
        if self._started_at is None:
            return Tier.NONE
        elapsed = self.elapsed_seconds
        if elapsed >= escalation_cfg.tier_3_trainer_alert_after_seconds:
            new_tier = Tier.TRAINER_ALERT
        elif elapsed >= escalation_cfg.tier_2_reference_video_after_seconds:
            new_tier = Tier.REFERENCE_VIDEO
        elif elapsed >= escalation_cfg.tier_1_reference_image_after_seconds:
            new_tier = Tier.REFERENCE_IMAGE
        else:
            new_tier = Tier.NONE

        if new_tier is Tier.TRAINER_ALERT and self._current_tier is not Tier.TRAINER_ALERT:
            self._trainer_alert_pending = True
        self._current_tier = new_tier
        return new_tier

    def consume_new_trainer_alert(self) -> bool:
        """Returns True exactly once per tier-3 transition, for the caller
        to log a trainer-dashboard flag without re-flagging every tick."""
        if self._trainer_alert_pending:
            self._trainer_alert_pending = False
            return True
        return False
