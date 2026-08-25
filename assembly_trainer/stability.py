"""Stability window (build plan §1/§7, build order item 7).

Majority-vote over the last `window_frames` *counted* (non-UNCERTAIN)
per-frame outcomes. UNCERTAIN frames are excluded from the window entirely —
they don't count for or against — which matters more here than in a
press-to-Check system since continuous detection sees far more mid-motion /
occluded / blurry frames.
"""

from __future__ import annotations

from collections import deque

from .gating import Outcome


class StabilityWindow:
    def __init__(self, window_frames: int):
        self._window_frames = window_frames
        self._counted: deque[Outcome] = deque(maxlen=window_frames)

    def push(self, outcome: Outcome) -> None:
        if outcome is Outcome.UNCERTAIN:
            return  # excluded entirely — doesn't enter the window
        self._counted.append(outcome)

    def is_stable_correct(self, pass_ratio: float) -> bool:
        # Require a full window of *counted* frames before declaring
        # stability — otherwise a single lucky frame (ratio 1/1) would pass
        # the ratio check instantly. "Last N counted frames" means N.
        if len(self._counted) < self._window_frames:
            return False
        correct = sum(1 for o in self._counted if o is Outcome.CORRECT)
        return (correct / len(self._counted)) >= pass_ratio

    def is_confidently_reset(self, pass_ratio: float) -> bool:
        """For the motor-wiring step's anti-gaming reset check: the window
        has enough counted data and is *not* stably reading CORRECT
        (connected)."""
        if len(self._counted) < self._window_frames:
            return False  # not enough fresh data since listening for reset began
        return not self.is_stable_correct(pass_ratio)

    def reset(self) -> None:
        self._counted.clear()
