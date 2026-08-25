"""Threaded camera capture wrapper (build plan §7, build order item 2).

Runs `cv2.VideoCapture.read()` continuously on a background thread into a
single-slot, always-latest-frame buffer, so the inference loop (running at
`inference_interval_ms`, decoupled from camera FPS) always gets the most
recent frame instead of blocking on or backing up behind camera I/O.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2

from .config import CameraConfig

# On Windows, cv2.VideoCapture's default backend (Media Foundation) has been
# observed to hang indefinitely on open() rather than fail fast — regardless
# of whether another app also has the camera open. DirectShow opens reliably
# and quickly, so force it explicitly on this platform.
_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


class ThreadedCamera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_time = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "ThreadedCamera":
        self._cap = cv2.VideoCapture(self.cfg.device_index, _BACKEND)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera device_index={self.cfg.device_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        # Leave auto-exposure alone: CAP_PROP_AUTO_EXPOSURE's value convention
        # is backend/driver-specific (the "0.75 = auto" DSHOW convention isn't
        # universal), and most webcams already default to auto — setting it
        # to the wrong magic number risks accidentally forcing manual mode.

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Auto-exposure/white-balance needs real time to converge after
        # open() — observed taking a couple seconds on this hardware, and
        # exiting as soon as the first frame arrives cuts it off mid-swing
        # (caught it under- *and* over-exposed at different cutoffs). Give it
        # a fixed settle window rather than racing on "a frame showed up".
        time.sleep(self.cfg.warmup_seconds)
        return self

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_time = time.monotonic()

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    @property
    def actual_resolution(self) -> tuple[int, int]:
        if self._cap is None:
            return (self.cfg.width, self.cfg.height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
