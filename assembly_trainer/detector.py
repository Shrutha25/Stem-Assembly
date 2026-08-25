"""Detection + tracking stage (build plan §2/§9, build order item 6).

`YoloByteTrackDetector` wraps an `ultralytics` YOLO model running with its
built-in ByteTrack tracker (`model.track(..., tracker="bytetrack.yaml")`) —
used for stable counting and occlusion tolerance, not instance
re-identification (per plan §2). `Detection` is the common output type both
this and `mock_detector.MockDetector` produce, so everything downstream
(gating, stability, state machine) is detector-implementation-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .roi import Bbox


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: Bbox  # (x1, y1, x2, y2) pixels
    track_id: int | None = None


class Detector(Protocol):
    """Common interface implemented by both the real and mock detectors."""

    def infer(self, frame) -> list[Detection]:
        """Run one inference cycle on a BGR frame, return tracked detections."""
        ...


class YoloByteTrackDetector:
    """Real detector: ultralytics YOLO model + ByteTrack.

    Lazily imports `ultralytics` so the rest of the package (config, rule
    engine, stability, escalation, smoke test) has zero hard dependency on
    it and can be exercised without the model/GPU stack installed.
    """

    def __init__(self, model_path: str, class_names: list[str], imgsz: int = 960, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install ultralytics` "
                "to use the real detector, or pass --mock to use MockDetector."
            ) from exc

        self._model = YOLO(model_path)
        self._class_names = class_names
        self._imgsz = imgsz
        self._device = device

    def infer(self, frame) -> list[Detection]:
        results = self._model.track(
            frame,
            imgsz=self._imgsz,
            device=self._device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names  # model's own class-id -> name mapping
        detections: list[Detection] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
            track_id = int(boxes.id[i].item()) if boxes.id is not None else None
            detections.append(
                Detection(class_name=class_name, confidence=conf, bbox=(x1, y1, x2, y2), track_id=track_id)
            )
        return detections
