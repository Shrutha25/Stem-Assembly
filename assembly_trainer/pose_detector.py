"""Second, optional model: corner-keypoint pose detector for block_7x7,
feeding hole_position.py's homography for steps 1/2's target_hole check.

Kept entirely separate from detector.py's YoloByteTrackDetector -- different
ultralytics task (pose vs detect), different checkpoint, and it's only ever
invoked for steps 1/2 (see pipeline.py), so a completely independent class
here keeps the always-on main detector free of any pose-specific branching.
"""

from __future__ import annotations

from .hole_position import Point

# Keep in sync with hole_position.CANONICAL_CORNERS's documented order --
# this is the order the pose model's keypoints are trained/labeled in
# (label_tool.py's keypoint mode enforces the same click order).
_MIN_KEYPOINT_CONFIDENCE = 0.5


class PoseCornerDetector:
    """Lazily imports `ultralytics` (same reasoning as detector.py: the rest
    of the package stays exercisable without the model/GPU stack installed,
    and this detector is only ever constructed at all when pose_model_path
    is configured)."""

    def __init__(self, model_path: str, imgsz: int = 960, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install ultralytics` "
                "to use hole-position checking, or leave pose_model_path unset."
            ) from exc

        self._model = YOLO(model_path)
        if self._model.task != "pose":
            # A real, silent-failure-prone gotcha: a plain ONNX export has no
            # embedded task metadata for ultralytics to read back, so its
            # task-guessing falls back to checking whether the filename
            # contains the literal substring "-pose" (hyphen) -- confirmed
            # directly: renaming block_pose.onnx (underscore) to
            # block-pose.onnx flipped this from 'detect' to 'pose'. Loaded as
            # 'detect', keypoints are silently never parsed and
            # infer_corners() would return None forever with no error at all
            # -- fail loudly here instead of reproducing that.
            raise RuntimeError(
                f"pose model at '{model_path}' loaded with task='{self._model.task}', expected 'pose'. "
                "If this is an ONNX export, ultralytics guesses the task from the filename when the "
                "file has no embedded task metadata -- the filename must contain '-pose' (with a "
                "hyphen, e.g. 'block-pose.onnx', not 'block_pose.onnx')."
            )
        self._imgsz = imgsz
        self._device = device

    def infer_corners(self, frame) -> list[Point | None] | None:
        """Returns exactly 4 entries in CANONICAL_CORNERS order (top-left,
        top-right, bottom-right, bottom-left), each an (x, y) pixel point or
        None if that corner wasn't confidently visible -- or None (not a
        4-list) if no block_7x7 instance was detected at all this frame."""
        results = self._model.predict(frame, imgsz=self._imgsz, device=self._device, verbose=False)
        if not results:
            return None

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            return None

        # Multiple block instances shouldn't happen in this system (one
        # fixed workstation, one block), but if the model hallucinates a
        # second one, trust whichever detection it's most confident about.
        best_idx = int(boxes.conf.argmax().item())

        xy = keypoints.xy[best_idx]  # (4, 2) tensor
        conf = keypoints.conf[best_idx] if keypoints.conf is not None else None
        if xy.shape[0] != 4:
            raise RuntimeError(
                f"pose model produced {xy.shape[0]} keypoints, expected exactly 4 "
                "(block_7x7 corners) -- was the wrong model/checkpoint loaded as pose_model_path?"
            )

        corners: list[Point | None] = []
        for i in range(4):
            x, y = float(xy[i][0].item()), float(xy[i][1].item())
            kpt_conf = float(conf[i].item()) if conf is not None else 1.0
            corners.append((x, y) if kpt_conf >= _MIN_KEYPOINT_CONFIDENCE else None)
        return corners
