"""ROI helpers for the per-step placement gate (§4 closing note) and the
motor-wiring step's fixed connector-zone ROI (§5-style loose/connected check).

A bbox is always `(x1, y1, x2, y2)` in pixel coordinates.
"""

from __future__ import annotations

from typing import Iterable

Bbox = tuple[float, float, float, float]


def union_bbox(boxes: Iterable[Bbox]) -> Bbox | None:
    boxes = list(boxes)
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def expand_bbox(box: Bbox, margin_ratio: float, frame_w: int, frame_h: int) -> Bbox:
    """Grow `box` by `margin_ratio` of its own width/height, clipped to the frame."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_ratio, h * margin_ratio
    return (
        max(0.0, x1 - mx),
        max(0.0, y1 - my),
        min(float(frame_w), x2 + mx),
        min(float(frame_h), y2 + my),
    )


def normalized_roi_to_pixels(roi: dict, frame_w: int, frame_h: int) -> Bbox:
    """Convert a `{x, y, w, h}` normalized (0-1) ROI into a pixel bbox."""
    x1 = roi["x"] * frame_w
    y1 = roi["y"] * frame_h
    return (x1, y1, x1 + roi["w"] * frame_w, y1 + roi["h"] * frame_h)


def bbox_center(box: Bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_inside(box: Bbox, roi: Bbox) -> bool:
    cx, cy = bbox_center(box)
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def placement_roi(
    assembly_bbox: Bbox | None,
    workstation_roi_px: Bbox | None,
    margin_ratio: float,
    frame_w: int,
    frame_h: int,
) -> Bbox:
    """The ROI a step's new-part detections must land inside of.

    Before anything has been confirmed (no assembly bbox yet, i.e. step 1),
    fall back to the fixed workstation zone if configured, else the whole
    frame. Once parts have been confirmed, grow the ROI from the union of
    their bboxes so later steps track wherever the assembly actually is.
    """
    if assembly_bbox is not None:
        return expand_bbox(assembly_bbox, margin_ratio, frame_w, frame_h)
    if workstation_roi_px is not None:
        return workstation_roi_px
    return (0.0, 0.0, float(frame_w), float(frame_h))


def connector_zone_roi(peecee_bbox: Bbox, frame_w: int, frame_h: int) -> Bbox:
    """The motor-wiring step's fixed ROI, anchored relative to the detected
    `peecee` bbox. The exact offset from the PCB edge to the wire-connector
    header is fixture-specific and unknown at spec time, so this v1 uses the
    peecee bbox itself, slightly padded, as the classification zone. Tune
    this to a tighter sub-region once the physical fixture/camera mount is
    fixed.
    """
    return expand_bbox(peecee_bbox, margin_ratio=0.15, frame_w=frame_w, frame_h=frame_h)
