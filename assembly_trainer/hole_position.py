"""Hole-position detection for steps 1/2 (see build plan for this feature).

The live detector's plain axis-aligned boxes can't answer "which of the 7
holes on this edge is the peg in" — the block is photographed at arbitrary
rotations, and the target hole itself is often hidden by the very peg
sitting in it. This module solves it geometrically instead of visually:
given the block's 4 corner keypoints (from a separate pose model, see
`pose_detector.py`), compute a homography into a canonical block rectangle,
project a peg's position through it, and look up the nearest known hole in
that canonical space. This works even when the target hole is occluded,
because we're computing where it geometrically must be, not looking for it.

Canonical corner order (must match the order the pose model was trained on,
and the order `label_tool.py`'s keypoint mode asks the user to click in):
top-left, top-right, bottom-right, bottom-left -- i.e. clockwise starting
from the corner nearest the block's origin corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

Point = tuple[float, float]

CANONICAL_CORNERS: tuple[Point, Point, Point, Point] = (
    (0.0, 0.0),  # top-left
    (1.0, 0.0),  # top-right
    (1.0, 1.0),  # bottom-right
    (0.0, 1.0),  # bottom-left
)

# Single source of truth for the corner order/names, shared by the pose
# model's keypoint order, label_tool.py's click-order prompts, and
# CANONICAL_CORNERS above -- keep all three in sync via this one tuple.
CORNER_NAMES: tuple[str, str, str, str] = ("top-left", "top-right", "bottom-right", "bottom-left")

# A homography from fewer than this many confidently-visible corners is too
# underdetermined to trust -- treat the frame as UNCERTAIN instead (see
# gating.py), the same "exclude uncertain frames" philosophy stability.py
# already uses elsewhere.
MIN_VISIBLE_CORNERS = 3

# Nearest-hole lookup is rejected past this canonical-space distance so a peg
# nowhere near any hole returns None rather than a false match. Canonical
# space is normalized 0-1, so 0.18 is roughly "further than one hole-spacing
# away" for a 7-hole edge (~0.14 spacing).
DEFAULT_MAX_HOLE_DISTANCE = 0.18


@dataclass
class Homography:
    matrix: np.ndarray  # 3x3

    def project(self, point: Point) -> Point:
        return project_point(self, point)


def compute_homography(observed_corners: list[Point | None]) -> Homography | None:
    """`observed_corners` is exactly 4 entries in CANONICAL_CORNERS order,
    each either an (x, y) pixel point or None if that corner wasn't
    confidently visible this frame. Returns None if fewer than
    MIN_VISIBLE_CORNERS are available.
    """
    if len(observed_corners) != 4:
        raise ValueError(f"expected exactly 4 corner entries, got {len(observed_corners)}")

    pairs = [(canon, obs) for canon, obs in zip(CANONICAL_CORNERS, observed_corners) if obs is not None]
    if len(pairs) < MIN_VISIBLE_CORNERS:
        return None

    canon_pts = np.array([p[0] for p in pairs], dtype=np.float32)
    obs_pts = np.array([p[1] for p in pairs], dtype=np.float32)

    if len(pairs) == 4:
        matrix = cv2.getPerspectiveTransform(obs_pts, canon_pts)
    else:
        # 3 points: fall back to an affine estimate (no perspective term) --
        # still enough to localize a hole approximately, just less robust to
        # steep viewing angles than a full 4-point homography.
        affine = cv2.getAffineTransform(obs_pts, canon_pts)
        matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)

    return Homography(matrix=matrix)


def project_point(h: Homography, point: Point) -> Point:
    x, y = point
    vec = h.matrix @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(vec[2]) < 1e-9:
        return (math.inf, math.inf)
    return (vec[0] / vec[2], vec[1] / vec[2])


def nearest_hole(
    canonical_point: Point,
    hole_template: dict[str, list[dict]],
    max_distance: float = DEFAULT_MAX_HOLE_DISTANCE,
) -> str | None:
    """Nearest-neighbor lookup across every edge's holes combined. Returns
    the hole name, or None if nothing is within `max_distance`."""
    px, py = canonical_point
    best_name: str | None = None
    best_dist = math.inf
    for holes in hole_template.values():
        for hole in holes:
            d = math.hypot(hole["x"] - px, hole["y"] - py)
            if d < best_dist:
                best_dist = d
                best_name = hole["name"]
    if best_name is None or best_dist > max_distance:
        return None
    return best_name
