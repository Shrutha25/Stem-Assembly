"""Quick visual test of the trained block_7x7 corner-pose model against the
live webcam. Draws the 4 detected corners (colored, numbered 1-4 in
CORNER_NAMES order) plus a connecting quadrilateral when all 4 are visible.

This talks to the pose model directly, NOT through the full server/pipeline
-- target_hole isn't set on any step yet, so the live server would load this
model but never actually invoke it (see pipeline.py: it only runs the pose
detector when the current step has target_hole set). This is the way to
verify the trained model itself actually works before wiring target_hole up.

Usage:
    python scripts/pose_smoke_test.py
    python scripts/pose_smoke_test.py --model models/block_pose.onnx --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from assembly_trainer.config import load_config
from assembly_trainer.pose_detector import PoseCornerDetector

# BGR, in CORNER_NAMES order: top-left, top-right, bottom-right, bottom-left
_CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="override config's pose_model_path")
    parser.add_argument("--app-config", default="config/default_config.yaml")
    parser.add_argument("--device", default=None, help="e.g. 'cpu' or '0'")
    args = parser.parse_args()

    cfg = load_config(args.app_config)
    model_path = args.model or cfg.pose_model_path
    if not model_path:
        raise SystemExit("no pose model configured -- pass --model or set pose_model_path in config")

    print(f"[pose_smoke_test] loading {model_path} ...")
    detector = PoseCornerDetector(model_path, imgsz=cfg.app.imgsz, device=args.device)

    cap = cv2.VideoCapture(cfg.camera.device_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    time.sleep(cfg.camera.warmup_seconds)

    window = "pose_smoke_test (ESC to quit)"
    cv2.namedWindow(window)
    print("[pose_smoke_test] running -- ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        corners = detector.infer_corners(frame)
        out = frame.copy()

        if corners is None:
            cv2.putText(out, "no block_7x7 detected", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            pts = []
            for i, pt in enumerate(corners):
                if pt is None:
                    continue
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(out, (px, py), 8, _CORNER_COLORS[i], -1)
                cv2.putText(out, str(i + 1), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CORNER_COLORS[i], 2)
                pts.append((px, py))
            n_visible = sum(1 for c in corners if c is not None)
            cv2.putText(out, f"{n_visible}/4 corners visible", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            if n_visible == 4:
                cv2.polylines(out, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 2)

        cv2.imshow(window, out)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
