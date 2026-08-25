"""ONNX export step (build plan §9, build order item 5).

A common failure mode called out in the plan: testing live against a stale
model that wasn't re-exported/re-deployed after a retrain. No automated
mtime-checking for v1 (explicitly deferred, §12) — just a loud reminder here
and a matching one printed at the end of `training/train.py`.

Usage:
    python -m assembly_trainer.training.export --weights models/runs/assembly_trainer/weights/best.pt \\
        --config config/default_config.yaml
    # For the block_7x7 corner-pose model (see hole_position.py / train_pose.py),
    # write to pose_model_path instead of clobbering the main detector's model_path:
    python -m assembly_trainer.training.export --weights models/runs/block_pose/weights/best.pt \\
        --config-field pose_model_path --update-config
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..config import load_config


def run(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run `pip install ultralytics` before exporting."
        ) from exc

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"weights not found: {weights_path}")

    # §7: same imgsz as live inference/training unless explicitly overridden
    # — a mismatched export imgsz is a silent way to tank accuracy.
    imgsz = args.imgsz if args.imgsz is not None else load_config(args.config).app.imgsz

    model = YOLO(str(weights_path))
    exported_path = model.export(format="onnx", imgsz=imgsz)
    exported_path = Path(exported_path)

    print(f"[export] wrote {exported_path}")

    if args.config:
        config_path = Path(args.config)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        field = args.config_field
        old_path = raw.get(field)
        if args.update_config:
            raw[field] = str(exported_path)
            config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            print(f"[export] updated {config_path}: {field} {old_path!r} -> {str(exported_path)!r}")
        elif old_path != str(exported_path):
            print(
                f"[export] REMINDER: {config_path} still points {field} at {old_path!r} — "
                f"update it to {str(exported_path)!r} (or re-run with --update-config) "
                f"before live testing. A stale {field} is the #1 cause of 'why isn't it detecting "
                "anything new' during live tests (§9)."
            )

    print("[export] REMINDER: re-export + re-deploy after every retrain, not just the first time.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="path to trained best.pt")
    parser.add_argument("--imgsz", type=int, default=None, help="override app.imgsz from --config")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--config-field", default="model_path",
                         help="which top-level config key to update/check -- 'model_path' (default) for the "
                              "main detector, 'pose_model_path' for the block_7x7 corner-pose model")
    parser.add_argument("--update-config", action="store_true", help="write the new path into --config's --config-field directly")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
