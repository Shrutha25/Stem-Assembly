"""YOLOv8-pose training for the block_7x7 corner-keypoint model (steps 1/2's
hole-position check -- see assembly_trainer/hole_position.py). Single class
(block_7x7), 4 keypoints in CORNER_NAMES order (top-left, top-right,
bottom-right, bottom-left), labeled via:
    python -m assembly_trainer.label_tool --split train --mode keypoints

No class-balanced oversampling here (unlike training/train.py) -- this is a
single-class dataset, there's no imbalance to correct. Uses a nano base
checkpoint rather than the main detector's yolov8s: one class + 4 keypoints
is a much lighter task, and this dataset is a subset of the main one (only
images that already have a block_7x7 box), so a smaller model both trains
faster and is less likely to overfit on the smaller sample count.

Usage:
    python -m assembly_trainer.training.train_pose --data data/dataset_pose --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..config import load_config


def _write_pose_data_yaml(pose_dataset_root: Path) -> Path:
    for split in ("train", "val", "test"):
        img_dir = pose_dataset_root / "images" / split
        if not img_dir.exists() or not any(img_dir.glob("*.jpg")):
            raise SystemExit(
                f"{img_dir} has no images -- label some first: "
                f"python -m assembly_trainer.label_tool --split {split} --mode keypoints"
            )
    data = {
        "path": str(pose_dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "block_7x7"},
        "kpt_shape": [4, 3],  # 4 corners, (x, y, visibility)
    }
    out_path = pose_dataset_root / "data.yaml"
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out_path


def run(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run `pip install ultralytics` before training."
        ) from exc

    imgsz = args.imgsz if args.imgsz is not None else load_config(args.app_config).app.imgsz

    pose_dataset_root = Path(args.data).resolve()
    data_yaml = _write_pose_data_yaml(pose_dataset_root)

    # Same reasoning as training/train.py: resolve --project to absolute --
    # a relative path has been observed landing under a stray global
    # runs_dir left in this machine's ultralytics settings.json from an
    # unrelated prior project, instead of under this one.
    project_dir = Path(args.project).resolve()

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=imgsz,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        batch=args.batch,
        # The 4 corners aren't a simple left-right swap under a horizontal
        # flip (the click order top-left/top-right/bottom-right/bottom-left
        # encodes real orientation) -- rather than work out a correct
        # flip_idx remapping, flip augmentation is just disabled here.
        fliplr=0.0,
    )
    save_dir = getattr(results, "save_dir", project_dir / args.name)
    print(f"[train_pose] done. Best weights: {save_dir}/weights/best.pt")
    print("[train_pose] REMINDER: run `python -m assembly_trainer.training.export "
          f"--weights {save_dir}/weights/best.pt --config-field pose_model_path --update-config`, "
          "then set target_hole on steps 1/2 in your config before live testing.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/dataset_pose", help="pose dataset root (images/labels/{train,val,test})")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="base weights to fine-tune from (nano, not small -- see module docstring)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--app-config", default="config/default_config.yaml", help="source of the shared default imgsz (app.imgsz)")
    parser.add_argument("--imgsz", type=int, default=None, help="override app.imgsz from --app-config")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="models/runs")
    parser.add_argument("--name", default="block_pose")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
