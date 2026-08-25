"""YOLOv8 training pipeline (build plan §9, build order item 4).

Wraps `ultralytics` training with class-balanced oversampling: classes that
appear in more steps of the sequence (config §8's `steps`) will naturally
have far more labeled frames than classes confined to one or two steps, so
we oversample training images containing rarer classes before handing off to the
`ultralytics` trainer — the practical way to get class balance without
patching ultralytics' internal sampler. Validation is left untouched (must
stay representative, and must be split by worker/session per §10, which is
the caller's job when building `data.yaml`).

Usage:
    python -m assembly_trainer.training.train --data data/dataset/data.yaml \\
        --model yolov8s.pt --epochs 100 --imgsz 960
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import yaml

from ..config import load_config


def _label_path_for_image(image_path: Path) -> Path:
    # Standard YOLO layout: .../images/xxx.jpg -> .../labels/xxx.txt
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _classes_in_label(label_path: Path) -> set[int]:
    if not label_path.exists():
        return set()
    classes = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        classes.add(int(line.split()[0]))
    return classes


def _check_fully_labeled(image_paths: list[Path], allow_unlabeled: bool) -> None:
    """A *missing* label file (as opposed to a deliberately empty one) is
    treated by ultralytics as "this image has zero objects" — training on a
    partially-labeled dataset silently teaches the model that every
    not-yet-labeled part doesn't exist. Refuse by default rather than let
    that happen quietly; --allow-unlabeled opts out for deliberate
    partial-dataset experiments."""
    unlabeled = [p for p in image_paths if not _label_path_for_image(p).exists()]
    if unlabeled and not allow_unlabeled:
        sample = ", ".join(p.name for p in unlabeled[:5])
        more = f" (+{len(unlabeled) - 5} more)" if len(unlabeled) > 5 else ""
        raise SystemExit(
            f"[train] {len(unlabeled)}/{len(image_paths)} training images have no label file yet "
            f"(e.g. {sample}{more}). ultralytics would treat these as confirmed-empty (zero objects), "
            "not 'not labeled yet' — that actively teaches wrong negatives for whatever's actually in "
            "them. Finish labeling first (see assembly_trainer.label_tool / "
            "assembly_trainer.training.verify_labels), or pass --allow-unlabeled if this is deliberate."
        )


def build_class_balanced_train_list(
    train_images_dir: Path, num_classes: int, oversample_cap: int = 8, seed: int = 0, allow_unlabeled: bool = False
) -> list[Path]:
    """Return a (possibly repeated) list of image paths so rarer classes get
    proportionally more exposure per training epoch."""
    image_paths = sorted(
        p for p in train_images_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise ValueError(f"no images found under {train_images_dir}")

    _check_fully_labeled(image_paths, allow_unlabeled)
    image_classes = {p: _classes_in_label(_label_path_for_image(p)) for p in image_paths}

    class_counts = Counter()
    for classes in image_classes.values():
        class_counts.update(classes)
    if not class_counts:
        raise ValueError(f"no YOLO-format labels found for images under {train_images_dir}")

    max_count = max(class_counts.values())
    class_weight = {
        c: (max_count / class_counts[c]) if class_counts.get(c) else 1.0 for c in range(num_classes)
    }

    rng = random.Random(seed)
    oversampled: list[Path] = []
    for path, classes in image_classes.items():
        weight = max((class_weight[c] for c in classes), default=1.0)
        repeats = min(oversample_cap, max(1, round(weight)))
        oversampled.extend([path] * repeats)
    rng.shuffle(oversampled)
    return oversampled


def _write_train_list_and_data_yaml(
    original_data_yaml: Path, train_list: list[Path], out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "train_oversampled.txt"
    list_path.write_text("\n".join(str(p.resolve()) for p in train_list), encoding="utf-8")

    original = yaml.safe_load(original_data_yaml.read_text(encoding="utf-8"))
    patched = dict(original)
    patched["train"] = str(list_path.resolve())
    # val/test in the original yaml are relative to *its* directory. The
    # patched yaml lives somewhere else (out_dir), so leaving them as-is
    # would have ultralytics re-resolve them against the wrong directory —
    # resolve to absolute paths now so it can't matter where this file ends up.
    for key in ("val", "test"):
        if patched.get(key):
            patched[key] = str((original_data_yaml.parent / patched[key]).resolve())
    patched_path = out_dir / "data_oversampled.yaml"
    patched_path.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")
    return patched_path


def run(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run `pip install ultralytics` before training."
        ) from exc

    imgsz = args.imgsz if args.imgsz is not None else load_config(args.app_config).app.imgsz

    data_yaml = Path(args.data)
    original = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    num_classes = len(original["names"])
    train_images_dir = (data_yaml.parent / original["train"]).resolve()
    if train_images_dir.is_file():
        raise SystemExit(
            f"--data 'train' already points at a file list ({train_images_dir}); "
            "point it at the raw train images directory instead — this script generates its own list."
        )

    print(f"[train] scanning {train_images_dir} for class balance ...")
    train_list = build_class_balanced_train_list(
        train_images_dir, num_classes, args.oversample_cap, allow_unlabeled=args.allow_unlabeled
    )
    print(f"[train] {len(train_list)} training samples after class-balanced oversampling "
          f"(cap={args.oversample_cap}x)")

    # Resolve to absolute: a relative --project has been observed landing
    # under a stray global `runs_dir` from this machine's ultralytics
    # settings.json (leftover from an unrelated prior project) instead of
    # under this project — an absolute path can't be reinterpreted like that.
    project_dir = Path(args.project).resolve()
    out_dir = project_dir / "_oversample_cache"
    patched_data_yaml = _write_train_list_and_data_yaml(data_yaml, train_list, out_dir)

    model = YOLO(args.model)
    results = model.train(
        data=str(patched_data_yaml),
        epochs=args.epochs,
        imgsz=imgsz,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        batch=args.batch,
        amp=not args.no_amp,
    )
    save_dir = getattr(results, "save_dir", project_dir / args.name)
    print(f"[train] done. Best weights: {save_dir}/weights/best.pt")
    print("[train] REMINDER: run `python -m assembly_trainer.training.export` on the new "
          "best.pt and update `model_path` in your config before live testing (see §9).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="YOLO data.yaml with 'train' pointing at the raw images dir")
    parser.add_argument("--model", default="yolov8s.pt", help="base weights to fine-tune from (§9 default)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--app-config", default="config/default_config.yaml", help="source of the shared default imgsz (app.imgsz)")
    parser.add_argument("--imgsz", type=int, default=None, help="override app.imgsz from --app-config (§7: start high, e.g. 960/1280, not the 640 default)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=8,
                         help="images per batch — kept modest (8) by default since batch=16 at imgsz=960 "
                              "was observed exceeding a 6GB GPU's VRAM (spilling into slow shared system "
                              "memory) and triggering NaN losses; pass -1 for ultralytics' auto-batch, "
                              "or higher if you have more VRAM")
    parser.add_argument("--no-amp", action="store_true",
                         help="disable mixed-precision training — safer against NaN losses (slower, more "
                              "VRAM use); recommended if you hit NaN losses even after lowering --batch")
    parser.add_argument("--project", default="models/runs")
    parser.add_argument("--name", default="assembly_trainer")
    parser.add_argument("--oversample-cap", type=int, default=8, help="max repeat factor for a single image")
    parser.add_argument("--allow-unlabeled", action="store_true",
                         help="skip the check that refuses to train while any image lacks a label file "
                              "(missing labels are otherwise treated by ultralytics as confirmed-empty)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
