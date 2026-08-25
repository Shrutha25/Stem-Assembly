"""Sanity-check labels against the images `split_sessions.py` organized,
before wasting a training run on a structural mistake (see build plan §10 —
labeling happens externally; this is the seam where a class-id typo or a
mismatched filename would otherwise go unnoticed until training misbehaves).

Checks, per split (train/val/test):
  - every image has a matching label file (an empty .txt is fine — that's a
    deliberate negative with nothing to label)
  - every label file has a matching image (flags orphans, e.g. from a
    renamed/removed image)
  - every line in every label file is well-formed: exactly 5 fields, class
    id within range, all four coordinates in [0, 1]
  - reports per-class instance counts per split, to eyeball coverage/balance
    before training

Usage:
    python -m assembly_trainer.training.verify_labels
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ..config import load_config


def _check_split(split: str, images_dir: Path, labels_dir: Path, num_classes: int) -> tuple[int, int, list[str], Counter]:
    errors: list[str] = []
    class_counts: Counter = Counter()

    image_stems = {p.stem for p in images_dir.glob("*.jpg")} if images_dir.exists() else set()
    label_stems = {p.stem for p in labels_dir.glob("*.txt")} if labels_dir.exists() else set()

    unlabeled = image_stems - label_stems
    orphans = label_stems - image_stems
    for stem in sorted(orphans):
        errors.append(f"[{split}] {stem}.txt has no matching image")

    for stem in sorted(image_stems & label_stems):
        label_path = labels_dir / f"{stem}.txt"
        for lineno, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            ctx = f"[{split}] {stem}.txt:{lineno}"
            if len(parts) != 5:
                errors.append(f"{ctx}: expected 5 fields, got {len(parts)}")
                continue
            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                errors.append(f"{ctx}: non-numeric field(s)")
                continue
            if not (0 <= class_id < num_classes):
                errors.append(f"{ctx}: class id {class_id} out of range (0-{num_classes - 1})")
                continue
            if any(not (0.0 <= c <= 1.0) for c in coords):
                errors.append(f"{ctx}: coordinates must be normalized 0-1, got {coords}")
                continue
            class_counts[class_id] += 1

    return len(image_stems), len(unlabeled), errors, class_counts


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.app_config)
    dataset_root = Path(args.dataset_root)

    all_errors: list[str] = []
    print(f"{'split':<8} {'images':>8} {'unlabeled':>10}")
    totals: Counter = Counter()
    for split in ("train", "val", "test"):
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        total, unlabeled, errors, class_counts = _check_split(split, images_dir, labels_dir, len(cfg.classes))
        print(f"{split:<8} {total:>8} {unlabeled:>10}")
        all_errors.extend(errors)
        totals.update(class_counts)

    print()
    print("per-class instance counts (all splits combined):")
    for i, cls in enumerate(cfg.classes):
        print(f"  {i}: {cls:<28} {totals.get(i, 0)}")
        if totals.get(i, 0) == 0:
            all_errors.append(f"class '{cls}' (id {i}) has zero labeled instances anywhere")

    print()
    if all_errors:
        print(f"VERIFY_LABELS: {len(all_errors)} issue(s) found")
        for e in all_errors[:50]:
            print(f"  - {e}")
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors) - 50} more")
        raise SystemExit(1)
    print("VERIFY_LABELS: OK — no structural issues found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default="data/dataset")
    parser.add_argument("--app-config", default="config/default_config.yaml")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
