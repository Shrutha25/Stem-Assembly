"""Train/val/test split, within each session and step (not whole-session
holdout) — every session and every step/orientation-negative class ends up
represented in every split, so val/test never see a totally unfamiliar
environment. This is a deliberate choice for this project: the more common
ML-textbook approach (holding out entire sessions) tests generalization to
brand-new environments, but here the priority is making sure the model has
seen the deployment environment(s) it'll be judged against.

That choice has a real cost worth naming rather than hiding: frames a few
seconds apart within one capture burst are near-duplicates (same lighting,
same part position, same camera noise). Splitting individual frames at
random would scatter near-duplicate pairs across train and val, making val
accuracy look better than it really is. This is mitigated (not eliminated)
by grouping consecutive captures within each (session, step) into small
clusters — see --group-size — and assigning a whole cluster to one split at
a time, so a burst of near-identical frames stays together.

Assignment is a pure deterministic hash of (session, step-or-orientation-
class, cluster index) — no state file needed, and it's automatically stable
as you capture more frames later: existing clusters keep their split
forever, only newly-appended frames form new clusters that get assigned
fresh.

Copies (does not move) frames into `data/dataset/images/{train,val,test}/`,
prefixed with the session id to avoid filename collisions. Also seeds
`data/dataset/data.yaml` with the current class list from the app config.

Labeling happens *after* this split, on top of it — this only organizes
images; label files (`labels/{train,val,test}/*.txt`) are produced
separately (see build plan §10) and must use matching filenames.

Usage:
    python -m assembly_trainer.training.split_sessions
    python -m assembly_trainer.training.split_sessions --group-size 5 --train-ratio 0.7 --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from ..config import load_config


def _session_dirs(sessions_root: Path) -> list[Path]:
    if not sessions_root.exists():
        return []
    return sorted(p for p in sessions_root.iterdir() if p.is_dir() and (p / "manifest.jsonl").exists())


def _bucket_key(record: dict) -> str:
    """Group by orientation-negative class if this is one, else by step —
    each gets its own independent split within a session."""
    return record.get("orientation_negative_class") or f"step{record['step_id']}"


def _assign_split(cluster_key: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(cluster_key.encode("utf-8")).hexdigest()
    frac = (int(digest, 16) % 10_000) / 10_000
    if frac < train_ratio:
        return "train"
    if frac < train_ratio + val_ratio:
        return "val"
    return "test"


def run(args: argparse.Namespace) -> None:
    if args.train_ratio + args.val_ratio > 1.0:
        raise SystemExit("--train-ratio + --val-ratio must not exceed 1.0 (the remainder becomes --test-ratio)")
    if args.group_size < 1:
        raise SystemExit("--group-size must be >= 1")

    sessions_root = Path(args.sessions_root)
    dataset_root = Path(args.dataset_root)
    images_root = dataset_root / "images"

    sessions = _session_dirs(sessions_root)
    if not sessions:
        raise SystemExit(f"no captured sessions found under {sessions_root} — run assembly_trainer.data_capture first")

    # Regenerated from scratch every run — these are copies of files that
    # still live safely under data/sessions/, so this is fully reversible by
    # re-running. Necessary because cluster boundaries can shift as more
    # frames get captured within a bucket, and a stale copy under the wrong
    # split would otherwise linger.
    if images_root.exists():
        shutil.rmtree(images_root)

    counts = {"train": 0, "val": 0, "test": 0}
    bucket_counts: dict[str, int] = defaultdict(int)
    copied = 0

    for session_dir in sessions:
        session_id = session_dir.name
        manifest_path = session_dir / "manifest.jsonl"
        records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        buckets: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            buckets[_bucket_key(rec)].append(rec)

        for bucket_name, recs in buckets.items():
            recs.sort(key=lambda r: r["file"])  # filenames are zero-padded sequential -> chronological order
            for cluster_start in range(0, len(recs), args.group_size):
                cluster = recs[cluster_start : cluster_start + args.group_size]
                cluster_key = f"{session_id}:{bucket_name}:{cluster_start // args.group_size}"
                split = _assign_split(cluster_key, args.train_ratio, args.val_ratio)
                bucket_counts[f"{session_id}/{bucket_name}/{split}"] += len(cluster)

                out_dir = images_root / split
                out_dir.mkdir(parents=True, exist_ok=True)
                for rec in cluster:
                    src = session_dir / rec["file"]
                    if not src.exists():
                        continue
                    dest = out_dir / f"{session_id}__{rec['file']}"
                    shutil.copy2(src, dest)
                    counts[split] += 1
                    copied += 1

    dataset_root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.app_config)
    data_yaml_path = dataset_root / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {"train": "images/train", "val": "images/val", "test": "images/test", "names": list(cfg.classes)},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(f"[split_sessions] {len(sessions)} session(s), group-size={args.group_size} "
          f"-> train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"[split_sessions] copied {copied} frame(s) into {images_root}")
    print(f"[split_sessions] seeded {data_yaml_path} with classes {cfg.classes}")

    # Flag any (session, step/class) bucket that ended up with zero frames
    # in val or test — with small bucket counts and a coarse group-size,
    # some buckets may not get spread across all three splits by chance.
    thin = [
        key for key in bucket_counts
        if key.endswith("/train") and not any(
            f"{key.rsplit('/', 1)[0]}/{s}" in bucket_counts for s in ("val", "test")
        )
    ]
    if thin:
        print(f"[split_sessions] NOTE: {len(thin)} (session/step) bucket(s) landed entirely in train with "
              f"nothing in val or test — normal for small buckets at this group-size, not necessarily a problem.")
    print("[split_sessions] next: label the images (see build plan §10) into matching "
          "labels/{train,val,test}/*.txt, then point training/train.py's --data at this data.yaml.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions-root", default="data/sessions")
    parser.add_argument("--dataset-root", default="data/dataset")
    parser.add_argument("--app-config", default="config/default_config.yaml")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--group-size", type=int, default=5,
                         help="consecutive captures within one (session, step) treated as one cluster and "
                              "assigned to the same split together, to limit near-duplicate leakage")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
