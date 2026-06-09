"""Build cleaned, split, deduplicated datasets and write them to disk.

Pipeline per dataset:
  * DiverseVul: load -> exact dedup -> near-dup dedup -> project-level split.
    (Deduplicating before splitting also removes cross-project near-duplicate
    leakage; the project-level split keeps each project within one split.)
  * Devign: load (keep its train/val/test) -> dedup across splits in priority
    order so cross-split leakage is removed while the train copy is kept.

Outputs, under code/data/processed/<dataset>/:
  * train.jsonl, validation.jsonl, test.jsonl  (one record per line)
  * stats.json  (raw/after-dedup counts and per-split label balance)

Runs offline; no GPU required.

Usage:
    conda activate syssec_env
    python code/scripts/build_dataset.py
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.dedup import dedup_across_splits, exact_dedup, near_dedup  # noqa: E402
from data.loaders import load_devign, load_diversevul  # noqa: E402
from data.splitting import project_level_split, split_label_stats  # noqa: E402

OUT_ROOT = ROOT / "data" / "processed"
SEED = 42


def write_splits(name: str, splits: dict[str, list[dict]], extra_stats: dict) -> None:
    """Write split JSONL files and a stats.json for one dataset.

    Args:
        name: Dataset name (used as the output subdirectory).
        splits: Mapping from split name to records.
        extra_stats: Additional stats to merge into stats.json.
    """
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, recs in splits.items():
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats = {"dataset": name, "seed": SEED, **extra_stats,
             "splits": split_label_stats(splits)}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n[{name}] wrote splits to {out_dir}")
    print(json.dumps(stats, indent=2))


def build_diversevul() -> None:
    """Build the DiverseVul splits with dedup-before-project-split."""
    print("=== DiverseVul ===")
    records = load_diversevul()
    raw_n = len(records)
    records, n_exact = exact_dedup(records)
    print(f"  exact dedup removed {n_exact}, remaining {len(records)}")
    records, n_near = near_dedup(records)
    print(f"  near dedup removed {n_near}, remaining {len(records)}")
    splits = project_level_split(records, ratios=(0.8, 0.1, 0.1), seed=SEED)
    write_splits(
        "diversevul",
        splits,
        {
            "raw_count": raw_n,
            "removed_exact": n_exact,
            "removed_near": n_near,
            "after_dedup": len(records),
            "split_strategy": "project_level",
        },
    )


def build_devign() -> None:
    """Build the Devign splits, keeping its split scheme but removing leakage."""
    print("=== Devign ===")
    devign = load_devign()
    raw_n = sum(len(v) for v in devign.values())
    order = [s for s in ("train", "validation", "test") if s in devign]
    ordered = [(s, devign[s]) for s in order]
    splits, removed = dedup_across_splits(ordered)
    print(f"  dedup across splits removed {removed}, remaining "
          f"{sum(len(v) for v in splits.values())}")
    write_splits(
        "devign",
        splits,
        {
            "raw_count": raw_n,
            "removed_dedup": removed,
            "after_dedup": sum(len(v) for v in splits.values()),
            "split_strategy": "predefined_leakage_cleaned",
        },
    )


def main():
    """Build both datasets."""
    build_devign()
    build_diversevul()
    print("\nAll processed datasets written under", OUT_ROOT)


if __name__ == "__main__":
    main()
