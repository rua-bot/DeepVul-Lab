"""Augment processed splits with key-context and structure-marked variants.

Reads each processed split (code/data/processed/<name>/{split}.jsonl), computes
two extra fields per record and rewrites the JSONL in place:

    func_sliced: vulnerability-focused slice (see data.slicing.extract_key_context)
    func_marked: full function with seed lines wrapped in sink markers

The original ``func`` field is preserved, so the full-function baseline is
unaffected and the three input variants share the same records / splits. A
tokenizer-based length report quantifies how much the slice reduces the share of
functions exceeding the model's 512-token limit, which is the motivation for the
variant.

Runs offline; no GPU required.

Usage:
    conda activate syssec_env
    python code/scripts/build_sliced.py
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.slicing import add_structure_markers, extract_key_context  # noqa: E402

PROCESSED_ROOT = ROOT / "data" / "processed"
STATS_MODEL = "microsoft/codebert-base"
MAX_LEN = 512


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with path.open() as f:
        return [json.loads(line) for line in f]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file, one object per line."""
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _truncation_rate(tokenizer, texts: list[str]) -> float:
    """Fraction of texts whose tokenization exceeds the model's length limit.

    Args:
        tokenizer: A HuggingFace tokenizer.
        texts: Input strings to measure.

    Returns:
        Share of inputs longer than ``MAX_LEN`` tokens, in [0, 1].
    """
    over = 0
    for t in texts:
        n = len(tokenizer(t, truncation=False)["input_ids"])
        if n > MAX_LEN:
            over += 1
    return over / max(1, len(texts))


def augment_dataset(name: str, tokenizer, sample: int) -> dict:
    """Add sliced/marked fields to every split of one dataset and report stats.

    Args:
        name: Processed dataset directory name.
        tokenizer: Tokenizer used only for the length report (may be None).
        sample: Number of records per split to use for the length report.

    Returns:
        Per-split truncation-rate stats for the full vs. sliced variants.
    """
    stats: dict = {}
    for split in ("train", "validation", "test"):
        path = PROCESSED_ROOT / name / f"{split}.jsonl"
        records = _read_jsonl(path)
        for rec in records:
            func = rec["func"]
            rec["func_sliced"] = extract_key_context(func)
            rec["func_marked"] = add_structure_markers(func)
        _write_jsonl(path, records)

        entry: dict = {"n": len(records)}
        if tokenizer is not None:
            subset = records[:sample]
            entry["trunc_full"] = round(
                _truncation_rate(tokenizer, [r["func"] for r in subset]), 4
            )
            entry["trunc_sliced"] = round(
                _truncation_rate(tokenizer, [r["func_sliced"] for r in subset]), 4
            )
        stats[split] = entry
        print(f"  [{name}/{split}] augmented {len(records)} records "
              f"-> {entry}")
    return stats


def main():
    """Augment both datasets and print a truncation-reduction report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-tokenizer-stats", action="store_true",
                   help="Skip the (slower) token-length truncation report")
    p.add_argument("--sample", type=int, default=2000,
                   help="Records per split used for the length report")
    args = p.parse_args()

    tokenizer = None
    if not args.no_tokenizer_stats:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(STATS_MODEL)

    report = {}
    for name in ("devign", "diversevul"):
        print(f"=== {name} ===")
        report[name] = augment_dataset(name, tokenizer, args.sample)

    out = PROCESSED_ROOT / "slicing_stats.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote slicing stats to {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
