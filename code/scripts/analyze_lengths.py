"""Analyze function token-length distributions for Devign and DiverseVul.

Tokenizes every function with the CodeBERT tokenizer and reports how many
functions exceed the 512-token model limit. This quantifies whether truncation
(and therefore key-context extraction) is actually needed, or whether most
functions already fit.

Outputs a histogram PNG and per-dataset statistics. Runs offline.

Usage:
    conda activate syssec_env
    python code/scripts/analyze_lengths.py
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.loaders import load_devign, load_diversevul  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

TOKENIZER = "microsoft/codebert-base"
MODEL_LIMIT = 512
OUT_DIR = ROOT / "outputs"


def token_lengths(funcs: list[str], tok, batch_size: int = 2048) -> np.ndarray:
    """Return per-function token counts (without special tokens).

    Args:
        funcs: Function source strings.
        tok: A fast HuggingFace tokenizer.
        batch_size: Number of functions to tokenize per batch.

    Returns:
        Array of token lengths, one per input function.
    """
    lengths: list[int] = []
    for start in range(0, len(funcs), batch_size):
        batch = funcs[start : start + batch_size]
        enc = tok(batch, add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
        if (start // batch_size) % 20 == 0:
            print(f"    tokenized {min(start + batch_size, len(funcs))}/{len(funcs)}")
    return np.array(lengths)


def summarize(name: str, lengths: np.ndarray) -> dict:
    """Compute and print summary statistics for a length array.

    Args:
        name: Dataset name for logging.
        lengths: Array of per-function token lengths.

    Returns:
        Dict of summary statistics.
    """
    over = int((lengths > MODEL_LIMIT).sum())
    stats = {
        "n": int(lengths.size),
        "mean": round(float(lengths.mean()), 1),
        "median": int(np.median(lengths)),
        "p90": int(np.percentile(lengths, 90)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        f"over_{MODEL_LIMIT}": over,
        f"over_{MODEL_LIMIT}_pct": round(100 * over / lengths.size, 2),
    }
    print(f"\n[{name}] {json.dumps(stats)}")
    return stats


def main():
    """Tokenize both datasets, print stats, and save a histogram figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    print("Loading Devign...")
    devign = load_devign()
    devign_funcs = [r["func"] for split in devign.values() for r in split]
    print("Loading DiverseVul...")
    diversevul = [r["func"] for r in load_diversevul()]

    print("Tokenizing Devign...")
    dv_len = token_lengths(devign_funcs, tok)
    print("Tokenizing DiverseVul...")
    div_len = token_lengths(diversevul, tok)

    all_stats = {
        "tokenizer": TOKENIZER,
        "model_limit": MODEL_LIMIT,
        "devign": summarize("devign", dv_len),
        "diversevul": summarize("diversevul", div_len),
    }
    (OUT_DIR / "length_stats.json").write_text(json.dumps(all_stats, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, lengths) in zip(axes, [("Devign", dv_len), ("DiverseVul", div_len)]):
        clipped = np.clip(lengths, 0, 1500)
        ax.hist(clipped, bins=60, color="#4C78A8", edgecolor="white")
        ax.axvline(MODEL_LIMIT, color="crimson", linestyle="--", label=f"limit={MODEL_LIMIT}")
        over_pct = 100 * (lengths > MODEL_LIMIT).mean()
        ax.set_title(f"{name} token lengths (>{MODEL_LIMIT}: {over_pct:.1f}%)")
        ax.set_xlabel("tokens (clipped at 1500)")
        ax.set_ylabel("functions")
        ax.legend()
    fig.tight_layout()
    fig_path = OUT_DIR / "length_hist.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\nSaved histogram to {fig_path}")
    print(f"Saved stats to {OUT_DIR / 'length_stats.json'}")


if __name__ == "__main__":
    main()
