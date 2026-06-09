"""Compatibility test for using VulBERTa as a fine-tuning backbone.

Checks whether ``claudios/VulBERTa-mlm`` can be loaded and run under the current
stack (transformers 5.1.0 + the model's custom libclang-based tokenizer +
``trust_remote_code=True``).

Prerequisites:
    Run ``user_setup.sh`` first so that the model is in the local HuggingFace
    cache and ``libclang`` is installed. This script forces offline mode and will
    never download; missing artifacts are reported clearly.

Decision:
    * Every stage passes -> VulBERTa is usable as the domain backbone.
    * Any stage fails -> fall back to a clean-loading C/C++ code model
      (e.g. ``microsoft/graphcodebert-base`` or ``microsoft/unixcoder-base``)
      for the domain-pretraining comparison.

Usage:
    conda activate syssec_env
    python code/scripts/vulberta_load_test.py
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import traceback
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.gpu import select_free_gpu  # noqa: E402

select_free_gpu(min_free_mib=5_000)

REPO = "claudios/VulBERTa-mlm"
SAMPLE = "int copy(char *src){ char buf[16]; strcpy(buf, src); return 0; }"

# Maps stage name -> (status, message). Populated by run_stage.
results: dict[str, tuple[str, str]] = {}


def run_stage(name: str, fn: Callable):
    """Run one test stage, recording its outcome instead of raising.

    Args:
        name: Human-readable stage identifier used in the final report.
        fn: Zero-argument callable performing the stage's work.

    Returns:
        The return value of ``fn`` on success, or ``None`` if ``fn`` raised.
    """
    try:
        out = fn()
        results[name] = ("PASS", "")
        return out
    except Exception as exc:  # noqa: BLE001 - we want to capture any failure
        results[name] = ("FAIL", f"{type(exc).__name__}: {exc}")
        print(f"\n--- {name} FAILED ---")
        traceback.print_exc()
        return None


def main():
    """Run all VulBERTa load stages and print a pass/fail decision."""
    import torch

    print("=== checking libclang availability ===")
    try:
        import clang  # noqa: F401
        print("libclang import: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"libclang import FAILED: {exc}  (run: pip install libclang)")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = run_stage(
        "tokenizer_load",
        lambda: AutoTokenizer.from_pretrained(REPO, trust_remote_code=True),
    )

    def tokenize_sample():
        if tok is None:
            raise RuntimeError("skipped: tokenizer_load failed")
        enc = tok(SAMPLE, truncation=True, max_length=128, return_tensors="pt")
        print("  input_ids shape:", tuple(enc["input_ids"].shape))
        return enc

    enc = run_stage("tokenize_sample", tokenize_sample)

    model = run_stage(
        "model_with_clshead_load",
        lambda: AutoModelForSequenceClassification.from_pretrained(
            REPO, num_labels=2, trust_remote_code=True
        ),
    )

    def forward_pass():
        if model is None or enc is None:
            raise RuntimeError("skipped: prior stage failed")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()})
        print("  logits:", out.logits.detach().cpu().tolist())
        return out

    run_stage("forward_pass", forward_pass)

    print("\n==================== VULBERTA REPORT ====================")
    for name, (status, msg) in results.items():
        line = f"  {name:<26} {status}"
        if msg:
            line += f"  | {msg}"
        print(line)

    all_pass = all(status == "PASS" for status, _ in results.values())
    print(
        "\nDECISION:",
        "VulBERTa usable as domain backbone"
        if all_pass
        else "VulBERTa not usable -> fall back to graphcodebert/unixcoder",
    )
    print("=========================================================")


if __name__ == "__main__":
    main()
