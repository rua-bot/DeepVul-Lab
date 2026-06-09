"""Locate functions that stall the VulBERTa libclang tokenizer.

Tokenizes a range of DiverseVul training functions one by one, printing any that
take unusually long. Used to find the input(s) that hang multi-process mapping.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.jsonl_dataset import load_processed  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

START, END = 200_000, 220_000


def main():
    """Tokenize the suspect index range, flagging slow functions."""
    tok = AutoTokenizer.from_pretrained("claudios/VulBERTa-mlm", trust_remote_code=True)
    funcs = load_processed("diversevul")["train"]["func"]
    print("total", len(funcs), flush=True)
    for i in range(START, min(END, len(funcs))):
        t = time.time()
        tok(funcs[i][:20000], truncation=True, max_length=512)
        dt = time.time() - t
        if dt > 1.0 or i % 1000 == 0:
            print(f"idx {i}: {dt:.2f}s len={len(funcs[i])}", flush=True)
    print("DONE range", flush=True)


if __name__ == "__main__":
    main()
