"""End-to-end smoke test of the transformers 5.1.0 fine-tuning stack.

Fine-tunes CodeBERT on a tiny subset of Devign (function-level binary
classification) for a few steps. The goal is not accuracy; it is to prove that
the data -> tokenize -> Trainer -> evaluate -> metrics path works under
transformers 5.x and that dynamic GPU selection works on this shared server.

Runs fully offline (``HF_HUB_OFFLINE`` forced on), so it can only use artifacts
already in the local cache (CodeBERT and Devign are cached) and never downloads.

Usage:
    conda activate syssec_env
    python code/scripts/smoke_train.py
"""

from __future__ import annotations

# Force offline so this script can never trigger a download.
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.gpu import select_free_gpu  # noqa: E402

# Pick a free GPU before torch creates a CUDA context.
select_free_gpu(min_free_mib=5_000)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL = "microsoft/codebert-base"
MAX_LEN = 256
N_TRAIN = 200
N_EVAL = 200


def compute_metrics(eval_pred):
    """Compute classification metrics for the Trainer.

    Args:
        eval_pred: Tuple of (logits, labels) provided by the Trainer.

    Returns:
        Dict mapping metric name to value (accuracy, precision, recall, f1, mcc).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
    }


def main():
    """Run the smoke fine-tune and print evaluation metrics."""
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("visible device:", torch.cuda.get_device_name(0))

    # Devign (DetectVul/devign, parquet-based so it loads under datasets 5.x).
    # Fields: 'func' (code), 'target' (bool), 'project', 'commit_id', plus
    # line-level 'vul_lines' we may use later.
    ds = load_dataset("DetectVul/devign")
    print("dataset splits:", {k: len(v) for k, v in ds.items()})

    train = ds["train"].shuffle(seed=42).select(range(N_TRAIN))
    eval_ = ds["validation"].shuffle(seed=42).select(range(N_EVAL))

    tok = AutoTokenizer.from_pretrained(MODEL)

    def preprocess(batch):
        enc = tok(batch["func"], truncation=True, max_length=MAX_LEN)
        enc["labels"] = [int(t) for t in batch["target"]]
        return enc

    keep = ["input_ids", "attention_mask", "labels"]
    train = train.map(preprocess, batched=True, remove_columns=train.column_names)
    eval_ = eval_.map(preprocess, batched=True, remove_columns=eval_.column_names)
    train.set_format("torch", columns=keep)
    eval_.set_format("torch", columns=keep)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tok)

    args = TrainingArguments(
        output_dir="/tmp/deepvul_smoke",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",          # 5.x name (was evaluation_strategy)
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        bf16=torch.cuda.is_available(),  # H200 supports bf16
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=eval_,
        data_collator=collator,
        processing_class=tok,           # 5.x name (was tokenizer=)
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\n=== SMOKE METRICS (tiny subset, not meaningful accuracy) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nSmoke test PASSED: 5.x training stack + GPU selection + metrics all work.")


if __name__ == "__main__":
    main()
