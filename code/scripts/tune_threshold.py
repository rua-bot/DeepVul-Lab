"""Tune the decision threshold on validation and report test metrics.

The default argmax decision rule (threshold 0.5 on the positive-class
probability) is rarely optimal for F1 / MCC on imbalanced data. This script
reuses the already-trained best checkpoints of a run, recomputes validation and
test probabilities, selects the threshold that maximizes the target metric on
the validation split, and reports test metrics at that threshold versus the
default. Tuning on validation and reporting on test keeps the protocol honest.

No retraining is performed. Threshold-independent metrics (ROC-AUC, PR-AUC) are
unchanged and shown for reference.

Runs offline; a free GPU is selected automatically on this shared server.

Usage:
    conda activate syssec_env
    python code/scripts/tune_threshold.py \
        --run-dir outputs/runs/diversevul_codebert_fullfunc \
        --model microsoft/codebert-base --dataset diversevul \
        --text-field func --metric f1
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
from utils.gpu import select_free_gpu  # noqa: E402

select_free_gpu(min_free_mib=10_000)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from metrics import best_threshold, metrics_at_threshold  # noqa: E402
from training.experiment import PadCollator, get_tokenized  # noqa: E402

AGG_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
               "roc_auc", "pr_auc"]


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Run directory containing seed* subdirs")
    p.add_argument("--model", required=True, help="HF model id used for tokenization/cache key")
    p.add_argument("--dataset", required=True, choices=["devign", "diversevul"])
    p.add_argument("--text-field", default="func")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--metric", default="f1", choices=["f1", "mcc"],
                   help="Validation metric to maximize when picking the threshold")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def find_checkpoint(seed_dir: Path) -> Path:
    """Return the single best checkpoint directory inside a seed run.

    Args:
        seed_dir: Per-seed run directory.

    Returns:
        Path to the checkpoint directory (the best model kept by the Trainer).

    Raises:
        FileNotFoundError: If no checkpoint is present.
    """
    cps = sorted((seed_dir / "checkpoints").glob("checkpoint-*"))
    if not cps:
        raise FileNotFoundError(f"No checkpoint under {seed_dir}")
    return cps[-1]


@torch.no_grad()
def predict_probs(model, dataset, pad_id: int, device, batch_size: int):
    """Run inference and return positive-class probabilities and labels.

    Args:
        model: A sequence-classification model on ``device``.
        dataset: Torch-formatted dataset with input_ids/attention_mask/labels.
        pad_id: Padding token id for the collator.
        device: Torch device.
        batch_size: Inference batch size.

    Returns:
        Tuple (pos_prob, labels) of 1-D numpy arrays.
    """
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=PadCollator(pad_id))
    probs, labels = [], []
    for batch in loader:
        labels.append(batch["labels"].numpy())
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def aggregate(per_seed: list[dict]) -> dict:
    """Aggregate per-seed metric dicts into mean and std."""
    agg = {}
    for m in AGG_METRICS:
        vals = [d[m] for d in per_seed if m in d and not np.isnan(d[m])]
        if vals:
            agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def main():
    """Tune the threshold per seed and report default vs. tuned test metrics."""
    args = parse_args()
    run_dir = (ROOT / args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    seed_dirs = sorted(d for d in run_dir.glob("seed*") if d.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"No seed* dirs under {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    tok_splits = get_tokenized(
        args.dataset, args.model, args.max_len, args.text_field,
        num_proc=1, trust_remote_code=args.trust_remote_code,
    )
    keep = ["input_ids", "attention_mask", "labels"]
    val_ds = tok_splits["validation"].with_format(
        "torch", columns=[c for c in keep if c in tok_splits["validation"].column_names])
    test_ds = tok_splits["test"].with_format(
        "torch", columns=[c for c in keep if c in tok_splits["test"].column_names])

    default_seed, tuned_seed, thresholds, val_seed = [], [], [], []
    for seed_dir in seed_dirs:
        ckpt = find_checkpoint(seed_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt), trust_remote_code=args.trust_remote_code).to(device).eval()

        val_prob, val_lab = predict_probs(model, val_ds, pad_id, device, args.eval_batch_size)
        test_prob, test_lab = predict_probs(model, test_ds, pad_id, device, args.eval_batch_size)

        t = best_threshold(val_prob, val_lab, metric=args.metric)
        thresholds.append(t)
        # Validation metrics at the default threshold support honest model
        # selection (e.g. choosing the best imbalance method) without peeking
        # at the test split.
        val_seed.append(metrics_at_threshold(val_prob, val_lab, 0.5))
        default_seed.append(metrics_at_threshold(test_prob, test_lab, 0.5))
        tuned_seed.append(metrics_at_threshold(test_prob, test_lab, t))
        print(f"  {seed_dir.name}: tuned threshold={t:.3f} "
              f"(val {args.metric}-opt) | test f1 {default_seed[-1]['f1']:.4f} "
              f"-> {tuned_seed[-1]['f1']:.4f}, mcc {default_seed[-1]['mcc']:.4f} "
              f"-> {tuned_seed[-1]['mcc']:.4f}")
        del model
        torch.cuda.empty_cache()

    summary = {
        "run_dir": str(run_dir),
        "model": args.model,
        "dataset": args.dataset,
        "text_field": args.text_field,
        "tuned_on": f"validation ({args.metric})",
        "thresholds": thresholds,
        "validation_default": aggregate(val_seed),
        "default": aggregate(default_seed),
        "tuned": aggregate(tuned_seed),
        "validation_default_per_seed": val_seed,
        "default_per_seed": default_seed,
        "tuned_per_seed": tuned_seed,
    }
    out = run_dir / f"threshold_tuned_{args.metric}.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n==== {run_dir.name}: default(0.5) vs tuned(val-{args.metric}) test metrics ====")
    print(f"{'metric':<18} {'default':>18} {'tuned':>18}")
    for m in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc"]:
        d, tt = summary["default"][m], summary["tuned"][m]
        print(f"{m:<18} {d['mean']:.4f} +/- {d['std']:.4f}   {tt['mean']:.4f} +/- {tt['std']:.4f}")
    print(f"mean tuned threshold: {np.mean(thresholds):.3f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
