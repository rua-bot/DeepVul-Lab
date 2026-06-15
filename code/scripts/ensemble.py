"""Soft-vote ensemble of the per-seed predictions already saved by train.py.

Each ``train.py`` run stores per-seed test logits (``seed*/test_logits.npy``) and
labels. Averaging the per-seed positive-class probabilities (a soft vote) is a
zero-extra-training way to reduce seed variance and usually improves the
threshold-independent ranking metrics (PR-AUC / ROC-AUC) over any single seed.

The script reports the ensemble metrics at the default 0.5 threshold and, if
asked, at a threshold tuned on a held-out signal. By default it tunes the
threshold on the test labels purely for the ranking-independent "best achievable
work point" reference; for the honest protocol use ``--val-from`` to instead pick
the threshold on a sibling run's validation logits (not used here by default).

Usage:
    python code/scripts/ensemble.py \
        --run-dir outputs/runs/diversevul_unixcoder_fullfunc
    python code/scripts/ensemble.py \
        --run-dir outputs/runs/devign_unixcoder_focal --tune-metric mcc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import best_threshold, metrics_at_threshold  # noqa: E402

REPORT_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1",
                  "mcc", "roc_auc", "pr_auc"]


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Run directory with seed*/test_logits.npy and test_labels.npy")
    p.add_argument("--tune-metric", default=None, choices=[None, "f1", "mcc"],
                   help="If set, also report metrics at the test-optimal threshold "
                        "(reference upper bound; the honest work point comes from "
                        "validation-tuned thresholds in tune_threshold.py)")
    return p.parse_args()


def load_seed_probs(run_dir: Path):
    """Load and align per-seed positive-class probabilities and labels.

    Args:
        run_dir: Run directory containing seed* subdirectories.

    Returns:
        Tuple (prob_stack, labels): prob_stack is (n_seeds, n) of positive-class
        probabilities; labels is the shared (n,) label vector.

    Raises:
        FileNotFoundError: If no per-seed logits are present.
        ValueError: If seeds disagree on labels (misaligned test sets).
    """
    seed_dirs = sorted(d for d in run_dir.glob("seed*") if d.is_dir())
    probs, labels_ref = [], None
    for sd in seed_dirs:
        logit_path = sd / "test_logits.npy"
        label_path = sd / "test_labels.npy"
        if not logit_path.exists() or not label_path.exists():
            continue
        logits = np.load(logit_path)
        labels = np.load(label_path)
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise ValueError(f"Label mismatch across seeds in {run_dir}")
        probs.append(softmax(logits, axis=-1)[:, 1])
    if not probs:
        raise FileNotFoundError(f"No per-seed test_logits.npy under {run_dir}")
    return np.stack(probs, axis=0), labels_ref


def main():
    """Compute and save the soft-vote ensemble metrics for one run."""
    args = parse_args()
    run_dir = (ROOT / args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    prob_stack, labels = load_seed_probs(run_dir)
    mean_prob = prob_stack.mean(axis=0)
    n_seeds = prob_stack.shape[0]

    ens_default = metrics_at_threshold(mean_prob, labels, 0.5)
    summary = {
        "run_dir": str(run_dir),
        "n_seeds": n_seeds,
        "method": "soft_vote_mean_prob",
        "ensemble_default": ens_default,
    }
    if args.tune_metric:
        t = best_threshold(mean_prob, labels, metric=args.tune_metric)
        summary["ensemble_tuned"] = metrics_at_threshold(mean_prob, labels, t)
        summary["tuned_metric"] = args.tune_metric

    out = run_dir / "ensemble.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"==== {run_dir.name}: {n_seeds}-seed soft-vote ensemble (default 0.5) ====")
    for m in REPORT_METRICS:
        print(f"  {m:<18} {ens_default[m]:.4f}")
    if args.tune_metric:
        print(f"-- at test-optimal threshold ({args.tune_metric}, reference only) --")
        for m in REPORT_METRICS:
            print(f"  {m:<18} {summary['ensemble_tuned'][m]:.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
