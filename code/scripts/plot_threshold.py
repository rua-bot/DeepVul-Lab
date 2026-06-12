"""Visualize the decision-threshold trade-off for the CodeBERT baseline.

For each dataset this plots how Precision, Recall, F1 and MCC vary as the
decision threshold sweeps over (0, 1), using the saved test-set logits of the
CodeBERT full-function runs (averaged over seeds). PR-AUC and ROC-AUC are drawn
as horizontal reference lines because they are threshold-independent. The figure
makes the RQ4 point explicit: moving the operating point trades F1 against MCC
(their optima sit at different thresholds), while the model's ranking ability
(PR-AUC / ROC-AUC) is unchanged.

Runs offline; no GPU required.

Usage:
    conda activate syssec_env
    python code/scripts/plot_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
SEEDS = [42, 1, 2]
GRID = np.linspace(0.01, 0.99, 197)


def load_probs(dataset: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load per-seed positive-class probabilities and labels for a dataset.

    Args:
        dataset: Dataset name ("devign" or "diversevul").

    Returns:
        Tuple of (list of pos_prob arrays, list of label arrays), one per seed.
    """
    probs, labels = [], []
    for s in SEEDS:
        d = RUNS / f"{dataset}_codebert_fullfunc" / f"seed{s}"
        logits = np.load(d / "test_logits.npy")
        probs.append(softmax(logits, axis=-1)[:, 1])
        labels.append(np.load(d / "test_labels.npy"))
    return probs, labels


def sweep(probs: list[np.ndarray], labels: list[np.ndarray]) -> dict:
    """Compute seed-averaged threshold-dependent metric curves.

    Args:
        probs: Per-seed positive-class probabilities.
        labels: Per-seed ground-truth labels.

    Returns:
        Dict with mean curves for precision/recall/f1/mcc over the grid plus
        the scalar (threshold-independent) roc_auc and pr_auc.
    """
    curves = {k: [] for k in ("precision", "recall", "f1", "mcc")}
    for t in GRID:
        per = {k: [] for k in curves}
        for p, y in zip(probs, labels):
            pred = (p >= t).astype(int)
            per["precision"].append(precision_score(y, pred, zero_division=0))
            per["recall"].append(recall_score(y, pred, zero_division=0))
            per["f1"].append(f1_score(y, pred, zero_division=0))
            per["mcc"].append(matthews_corrcoef(y, pred))
        for k in curves:
            curves[k].append(np.mean(per[k]))
    curves = {k: np.array(v) for k, v in curves.items()}
    curves["roc_auc"] = float(np.mean([roc_auc_score(y, p) for p, y in zip(probs, labels)]))
    curves["pr_auc"] = float(np.mean([average_precision_score(y, p) for p, y in zip(probs, labels)]))
    return curves


def plot_panel(ax, dataset: str, curves: dict) -> None:
    """Draw one dataset's metric-vs-threshold panel.

    Args:
        ax: Matplotlib axis.
        dataset: Dataset name for the title.
        curves: Output of :func:`sweep`.
    """
    ax.plot(GRID, curves["f1"], label="F1", color="#1f77b4", lw=2)
    ax.plot(GRID, curves["mcc"], label="MCC", color="#d62728", lw=2)
    ax.plot(GRID, curves["precision"], label="Precision", color="#2ca02c", lw=1.3, ls="--")
    ax.plot(GRID, curves["recall"], label="Recall", color="#ff7f0e", lw=1.3, ls="--")

    ax.axhline(curves["pr_auc"], color="#7f7f7f", lw=1.2, ls=":",
               label=f"PR-AUC={curves['pr_auc']:.3f} (thr-indep.)")
    ax.axhline(curves["roc_auc"], color="#17becf", lw=1.2, ls=":",
               label=f"ROC-AUC={curves['roc_auc']:.3f} (thr-indep.)")

    t_f1 = GRID[int(np.argmax(curves["f1"]))]
    t_mcc = GRID[int(np.argmax(curves["mcc"]))]
    ax.axvline(0.5, color="black", lw=1.0, alpha=0.5)
    ax.text(0.5, 1.005, "default 0.5", rotation=90, va="bottom", ha="right",
            fontsize=7, color="black", alpha=0.7, transform=ax.get_xaxis_transform())
    ax.axvline(t_f1, color="#1f77b4", lw=1.0, ls="-.", alpha=0.7)
    ax.axvline(t_mcc, color="#d62728", lw=1.0, ls="-.", alpha=0.7)
    ax.annotate(f"F1-opt\n@{t_f1:.2f}", xy=(t_f1, np.max(curves["f1"])),
                xytext=(4, 0), textcoords="offset points", fontsize=7, color="#1f77b4")
    ax.annotate(f"MCC-opt\n@{t_mcc:.2f}", xy=(t_mcc, np.max(curves["mcc"])),
                xytext=(4, -18), textcoords="offset points", fontsize=7, color="#d62728")

    ax.set_title(f"{dataset} (CodeBERT, seed-avg)")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")


def main():
    """Render the two-panel threshold trade-off figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dataset in zip(axes, ("devign", "diversevul")):
        probs, labels = load_probs(dataset)
        plot_panel(ax, dataset, sweep(probs, labels))
    fig.suptitle("Decision-threshold trade-off: F1 vs MCC optima diverge; "
                 "ranking metrics (PR-AUC / ROC-AUC) are threshold-independent")
    fig.tight_layout()
    out = ROOT / "outputs" / "threshold_tradeoff.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
