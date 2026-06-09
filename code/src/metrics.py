"""Classification metrics for vulnerability detection.

Because vulnerability datasets are often highly imbalanced, accuracy alone is
misleading. These helpers report precision, recall, F1, MCC and PR-AUC (average
precision) alongside accuracy and ROC-AUC, plus the confusion-matrix counts.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Compute the full metric suite from raw logits.

    Args:
        logits: Array of shape (n, 2) with class logits.
        labels: Array of shape (n,) with ground-truth binary labels.

    Returns:
        Dict with accuracy, precision, recall, f1, mcc, roc_auc, pr_auc and the
        confusion-matrix counts tn/fp/fn/tp.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    pos_prob = softmax(logits, axis=-1)[:, 1]
    preds = logits.argmax(axis=-1)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    # AUC metrics need both classes present in labels.
    if len(np.unique(labels)) == 2:
        out["roc_auc"] = roc_auc_score(labels, pos_prob)
        out["pr_auc"] = average_precision_score(labels, pos_prob)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def metrics_at_threshold(pos_prob: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Compute the metric suite using a custom decision threshold.

    The positive class is predicted when ``pos_prob >= threshold``. ROC-AUC and
    PR-AUC are threshold-independent and computed from the probabilities.

    Args:
        pos_prob: Array of shape (n,) with positive-class probabilities.
        labels: Array of shape (n,) with ground-truth binary labels.
        threshold: Decision threshold in [0, 1].

    Returns:
        Dict with accuracy, precision, recall, f1, mcc, roc_auc, pr_auc, the
        confusion-matrix counts tn/fp/fn/tp, and the threshold used.
    """
    pos_prob = np.asarray(pos_prob)
    labels = np.asarray(labels)
    preds = (pos_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(labels)) == 2:
        out["roc_auc"] = roc_auc_score(labels, pos_prob)
        out["pr_auc"] = average_precision_score(labels, pos_prob)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def best_threshold(pos_prob: np.ndarray, labels: np.ndarray, metric: str = "f1",
                   n_grid: int = 200) -> float:
    """Find the decision threshold maximizing a metric on the given split.

    Args:
        pos_prob: Positive-class probabilities, shape (n,).
        labels: Ground-truth binary labels, shape (n,).
        metric: Metric to maximize ("f1" or "mcc").
        n_grid: Number of evenly spaced candidate thresholds in (0, 1).

    Returns:
        The threshold achieving the highest value of ``metric``.
    """
    pos_prob = np.asarray(pos_prob)
    labels = np.asarray(labels)
    grid = np.linspace(0.01, 0.99, n_grid)
    best_t, best_v = 0.5, -np.inf
    for t in grid:
        preds = (pos_prob >= t).astype(int)
        # Compute only the cheap, threshold-dependent target metric in the loop;
        # threshold-independent AUCs are intentionally not recomputed here.
        v = f1_score(labels, preds, zero_division=0) if metric == "f1" \
            else matthews_corrcoef(labels, preds)
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t)


def make_compute_metrics():
    """Return a ``compute_metrics`` callable compatible with the HF Trainer.

    Returns:
        Function mapping an (logits, labels) eval prediction to a metrics dict.
    """
    def compute(eval_pred):
        logits, labels = eval_pred
        return metrics_from_logits(logits, labels)

    return compute
