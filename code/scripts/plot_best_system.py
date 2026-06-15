"""Plot the RQ5 improvement evidence: the ladder and baseline-vs-final gains.

Reads outputs/best_system.json (the cumulative ladder) and the relevant
ensemble.json files, then renders one figure with two panels:

  (a) DiverseVul improvement ladder S0->S4 as lines over F1 / MCC / PR-AUC,
      showing a clean monotonic climb on the strongly imbalanced set.
  (b) Devign baseline vs. final system as grouped bars, where the structure
      backbone plus ensembling already delivers the gain on the balanced set.

Output: outputs/best_system_ladder.png

Usage:
    conda activate syssec_env
    python code/scripts/plot_best_system.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RUNS = OUT / "runs"


def _load_best_system() -> dict:
    """Load the assembled ladder produced by build_best_system.py."""
    return json.loads((OUT / "best_system.json").read_text())


def _ensemble(run: str) -> dict:
    """Load the default-threshold ensemble metrics for a run."""
    return json.loads((RUNS / run / "ensemble.json").read_text())["ensemble_default"]


def main():
    """Render the two-panel RQ5 figure."""
    bs = _load_best_system()
    metrics = ["f1", "mcc", "pr_auc"]
    mlabels = ["F1", "MCC", "PR-AUC"]
    colors = {"f1": "#1f77b4", "mcc": "#ff7f0e", "pr_auc": "#2ca02c"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # (a) DiverseVul ladder as lines over steps S0..S4.
    dv = bs["diversevul"]
    steps = [r["step"] for r in dv]
    x = np.arange(len(steps))
    for m, ml in zip(metrics, mlabels):
        ys = [r[m] for r in dv]
        ax1.plot(x, ys, marker="o", linewidth=2, color=colors[m], label=ml)
        for xi, yi in zip(x, ys):
            ax1.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=7, color=colors[m])
    ax1.set_xticks(x, steps)
    ax1.set_xlabel("S0 base → S1 +structure → S2 +focal → S3 +threshold → S4 +ensemble")
    ax1.set_ylabel("score")
    ax1.set_title("(a) DiverseVul: improvement ladder (test set)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=9)

    # (b) Devign baseline vs. final (UniXcoder + 3-seed ensemble) grouped bars.
    base = bs["devign"][0]
    final = _ensemble("devign_unixcoder_fullfunc")
    xb = np.arange(len(metrics))
    width = 0.36
    b0 = [base[m] for m in metrics]
    b1 = [final[m] for m in metrics]
    bars0 = ax2.bar(xb - width / 2, b0, width, label="CodeBERT baseline", color="#9aa4b2")
    bars1 = ax2.bar(xb + width / 2, b1, width, label="final (UniXcoder + ensemble)", color="#2ca02c")
    for bars in (bars0, bars1):
        for b in bars:
            ax2.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                         textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax2.set_xticks(xb, mlabels)
    ax2.set_ylabel("score")
    ax2.set_ylim(0, 0.85)
    ax2.set_title("(b) Devign: baseline vs. final system (test set)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "best_system_ladder.png", dpi=120)
    plt.close(fig)
    print("Wrote", OUT / "best_system_ladder.png")


if __name__ == "__main__":
    main()
