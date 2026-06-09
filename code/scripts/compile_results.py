"""Aggregate all experiment runs into a single comparison table and figures.

Scans outputs/runs/*/summary.json, collects the per-run mean +/- std metrics, and
writes a Markdown table, a CSV, and one grouped bar chart per dataset. Re-run any
time after new experiments finish to refresh the comparison.

Usage:
    conda activate syssec_env
    python code/scripts/compile_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
OUT = ROOT / "outputs"
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]
BAR_METRICS = ["f1", "mcc", "pr_auc", "roc_auc"]


def collect() -> list[dict]:
    """Read every run summary into a flat list of row dicts.

    Returns:
        List of rows with dataset, tag, model, seeds and per-metric mean/std.
    """
    rows = []
    for summ in sorted(RUNS.glob("*/summary.json")):
        d = json.loads(summ.read_text())
        row = {
            "dataset": d["dataset"],
            "tag": d["tag"],
            "model": d["model"],
            "seeds": d["seeds"],
        }
        for m in METRICS:
            agg = d["aggregate"].get(m, {})
            row[f"{m}_mean"] = agg.get("mean", float("nan"))
            row[f"{m}_std"] = agg.get("std", float("nan"))
        rows.append(row)
    return rows


def write_markdown(rows: list[dict], path: Path) -> None:
    """Write the comparison rows as a grouped Markdown table.

    Args:
        rows: Collected run rows.
        path: Output Markdown path.
    """
    lines = ["# Experiment comparison\n"]
    for dataset in sorted({r["dataset"] for r in rows}):
        lines.append(f"\n## {dataset}\n")
        header = "| run | " + " | ".join(METRICS) + " |"
        sep = "|" + "---|" * (len(METRICS) + 1)
        lines += [header, sep]
        for r in sorted([x for x in rows if x["dataset"] == dataset], key=lambda x: x["tag"]):
            cells = [f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}" for m in METRICS]
            lines.append(f"| {r['tag']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_csv(rows: list[dict], path: Path) -> None:
    """Write all rows to a flat CSV.

    Args:
        rows: Collected run rows.
        path: Output CSV path.
    """
    if not rows:
        return
    fields = ["dataset", "tag", "model"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_per_dataset(rows: list[dict]) -> None:
    """Save a grouped bar chart (metrics x runs) for each dataset.

    Args:
        rows: Collected run rows.
    """
    for dataset in sorted({r["dataset"] for r in rows}):
        drows = sorted([x for x in rows if x["dataset"] == dataset], key=lambda x: x["tag"])
        if not drows:
            continue
        x = np.arange(len(BAR_METRICS))
        width = 0.8 / max(len(drows), 1)
        fig, ax = plt.subplots(figsize=(1.6 * len(BAR_METRICS) + 2, 4.2))
        for i, r in enumerate(drows):
            means = [r[f"{m}_mean"] for m in BAR_METRICS]
            errs = [r[f"{m}_std"] for m in BAR_METRICS]
            ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=r["tag"])
        ax.set_xticks(x + width * (len(drows) - 1) / 2, BAR_METRICS)
        ax.set_ylabel("score")
        ax.set_title(f"{dataset}: model/input comparison (mean ± std over seeds)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"compare_{dataset}.png", dpi=120)
        plt.close(fig)


def main():
    """Collect runs and emit table, CSV and figures."""
    rows = collect()
    if not rows:
        print("No runs found under", RUNS)
        return
    write_markdown(rows, OUT / "results_table.md")
    write_csv(rows, OUT / "results_table.csv")
    plot_per_dataset(rows)
    print(f"Collected {len(rows)} runs.")
    print("Wrote:", OUT / "results_table.md", OUT / "results_table.csv")
    for dataset in sorted({r["dataset"] for r in rows}):
        print(f"  figure: {OUT / f'compare_{dataset}.png'}")


if __name__ == "__main__":
    main()
