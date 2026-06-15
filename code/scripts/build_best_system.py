"""Assemble the 'improvement ladder' and best-system comparison per dataset.

This addresses the project requirement of *improving test-set detection over a
baseline*. Starting from the CodeBERT baseline, it stacks one technique at a
time and reports the cumulative effect on the test split:

    S0  CodeBERT baseline (class-weighted CE)
    S1  + structure-aware backbone (UniXcoder)
    S2  + stronger imbalance handling (focal vs oversample; the variant with the
          higher *validation* F1 is selected, so test stays untouched)
    S3  + validation-tuned decision threshold (single model)
    S4  + 3-seed soft-vote ensemble (reported at the default threshold; the gain
          shows up in the threshold-independent PR-AUC / ROC-AUC)

It reads artifacts produced by train.py (summary.json), ensemble.py
(ensemble.json) and tune_threshold.py (threshold_tuned_*.json); run those first.
Outputs outputs/best_system.md and best_system.json.

Usage:
    conda activate syssec_env
    python code/scripts/build_best_system.py --threshold-metric mcc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
OUT = ROOT / "outputs"
SHOW = ["accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc"]


def _summary_mean(run: str) -> dict | None:
    """Return the test-metric means for a run, or None if absent."""
    p = RUNS / run / "summary.json"
    if not p.exists():
        return None
    agg = json.loads(p.read_text())["aggregate"]
    return {m: agg.get(m, {}).get("mean", float("nan")) for m in SHOW}


def _ensemble_mean(run: str) -> dict | None:
    """Return the ensemble (default-threshold) metrics for a run, or None."""
    p = RUNS / run / "ensemble.json"
    if not p.exists():
        return None
    e = json.loads(p.read_text())["ensemble_default"]
    return {m: e.get(m, float("nan")) for m in SHOW}


def _threshold_block(run: str, metric: str) -> dict | None:
    """Return the validation-tuned threshold file content for a run, or None."""
    p = RUNS / run / f"threshold_tuned_{metric}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _val_f1(run: str, metric: str) -> float:
    """Return mean validation F1 for a run from its threshold file (else -inf)."""
    blk = _threshold_block(run, metric)
    if not blk or "validation_default" not in blk:
        return float("-inf")
    return blk["validation_default"].get("f1", {}).get("mean", float("-inf"))


def _tuned_mean(run: str, metric: str) -> dict | None:
    """Return the test metrics at the validation-tuned threshold, or None."""
    blk = _threshold_block(run, metric)
    if not blk or "tuned" not in blk:
        return None
    return {m: blk["tuned"].get(m, {}).get("mean", float("nan")) for m in SHOW}


def select_imbalance(ds: str, metric: str) -> tuple[str, str]:
    """Pick the best UniXcoder imbalance variant by validation F1.

    Args:
        ds: Dataset name.
        metric: Threshold-tuning metric whose file carries validation metrics.

    Returns:
        Tuple (run_name, label) of the selected variant.
    """
    candidates = {
        f"{ds}_unixcoder_fullfunc": "UniXcoder + class-weighted CE",
        f"{ds}_unixcoder_focal": "UniXcoder + focal loss",
        f"{ds}_unixcoder_oversample": "UniXcoder + oversampling",
    }
    avail = {r: lab for r, lab in candidates.items() if (RUNS / r / "summary.json").exists()}
    if not avail:
        return f"{ds}_unixcoder_fullfunc", candidates[f"{ds}_unixcoder_fullfunc"]
    best = max(avail, key=lambda r: _val_f1(r, metric))
    if _val_f1(best, metric) == float("-inf"):
        # No validation info yet; default to the class-weighted variant.
        best = f"{ds}_unixcoder_fullfunc" if f"{ds}_unixcoder_fullfunc" in avail else next(iter(avail))
    return best, avail[best]


def ladder_for(ds: str, metric: str) -> list[dict]:
    """Build the ordered improvement-ladder rows for one dataset."""
    rows = []

    def add(step, label, vals):
        if vals is not None:
            rows.append({"step": step, "config": label, **vals})

    add("S0", "CodeBERT baseline", _summary_mean(f"{ds}_codebert_fullfunc"))
    add("S1", "+ structure backbone (UniXcoder)", _summary_mean(f"{ds}_unixcoder_fullfunc"))
    sel_run, sel_label = select_imbalance(ds, metric)
    add("S2", f"+ best imbalance [{sel_label}]", _summary_mean(sel_run))
    add("S3", f"+ val-tuned threshold ({metric})", _tuned_mean(sel_run, metric))
    add("S4", "+ 3-seed ensemble (final)", _ensemble_mean(sel_run))
    return rows


def render(ds: str, rows: list[dict]) -> list[str]:
    """Render one dataset's ladder as Markdown with deltas vs the baseline."""
    lines = [f"\n## {ds}: improvement ladder (test set)\n"]
    head = "| step | config | " + " | ".join(SHOW) + " |"
    lines += [head, "|" + "---|" * (len(SHOW) + 2)]
    base = rows[0] if rows else None
    for r in rows:
        cells = []
        for m in SHOW:
            v = r[m]
            if base and r is not base:
                cells.append(f"{v:.4f} ({v - base[m]:+.4f})")
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| {r['step']} | {r['config']} | " + " | ".join(cells) + " |")
    return lines


def main():
    """Assemble and write the best-system ladder for both datasets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold-metric", default="mcc", choices=["f1", "mcc"])
    args = ap.parse_args()

    all_rows = {}
    md = ["# Best-system improvement ladder\n",
          "Each step adds one technique on top of the previous; deltas are vs. "
          "the CodeBERT baseline (S0). Threshold-independent PR-AUC / ROC-AUC are "
          "the primary evidence of a genuinely stronger detector.\n"]
    for ds in ("devign", "diversevul"):
        rows = ladder_for(ds, args.threshold_metric)
        if rows:
            all_rows[ds] = rows
            md += render(ds, rows)

    (OUT / "best_system.md").write_text("\n".join(md) + "\n")
    (OUT / "best_system.json").write_text(json.dumps(all_rows, indent=2))
    print("Wrote", OUT / "best_system.md")
    for ds, rows in all_rows.items():
        print(f"\n{ds}:")
        for r in rows:
            print(f"  {r['step']} {r['config']:<42} "
                  f"F1={r['f1']:.4f} MCC={r['mcc']:.4f} PR-AUC={r['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
