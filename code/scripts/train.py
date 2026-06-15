"""Fine-tune a model on a processed dataset across multiple seeds.

Runs ``run_experiment`` once per seed, aggregates test metrics as mean +/- std,
writes a summary JSON, and saves a confusion-matrix figure for the first seed.
Reporting mean +/- std over seeds is required to tell real improvements from
run-to-run noise.

Runs offline; a free GPU is selected automatically on this shared server.

Usage:
    conda activate syssec_env
    python code/scripts/train.py --model microsoft/codebert-base \
        --dataset devign --seeds 42 1 2 --tag codebert_fullfunc
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

select_free_gpu(min_free_mib=20_000)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from training.experiment import run_experiment  # noqa: E402

AGG_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
               "roc_auc", "pr_auc"]


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--dataset", required=True, choices=["devign", "diversevul"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--tag", required=True, help="Short run label for the output dir")
    p.add_argument("--text-field", default="func")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--loss-type", default="ce_weighted",
                   choices=["ce_weighted", "focal", "ce"],
                   help="Training loss: inverse-freq weighted CE / focal / plain CE")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focusing parameter when --loss-type focal")
    p.add_argument("--sampler", default="none", choices=["none", "oversample"],
                   help="Training sampler; 'oversample' balances classes by duplication")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--tokenize-num-proc", type=int, default=1,
                   help="Workers for the cached tokenization step (speeds up slow tokenizers)")
    return p.parse_args()


def aggregate(per_seed: list[dict]) -> dict:
    """Aggregate per-seed metrics into mean and std.

    Args:
        per_seed: List of metric dicts, one per seed.

    Returns:
        Mapping from metric name to {"mean", "std", "values"}.
    """
    agg = {}
    for m in AGG_METRICS:
        vals = [d[m] for d in per_seed if m in d]
        agg[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(v) for v in vals],
        }
    return agg


def save_confusion_figure(metrics: dict, path: Path, title: str) -> None:
    """Save a 2x2 confusion-matrix heatmap from tn/fp/fn/tp counts.

    Args:
        metrics: Metric dict containing tn, fp, fn, tp.
        path: Output PNG path.
        title: Figure title.
    """
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    """Run the multi-seed experiment and write summary artifacts."""
    args = parse_args()
    run_root = ROOT / "outputs" / "runs" / f"{args.dataset}_{args.tag}"
    run_root.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in args.seeds:
        print(f"\n########## {args.tag} | {args.dataset} | seed {seed} ##########")
        metrics = run_experiment(
            model_name=args.model,
            dataset=args.dataset,
            seed=seed,
            output_dir=run_root / f"seed{seed}",
            text_field=args.text_field,
            max_len=args.max_len,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.lr,
            use_class_weights=not args.no_class_weights,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            sampler=args.sampler,
            trust_remote_code=args.trust_remote_code,
            tokenize_num_proc=args.tokenize_num_proc,
        )
        per_seed.append(metrics)
        print(f"seed {seed} test: " + ", ".join(
            f"{m}={metrics[m]:.4f}" for m in ["f1", "mcc", "pr_auc"]))

    agg = aggregate(per_seed)
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "tag": args.tag,
        "seeds": args.seeds,
        "config": {
            "max_len": args.max_len, "epochs": args.epochs, "lr": args.lr,
            "train_batch_size": args.train_batch_size,
            "class_weights": not args.no_class_weights, "text_field": args.text_field,
            "loss_type": args.loss_type, "focal_gamma": args.focal_gamma,
            "sampler": args.sampler,
        },
        "aggregate": agg,
        "per_seed": per_seed,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2))
    save_confusion_figure(
        per_seed[0], run_root / "confusion_seed_first.png",
        f"{args.dataset} | {args.tag} | seed {args.seeds[0]}",
    )

    print(f"\n==================== SUMMARY: {args.tag} on {args.dataset} ====================")
    for m in AGG_METRICS:
        print(f"  {m:<10} {agg[m]['mean']:.4f} +/- {agg[m]['std']:.4f}")
    print(f"\nSaved summary to {run_root / 'summary.json'}")


if __name__ == "__main__":
    main()
