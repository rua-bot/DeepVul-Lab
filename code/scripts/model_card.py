"""Tabulate the structural and pretraining characteristics of each backbone.

This script measures the *controllable*factors directly 
from the local model files (parameter count, layers, hidden
size, vocabulary, max positions) and pairs them with the *pretraining* factors
(corpus, scale, objectives, tokenizer, whether program structure is modeled)
taken from each model's original paper.

The point is to make the confounds explicit: parameter count and architecture
are essentially held constant (all are RoBERTa-base ~125M), which rules out the
"bigger model" explanation, whereas pretraining corpus / scale / objective are
*not* controlled and are reported so that causal claims can be appropriately
qualified.

Outputs outputs/model_characteristics.json and .md.

Usage:
    conda activate syssec_env
    python code/scripts/model_card.py
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from pathlib import Path

from transformers import AutoConfig, AutoModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Model ids in the comparison, in reporting order.
MODELS = [
    "microsoft/codebert-base",
    "claudios/VulBERTa-mlm",
    "microsoft/graphcodebert-base",
    "microsoft/unixcoder-base",
]

# Pretraining facts from the original papers. Scale figures are approximate and
# are meant for qualitative comparison, not exact reproduction.
PRETRAIN_INFO = {
    "microsoft/codebert-base": {
        "short": "CodeBERT",
        "pretrain_corpus": "CodeSearchNet (6 PLs: go/java/js/php/python/ruby)",
        "pretrain_scale": "~2.1M NL-PL pairs + ~6.4M unimodal functions",
        "objectives": "MLM + Replaced Token Detection",
        "tokenizer": "RoBERTa byte-level BPE (shared NL+PL)",
        "domain_adapted_cpp": "No",
        "structure_aware": "No (token sequence only)",
    },
    "claudios/VulBERTa-mlm": {
        "short": "VulBERTa",
        "pretrain_corpus": "C/C++ only (Draper VDISC + real-world GitHub projects)",
        "pretrain_scale": "~1.1M+ C/C++ functions (far smaller, single-domain)",
        "objectives": "MLM only",
        "tokenizer": "Custom libclang + BPE (C/C++ specific)",
        "domain_adapted_cpp": "Yes",
        "structure_aware": "No (token sequence only)",
    },
    "microsoft/graphcodebert-base": {
        "short": "GraphCodeBERT",
        "pretrain_corpus": "CodeSearchNet (same corpus & tokenizer as CodeBERT)",
        "pretrain_scale": "~2.3M bimodal functions (comparable to CodeBERT)",
        "objectives": "MLM + Edge Prediction + Node Alignment (data flow)",
        "tokenizer": "Same as CodeBERT (RoBERTa BPE, identical vocab)",
        "domain_adapted_cpp": "No",
        "structure_aware": "Yes (data-flow graph)",
    },
    "microsoft/unixcoder-base": {
        "short": "UniXcoder",
        "pretrain_corpus": "CodeSearchNet + C4 (NL) + flattened AST (multimodal)",
        "pretrain_scale": "Larger / multimodal (code + NL + AST)",
        "objectives": "MLM + Uni-LM + Denoising + multimodal contrastive",
        "tokenizer": "Own BPE (vocab 51416)",
        "domain_adapted_cpp": "No",
        "structure_aware": "Yes (flattened AST)",
    },
}


def measure(model_id: str) -> dict:
    """Measure structural facts of one backbone from its local files.

    Args:
        model_id: HuggingFace model id present in the local cache.

    Returns:
        Dict with parameter counts and architecture hyperparameters, merged with
        the curated pretraining info for the model.
    """
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings()
    emb_params = sum(p.numel() for p in emb.parameters()) if emb is not None else 0
    info = {
        "model": model_id,
        "params_total_M": round(total / 1e6, 1),
        "params_non_embedding_M": round((total - emb_params) / 1e6, 1),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_layers": getattr(cfg, "num_hidden_layers", None),
        "num_heads": getattr(cfg, "num_attention_heads", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
    }
    info.update(PRETRAIN_INFO.get(model_id, {}))
    del model
    return info


def write_markdown(rows: list[dict], path: Path) -> None:
    """Write the model-characteristics tables as Markdown.

    Args:
        rows: Per-model characteristic dicts.
        path: Output Markdown path.
    """
    lines = ["# Backbone characteristics\n",
             "\n## Controlled factors (measured from model files)\n"]
    cols = ["short", "params_total_M", "params_non_embedding_M", "hidden_size",
            "num_layers", "num_heads", "vocab_size", "max_position_embeddings"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")

    lines.append("\n## Uncontrolled factors (from original papers)\n")
    cols2 = ["short", "pretrain_corpus", "pretrain_scale", "objectives",
             "tokenizer", "domain_adapted_cpp", "structure_aware"]
    lines.append("| " + " | ".join(cols2) + " |")
    lines.append("|" + "---|" * len(cols2))
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols2) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    """Measure all backbones and write the characteristics tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [measure(m) for m in MODELS]
    (OUT / "model_characteristics.json").write_text(json.dumps(rows, indent=2))
    write_markdown(rows, OUT / "model_characteristics.md")
    print("Backbone parameter counts (M):")
    for r in rows:
        print(f"  {r['short']:<14} total={r['params_total_M']:>6}  "
              f"non-emb={r['params_non_embedding_M']:>6}  "
              f"struct={r['structure_aware']}")
    print(f"\nWrote {OUT / 'model_characteristics.md'}")


if __name__ == "__main__":
    main()
