# Backbone characteristics


## Controlled factors (measured from model files)

| short | params_total_M | params_non_embedding_M | hidden_size | num_layers | num_heads | vocab_size | max_position_embeddings |
|---|---|---|---|---|---|---|---|
| CodeBERT | 124.6 | 86.0 | 768 | 12 | 12 | 50265 | 514 |
| VulBERTa | 124.8 | 86.4 | 768 | 12 | 12 | 50000 | 1026 |
| GraphCodeBERT | 124.6 | 86.0 | 768 | 12 | 12 | 50265 | 514 |
| UniXcoder | 125.9 | 86.4 | 768 | 12 | 12 | 51416 | 1026 |

## Uncontrolled factors (from original papers)

| short | pretrain_corpus | pretrain_scale | objectives | tokenizer | domain_adapted_cpp | structure_aware |
|---|---|---|---|---|---|---|
| CodeBERT | CodeSearchNet (6 PLs: go/java/js/php/python/ruby) | ~2.1M NL-PL pairs + ~6.4M unimodal functions | MLM + Replaced Token Detection | RoBERTa byte-level BPE (shared NL+PL) | No | No (token sequence only) |
| VulBERTa | C/C++ only (Draper VDISC + real-world GitHub projects) | ~1.1M+ C/C++ functions (far smaller, single-domain) | MLM only | Custom libclang + BPE (C/C++ specific) | Yes | No (token sequence only) |
| GraphCodeBERT | CodeSearchNet (same corpus & tokenizer as CodeBERT) | ~2.3M bimodal functions (comparable to CodeBERT) | MLM + Edge Prediction + Node Alignment (data flow) | Same as CodeBERT (RoBERTa BPE, identical vocab) | No | Yes (data-flow graph) |
| UniXcoder | CodeSearchNet + C4 (NL) + flattened AST (multimodal) | Larger / multimodal (code + NL + AST) | MLM + Uni-LM + Denoising + multimodal contrastive | Own BPE (vocab 51416) | No | Yes (flattened AST) |
