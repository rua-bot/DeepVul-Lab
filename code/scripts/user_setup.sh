#!/usr/bin/env bash
# Environment setup: install dependencies and download datasets/models.
# Run this manually (it performs installs and downloads):
#
#   conda activate syssec_env
#   bash code/scripts/user_setup.sh
#
# HF mirror is already configured (HF_ENDPOINT=https://hf-mirror.com); set again
# here just to be safe.

set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "Using HF_ENDPOINT=$HF_ENDPOINT"

echo "=========================================================="
echo "[1/3] Install libclang (required by VulBERTa custom tokenizer)"
echo "      datasketch is for near-duplicate dedup later (install now to save a round-trip)"
echo "=========================================================="
pip install libclang datasketch

echo "=========================================================="
echo "[2/3] Download DiverseVul dataset into the HF cache"
echo "=========================================================="
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("claudios/DiverseVul")
print("DiverseVul splits:", {k: len(v) for k, v in ds.items()})
print("columns:", ds[list(ds.keys())[0]].column_names)
PY

echo "=========================================================="
echo "[3/3] Download VulBERTa core pretrained backbone (claudios/VulBERTa-mlm)"
echo "      (weights + custom tokenizer code; we do NOT execute it here)"
echo "=========================================================="
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="claudios/VulBERTa-mlm")
print("VulBERTa-mlm downloaded to:", path)
PY

echo ""
echo "DONE. Next: run the VulBERTa compatibility test (decides main vs fallback):"
echo "    python code/scripts/vulberta_load_test.py"
