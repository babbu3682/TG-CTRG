#!/usr/bin/env bash
# Download the released checkpoints from Hugging Face into the layout docker/predict.py expects.
#   bash scripts/hf_download_weights.sh <hf_repo_id> [dest_dir]
set -euo pipefail
REPO=${1:?usage: hf_download_weights.sh <hf_repo_id> [dest_dir]}
DEST=${2:-docker/weights}
mkdir -p "$DEST"
HF=${HF_CLI:-hf}
command -v "$HF" >/dev/null || { echo "hf CLI not found: pip install -U huggingface_hub" >&2; exit 1; }
"$HF" download "$REPO" --repo-type model --local-dir "$DEST"
echo "weights in $DEST"; ls "$DEST"
