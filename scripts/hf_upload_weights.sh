#!/usr/bin/env bash
# Upload the weights staging directory to a Hugging Face model repo.
#   huggingface-cli login            (once)
#   bash scripts/hf_upload_weights.sh <hf_repo_id> [weights_dir]
# Qwen3-8B base shards are excluded: users fetch them from the Qwen repo.
set -euo pipefail
REPO=${1:?usage: hf_upload_weights.sh <hf_repo_id> [weights_dir]}
SRC=${2:-../hf_weights}    # staging dir (renamed files + the 2 adapters that shipped inside the image)
[ -d "$SRC" ] || { echo "no such dir: $SRC" >&2; exit 1; }
huggingface-cli repo create "$REPO" --type model -y 2>/dev/null || true
huggingface-cli upload "$REPO" "$SRC" . --repo-type model \
    --exclude "Qwen3-8B/*" --exclude "*.gitkeep" --exclude "radbert/radbert_local/.cache/*"
