#!/usr/bin/env bash
# Upload the weights staging directory to a Hugging Face model repo.
#   hf auth login                    (once; needs a token with write access)
#   bash scripts/hf_upload_weights.sh <hf_repo_id> [weights_dir]
# Qwen3-8B base shards are excluded: users fetch them from the Qwen repo.
# HF_CLI=/path/to/hf overrides the CLI binary (huggingface_hub >= 1.0 ships it as `hf`).
set -euo pipefail
REPO=${1:?usage: hf_upload_weights.sh <hf_repo_id> [weights_dir]}
SRC=${2:-../hf_weights}    # staging dir (renamed files + the 2 adapters that shipped inside the image)
HF=${HF_CLI:-hf}
[ -d "$SRC" ] || { echo "no such dir: $SRC" >&2; exit 1; }
command -v "$HF" >/dev/null || { echo "hf CLI not found: pip install -U huggingface_hub" >&2; exit 1; }
"$HF" repo create "$REPO" --repo-type model --private 2>/dev/null || true   # no-op if it already exists
"$HF" upload "$REPO" "$SRC" . --repo-type model \
    --exclude "Qwen3-8B/*" --exclude "*.gitkeep" --exclude "radbert/radbert_local/.cache/*"
