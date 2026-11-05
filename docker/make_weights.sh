#!/usr/bin/env bash
# ==============================================================================
# Assemble ./weights/ (for local test.sh) and weights.zip (for forithmus
# submit --weights) from the cluster checkpoints.
#
# Runtime layout expected by predict.py under /weights:
#
#   lapa_epoch12.pt                       stage-A encoder (pillar-finetune port)
#   cls_finetune_head.pt                         384-d abn probe (MLP[1024])
#   abn_thresholds_f1max_train_lapa.json  per-class F1-max thresholds
#   ret_head.pt                                  Simple_TranSQ (input_dim 384), stripped
#   gallery_embeddings_lasthidden.pt             last-token gallery (79,166 x 4096)
#   gallery_sentences.pt                         gallery texts, same order
#   Qwen3-8B/                                    base LLM (tokenizer + shards)
#   ctrg3d/model.safetensors                     CTRG-3D newenc-B checkpoint
#   ctrg3d/vision.safetensors                    its vision_encoder + perceiver + fc
#   lora/<tag>.safetensors                       7 adapters, lora_* tensors only
#   radbert/RadBertClassifier.pth                the scorer's labeller
#   radbert/radbert_local/                       its tokenizer + base model
#
# WHY THE ADAPTERS ARE SLICED
# ---------------------------
# GRPOTrainer saved adapter_model.safetensors with save_embedding_layers=True,
# so each file is 2.4 GB: 144 lora_* tensors (15.3 MB) plus embed_tokens and
# lm_head (2.48 GB), which were FROZEN during RL and are therefore identical to
# the base checkpoint's. predict.py loads only the lora_* keys — as
# generate_batched.py does — so shipping the rest would add 17 GB of duplicated
# weights to the upload for nothing.
#
# WHY THE VISION TENSORS ARE SPLIT OUT
# ------------------------------------
# Stage A' runs CTRG-3D's own vision path before the 8B LLM is loaded, so it
# needs those tensors without the 17.8 GB they live in.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

P0=/data/drdcad/sunggu/0.Challenge/pillar-pretrain
PF=/data/drdcad/sunggu/0.Challenge/pillar-finetune
PT=/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D
EV=/data/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_evaluation

ENCODER_CKPT=${ENCODER_CKPT:-$P0/logs/lapa_pillar0_lz4_v1/checkpoints/epoch_12.pt}
PROBE_HEAD=${PROBE_HEAD:-$P0/data/abn_probe_cache/lapa_lz4_half/finetune_head.pt}
THRESHOLDS=${THRESHOLDS:-$PF/data/abn_thresholds_f1max_train_lapa.json}
RET_CKPT=${RET_CKPT:-$PF/logs/CTRATE/CTRATE_FT_Atlas_retrieval_self_lapa/checkpoints/best.ckpt}
GALLERY_EMB=${GALLERY_EMB:-$PF/data/gallery_embeddings_lasthidden.pt}
GALLERY_TXT=${GALLERY_TXT:-$PF/data/gallery_sentences.pt}
QWEN_DIR=${QWEN_DIR:-/data/drdcad/sunggu/llm_models/Qwen3-8B}
CTRG3D_CKPT=${CTRG3D_CKPT:-$PT/scripts/outputs/preserved/newencB_ep15_ckpt7455/model.safetensors}
RADBERT_CKPT=${RADBERT_CKPT:-/data/MIDS/datasets/CT-RATE/models/RadBertClassifier.pth}
RADBERT_DIR=${RADBERT_DIR:-$EV/assets/radbert_local}

# The seven distinct adapters behind the ten draws (draws 2..4 re-sample #2).
# name=path, order irrelevant here — predict.py names them.
ADAPTERS=(
  "grpo_newencB_700=$PT/outputs/grpo_newencB/checkpoint-700"
  "grpo_newencB_800=$PT/outputs/grpo_newencB/checkpoint-800"
  "grpo_newencB_900=$PT/outputs/grpo_newencB/checkpoint-900"
  "grpo_B_a2from700_1000=$PT/outputs/grpo_B_a2from700/checkpoint-1000"
  "grpo_B_a2from700_900=$PT/outputs/grpo_B_a2from700/checkpoint-900"
  "grpo_B_nobleu_700=$PT/outputs/grpo_B_nobleu/checkpoint-700"
  "grpo_B_g3_700=$PT/outputs/grpo_B_g3/checkpoint-700"
)

PYTHON=${PYTHON:-$PT/.venv/bin/python}
W=weights
mkdir -p "$W/ctrg3d" "$W/lora" "$W/radbert"

echo "[weights] stage-A encoder  <- $ENCODER_CKPT"
cp "$ENCODER_CKPT" "$W/lapa_epoch12.pt"

echo "[weights] cls probe head + thresholds"
cp "$PROBE_HEAD" "$W/cls_finetune_head.pt"
cp "$THRESHOLDS" "$W/abn_thresholds_f1max_train_lapa.json"

echo "[weights] retrieval head (stripping head_models.retrieval.* from $RET_CKPT)"
"$PYTHON" - "$RET_CKPT" "$W/ret_head.pt" <<'EOF'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
prefix = "head_models.retrieval."
head = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
assert head, f"no {prefix}* keys in {sys.argv[1]}"
# input_proj's in_channels is what pins this to the lapa encoder.
w = head.get("input_proj.weight")
assert w is not None and w.shape[1] == 384, \
    f"input_proj expects {None if w is None else w.shape[1]} channels, not 384 — " \
    f"this is not the lapa retrieval head"
torch.save(head, sys.argv[2])
print(f"  saved {len(head)} tensors -> {sys.argv[2]} (epoch {ckpt.get('epoch')})")
EOF

echo "[weights] retrieval gallery"
cp "$GALLERY_EMB" "$W/gallery_embeddings_lasthidden.pt"
cp "$GALLERY_TXT" "$W/gallery_sentences.pt"

echo "[weights] Qwen3-8B base LLM"
mkdir -p "$W/Qwen3-8B"
# .git = HF clone's LFS object store (~15 GB of duplicated weights) — exclude.
rsync -a --exclude 'LICENSE' --exclude '.git' --exclude '.gitattributes' \
    --exclude 'README.md' "$QWEN_DIR/" "$W/Qwen3-8B/"

echo "[weights] CTRG-3D checkpoint <- $CTRG3D_CKPT"
cp "$CTRG3D_CKPT" "$W/ctrg3d/model.safetensors"

echo "[weights] splitting the vision path out of the checkpoint"
"$PYTHON" - "$W/ctrg3d/model.safetensors" "$W/ctrg3d/vision.safetensors" <<'EOF'
import sys
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
keep = ("embedding_layer.vision_encoder.", "embedding_layer.perceiver.",
        "embedding_layer.fc.")
all_t = load_file(src, device="cpu")
sub = {k: v.clone() for k, v in all_t.items() if k.startswith(keep)}
for p in keep:
    assert any(k.startswith(p) for k in sub), f"no {p}* tensors in {src}"
save_file(sub, dst)
print(f"  {len(sub)} tensors -> {dst}")
EOF

echo "[weights] LoRA slices (lora_* only)"
for entry in "${ADAPTERS[@]}"; do
    tag="${entry%%=*}"; dir="${entry#*=}"
    "$PYTHON" - "$dir/adapter_model.safetensors" "$W/lora/$tag.safetensors" "$tag" <<'EOF'
import sys
from safetensors.torch import load_file, save_file
src, dst, tag = sys.argv[1], sys.argv[2], sys.argv[3]
raw = load_file(src, device="cpu")
lora = {k: v.clone() for k, v in raw.items() if "lora_" in k}
# 144 is the count every GRPO run on this base reported as its trainable set.
# A different number means a different LoRA layout, which predict.py's strict
# loader would reject at runtime — fail here instead, before the upload.
assert len(lora) == 144, f"{tag}: {len(lora)} lora_* tensors, expected 144"
save_file(lora, dst)
print(f"  {tag}: {len(lora)} tensors, {sum(v.numel()*v.element_size() for v in lora.values())/1e6:.1f} MB")
EOF
done

echo "[weights] RadBERT labeller"
cp "$RADBERT_CKPT" "$W/radbert/RadBertClassifier.pth"
mkdir -p "$W/radbert/radbert_local"
rsync -a --exclude 'README.md' --exclude 'training_args.bin' \
    "$RADBERT_DIR/" "$W/radbert/radbert_local/"

if [ "${SKIP_ZIP:-0}" = 1 ]; then
    echo "[weights] SKIP_ZIP=1 -- staged ./weights only, weights.zip NOT rebuilt"
    du -sh "$W"
    exit 0
fi

echo "[weights] zipping -> weights.zip (store-only; contents are already compressed)"
rm -f weights.zip
( cd "$W" && zip -0 -r -q ../weights.zip . -x '.gitkeep' )
du -sh "$W" weights.zip
echo "[weights] done."
