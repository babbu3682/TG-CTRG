#!/bin/bash

# lapa -- same local-only PhraseAlign objective as `phrase_align`, with the fixes taken from
# the reference stage-1 implementation (ct-rate-rrg/stage1_phrase_align_ct). See CLAUDE.md.
#
# Differences from train_unimodal_pillar0_ctrate_chest_ct_1node_phrase_align.sh:
#   init          own atlas ep300 --> released Pillar0-ChestCT (direct safetensors load)
#   lr            flat 2.5e-4     --> 3e-5 deepest vision stage, layer_decay 0.8
#                                     (patch_embeds 1.54e-5), heads 1e-4, text proj 1e-4
#   tokens        32768 (3 scales upsampled+concat) --> 1024 (scale index 1)
#   patch embed   isotropic 8x8x8 --> (H,W,D)=(8,8,4) with the axis order FIXED
#   batch         2/GPU, window 32 vols --> 4/GPU, window 64 vols
#   phrases       all --> random 80% per epoch
#   duplicates    unhandled --> verbatim-identical phrases removed from both negative sets
#                               and promoted to positives
#   VL-CABS tau   frozen log(0.07) buffer --> trainable nn.Parameter

export OMP_NUM_THREADS=2
export NCCL_TIMEOUT=300

uv run python -m torch.distributed.run \
    --master_port 6177 \
    --master_addr 127.0.0.1  \
    --nproc_per_node gpu \
    -m trainer.main \
    --dataset-type multimodal \
    --multimodal-config src/miniclip/data_configs/pillar0_ctrate_chest_ct_256_lapa.yaml \
    --accum-freq 8 \
    --val-frequency 1 \
    --save-frequency 1 \
    --save-frequency-step 200 \
    --save-most-recent \
    --warmup 300 \
    --epochs 50 \
    --log-every-n-steps 2 \
    --lr 3.0e-5 \
    --layer-decay 0.8 \
    --head-lr 1.0e-4 \
    --text-lr-multiplier 3.3333 \
    --wd 0.0 \
    --grad-clip-norm 1.0 \
    --precision amp_bf16 \
    --image-mean "0.289" \
    --image-std "0.198" \
    --aug-cfg med_3d_aug=True \
    --image-interpolation bilinear \
    --model "Qwen3-Embedding-8B-Atlas_small_d2d2d8_multiscale__pillar0_multimodal_ucsf_chest_ct_multiwindow_lapa" \
    --name "lapa_pillar0_lz4_v1" \
    --report-to "wandb" \
    --wandb-project-name "pilar-pretrain" \
    --wandb-entity "babbu3682-university-of-ulsan" \
    --resume latest \
    --window-type "all" \
    --workers 2 \
    --dataloader-timeout 300 \
    --multimodal-rebuild-on-exhaust \
    --max-val-batches 32 \
    --enable-gpu-rotation \
    --gpu-rotation-degrees 20 \
    --gpu-rotation-p 0.5 \
    --mask-duplicate-phrases \
    --pretrained-safetensors "/data/drdcad/sunggu/0.Challenge/pillar-pretrain/checkpoints/Pillar0-ChestCT/model.safetensors" \
    --lapa
