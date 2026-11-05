#!/bin/bash
# ==============================================================================
# storm(로컬 서버, <storm-host>) 이 CTRG 채점을 돌릴 수 있는지 확인한다.
# Biowulf 가 아니므로 singularity 가 없고 docker 를 쓴다.
#
#   bash /mnt/biowulf/drdcad/sunggu/0.Challenge/PETRG-3D/storm/probe.sh
#
# 아무것도 바꾸지 않는다 -- 읽기만 한다.
# ==============================================================================
MNT=${MNT:-/mnt/biowulf}
REPO=$MNT/drdcad/sunggu/0.Challenge/PETRG-3D
echo "=============================================================="
echo " storm probe   $(date '+%F %T')   host=$(hostname)  user=$(whoami)"
echo "=============================================================="

echo "### 1. GPU / 드라이버  (torch 2.13+cu130 은 드라이버 >= 580 을 요구한다)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 | head -10
echo

echo "### 2. docker"
docker --version 2>&1 | head -2
echo -n "  docker 권한: "; docker ps >/dev/null 2>&1 && echo "OK (sudo 불필요)" || echo "NO (sudo 필요하거나 daemon 미기동)"
echo -n "  --gpus 지원: "; docker run --rm --gpus all nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 \
    nvidia-smi -L 2>&1 | head -4
echo

echo "### 3. Biowulf 마운트가 같은 파일을 보는가"
for p in "$REPO/scripts/generate_batched.py" \
         "$REPO/outputs/grpo_newencB/checkpoint-200/adapter_model.safetensors" \
         "$REPO/data/vision_cache/newencB_ep15_ckpt7455/valid.npy" \
         "$REPO/data/vision_cache/checkpoint-16412/valid.npy" \
         "$REPO/outputs/grpo_ab_gamma2/checkpoint-800/adapter_model.safetensors" \
         "$MNT/drdcad/sunggu/llm_models/Qwen3-8B/config.json" \
         "$MNT/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_valid_filtered_prob_half.json" \
         "$MNT/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_evaluation/assets/radbert_local/config.json" \
         "$MNT/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_evaluation/assets/RadBertClassifier.pth"; do
    if [ -e "$p" ]; then printf "  OK   %10s  %s\n" "$(du -Lh "$p" 2>/dev/null | cut -f1)" "$p"
    else                 printf "  MISS %10s  %s\n" "-" "$p"; fi
done
echo -n "  쓰기 가능(results/): "; touch "$REPO/results/.storm_write_test" 2>/dev/null \
    && { echo OK; rm -f "$REPO/results/.storm_write_test"; } || echo NO
echo

echo "### 4. venv (Biowulf 에서 만든 것을 그대로 쓸 수 있는가)"
# repo 의 .venv 는 storm 에서 못 쓴다(torch 2.13+cu130 이 드라이버 >=580 요구, storm 은 535).
# 확인용으로 남겨 두되, 실제로 쓰는 것은 reportgen-pillar-ctrg3d 이미지다 (storm/README.md §2).
echo -n "  .venv/bin/python -> "; readlink -f "$REPO/.venv/bin/python" 2>&1
echo -n "  image torch      -> "
docker run --rm --entrypoint python reportgen-pillar-ctrg3d:latest \
    -c "import torch;print(torch.__version__, 'cuda_avail', torch.cuda.is_available())" 2>&1 | tail -1
echo "  (컨테이너 안에서 /data/drdcad 와 /vf/users/drdcad 둘 다 같은 곳을 가리켜야 shebang 이 맞는다)"
echo

echo "### 5. 디스크 / RAM"
df -h "$MNT" 2>/dev/null | tail -1
free -g 2>/dev/null | awk 'NR<=2'
nproc
echo "=============================================================="
