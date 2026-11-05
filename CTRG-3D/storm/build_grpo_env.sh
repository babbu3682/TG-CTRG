#!/usr/bin/env bash
# storm 에서 GRPO 를 돌리기 위한 python 3.10 스택을 만든다.
#
# 왜 필요한가: 이미지의 python 은 3.9.5 인데 **trl >= 0.25 는 python >= 3.10 을 요구**한다.
#   설치 가능한 최대는 trl 0.24.0 이고, Biowulf 는 trl 1.10.0 이다. 10 마이너 버전 차이는
#   GRPO 손실 경로(old_per_token_logps 분기, loss_type, scale_rewards 의미)를 바꿀 수 있고,
#   그러면 storm 결과와 Biowulf 결과가 "같은 실험"이 아니게 된다. 이 저장소에서 그런 조용한
#   불일치는 며칠을 버리게 만든 적이 있으므로, 버전을 맞추는 쪽을 택한다.
#
# 방법: uv 로 standalone python 3.10 을 받아 그 위에 Biowulf 와 **같은 버전**을 깐다.
#   torch 만 드라이버 535 에 맞춰 cu126 (Biowulf 는 cu130 / 드라이버 580+).
set -euo pipefail
IMAGE=${IMAGE:-reportgen-pillar-ctrg3d:latest}
ENV_HOST=${ENV_HOST:-$HOME/storm_grpoenv}
mkdir -p "$ENV_HOST"; chmod 777 "$ENV_HOST"

docker run --rm -v "$ENV_HOST":/env --entrypoint bash "$IMAGE" -lc '
set -e
export HOME=/tmp XDG_CACHE_HOME=/tmp/.cache UV_CACHE_DIR=/env/.uvcache
if [ ! -x /env/uv ]; then
  pip -q install --target=/tmp/uvpkg uv 2>&1 | tail -2
  cp /tmp/uvpkg/bin/uv /env/uv 2>/dev/null || PYTHONPATH=/tmp/uvpkg python -m uv --version
  export PYTHONPATH=/tmp/uvpkg
fi
UV="python -m uv"; export PYTHONPATH=/tmp/uvpkg
$UV python install 3.10 --install-dir /env/py 2>&1 | tail -2
PY310=$(ls -d /env/py/*/bin/python3.10 2>/dev/null | head -1)
echo "python3.10 -> $PY310"; $PY310 -V
$PY310 -m ensurepip --upgrade >/dev/null 2>&1 || true
$PY310 -m pip -q install --upgrade pip 2>&1 | tail -1
echo "--- torch 2.8.0 cu126 (드라이버 535 대응) ---"
$PY310 -m pip -q install --target=/env/lib torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -3
echo "--- torchvision 은 torch 2.8.0 에 맞춰 0.23.0+cu126 (이미지와 동일) ---"
$PY310 -m pip -q install --target=/env/lib --upgrade \
    torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2
echo "--- Biowulf 와 동일 버전 ---"
$PY310 -m pip -q install --target=/env/lib \
    "transformers==4.57.0" "peft==0.17.1" "accelerate==1.14.0" "trl==1.10.0" \
    "numpy==2.2.6" "pandas==2.3.3" einops safetensors sentencepiece timm lz4 \
    datasets pycocoevalcap scipy scikit-learn 2>&1 | tail -4
'
echo "[build_grpo_env] done -> $ENV_HOST"
