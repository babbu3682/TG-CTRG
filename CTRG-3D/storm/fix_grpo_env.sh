#!/usr/bin/env bash
# --target 설치는 upgrade/downgrade 가 신뢰할 수 없다. timm 이 끌어온 torchvision 0.28.0
# (torch 2.13 용)이 남아 torch 2.8.0 과 ABI 가 어긋났고, 증상은 import 시
# `operator torchvision::nms does not exist` 였다. 지우고 정확한 짝을 다시 깐다.
set -euo pipefail
IMAGE=${IMAGE:-reportgen-pillar-ctrg3d:latest}
ENV_HOST=${ENV_HOST:-$HOME/storm_grpoenv}
chmod -R u+w "$ENV_HOST/lib" 2>/dev/null || true
docker run --rm -v "$ENV_HOST":/env --entrypoint bash "$IMAGE" -lc '
set -e
PY=/env/py/cpython-3.10-linux-x86_64-gnu/bin/python3.10
export HOME=/tmp
rm -rf /env/lib/torchvision /env/lib/torchvision.libs /env/lib/torchvision-*.dist-info
echo "--- torchvision 0.23.0+cu126 (torch 2.8.0 의 짝) ---"
$PY -m pip -q install --target=/env/lib --no-deps \
    torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -3
echo "--- pycocoevalcap (보상의 BLEU 항) -- 의존성 없이. numpy 2 를 끌어오면 안 된다 ---"
$PY -m pip -q install --target=/env/lib --no-deps pycocoevalcap pycocotools 2>&1 | tail -3
echo "--- numpy 를 하나의 버전으로 정리 ---"
#   pycocoevalcap 이 numpy 2.2.6 을 끌어와 1.26.4 위에 반쯤 덮었다. 그러면
#   dist-info 는 2.x 인데 파일은 1.x 라, accelerate 가 numpy 버전을 보고 고르는
#   `numpy._core.multiarray` 경로에서 AttributeError 가 난다. 증상은 엉뚱하게도
#   "circular import" 로 보인다. **Biowulf 가 numpy 2.2.6 / pandas 2.3.3** 이므로 거기에 맞춘다
#   (빌드 스크립트의 "numpy<2" 핀은 틀렸다 -- 실측으로 확인).
rm -rf /env/lib/numpy /env/lib/numpy.libs /env/lib/numpy-*.dist-info /env/lib/pandas /env/lib/pandas-*.dist-info
$PY -m pip -q install --target=/env/lib --no-deps "numpy==2.2.6" "pandas==2.3.3" 2>&1 | tail -2
echo "--- repo 가 요구하는 나머지 (Biowulf 와 같은 버전) ---"
$PY -m pip -q install --target=/env/lib --no-deps \
    "nibabel==5.4.2" "monai==1.6.0" "easydict==1.13" "PyYAML==6.0.3" "scipy==1.15.3" \
    einops_exts "timm==1.0.28" "lz4==4.4.5" "SimpleITK==2.5.5" \
    "scikit-image==0.25.2" "huggingface_hub==0.36.2" 2>&1 | tail -3
echo "--- wandb (TRL 의 WandbCallback 이 요구한다. WANDB_MODE=offline 로 돈다) ---"
$PY -m pip -q install --target=/env/lib "wandb==0.28.1" 2>&1 | tail -2
echo "--- datasets / pyarrow / fsspec 를 Biowulf 와 정확히 맞춘다 ---"
#   storm 은 datasets 5.0.1 / pyarrow 25.0.1 이 깔려 있었고 fsspec dist-info 가 둘이었다.
#   datasets 4 -> 5 는 major 차이라 RL 프롬프트 데이터셋 구성이 달라질 수 있고, 실제로
#   PicklingError (MonthDayNano 를 pickle 할 수 없다) 로 실제로 터졌다.
rm -rf /env/lib/datasets /env/lib/datasets-*.dist-info \
       /env/lib/pyarrow /env/lib/pyarrow-*.dist-info \
       /env/lib/fsspec /env/lib/fsspec-*.dist-info
$PY -m pip -q install --target=/env/lib --no-deps \
    "datasets==4.8.4" "pyarrow==25.0.0" "fsspec==2026.2.0" 2>&1 | tail -3
$PY -c "import sys; sys.path.insert(0,\"/env/lib\")
import numpy, pandas, datasets, pyarrow, fsspec
print(\"numpy\", numpy.__version__, \"| pandas\", pandas.__version__,
      \"| datasets\", datasets.__version__, \"| pyarrow\", pyarrow.__version__,
      \"| fsspec\", fsspec.__version__)"
'
echo "[fix_grpo_env] done"
