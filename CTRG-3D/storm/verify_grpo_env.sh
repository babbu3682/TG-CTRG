#!/usr/bin/env bash
# storm 의 python3.10 GRPO 스택이 Biowulf 와 "같은 실험"을 돌릴 수 있는지 검증한다.
# 버전 문자열만 맞추고 넘어가면 손실 경로가 달라도 모른 채 며칠을 쓴다.
set -euo pipefail
IMAGE=${IMAGE:-reportgen-pillar-ctrg3d:latest}
ENV_HOST=${ENV_HOST:-$HOME/storm_grpoenv}
MNT=${MNT:-/mnt/biowulf}
CREPO=/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D
GPU=${GPU:-0}

docker run --rm -i --gpus all --ipc=host \
  -v "$MNT":/vf/users -v "$MNT":/data -v "$ENV_HOST":/env \
  -e CUDA_VISIBLE_DEVICES="$GPU" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/env/lib:/opt/app/rad-vision-engine -w "$CREPO" --entrypoint /env/py/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
  "$IMAGE" - <<'PYEOF'
import sys, inspect
sys.path.insert(0, "/env/lib")
import torch, transformers, peft, accelerate, trl
print(f"python {sys.version.split()[0]}")
print(f"torch {torch.__version__}  cuda_avail {torch.cuda.is_available()}  "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
print(f"transformers {transformers.__version__} | peft {peft.__version__} | "
      f"accelerate {accelerate.__version__} | trl {trl.__version__}")

exp = {"transformers":"4.57.0","peft":"0.17.1","accelerate":"1.14.0","trl":"1.10.0"}
bad = [k for k,v in exp.items() if sys.modules[k].__version__ != v]
assert not bad, f"버전 불일치 {bad} -- Biowulf 와 다른 실험이 된다"

from trl import GRPOTrainer, GRPOConfig
src = inspect.getsource(GRPOTrainer._compute_loss)
assert "old_per_token_logps is None" in src, "순수 on-policy 분기가 없다 (ratio != 1)"
print("[trl] old_per_token_logps 분기 OK -> ratio == 1 (Biowulf 와 동일)")

import inspect as I
sig = I.signature(GRPOConfig.__init__).parameters
need = ["steps_per_generation","mask_truncated_completions","scale_rewards",
        "loss_type","num_generations","max_completion_length","beta"]
miss = [n for n in need if n not in sig]
assert not miss, f"GRPOConfig 인자 없음 {miss}"
print("[trl] GRPOConfig 인자 7종 OK")

# 우리 코드가 실제로 import 되는가 -- GRPO 가 실행 중에 지연 import 하는 것까지 전부
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import grpo_crg, reward_crgf1bleu, reward_challenge_metrics, label_report_sentences
import datasets
from safetensors.torch import load_file  # noqa
from Dataset.ctrate_dataset_test import CTRATEDataset_Test  # noqa  (rve 를 끌어온다)
from Model.CTRG_3D import CTRG3D  # noqa
from reward_crgf1bleu import CRGF1BLEUReward  # noqa
from pycocoevalcap.bleu.bleu import Bleu  # noqa  (보상의 BLEU 항)
print("[repo] 지연 import 전부 OK (dataset/model/reward/BLEU)")

# A100 40GB 에서 bf16 8B 가 올라가는지 (가중치만)
free, total = torch.cuda.mem_get_info()
print(f"[gpu] free {free/2**30:.1f} / total {total/2**30:.1f} GiB")
assert total/2**30 > 38, "40GB 카드가 아니다"
print("VERIFY OK")
PYEOF
