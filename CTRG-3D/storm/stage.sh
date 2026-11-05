#!/bin/bash
# ==============================================================================
# storm 로컬 디스크(/home, NVMe)로 무거운 입력을 옮긴다.  storm 전용.
#
#   bash storm/stage.sh                       # 전부 (rlB 먼저, ab_* 나중)
#   NEWENCB_TAGS="300 400" bash storm/stage.sh
#   ONLY=lz4 bash storm/stage.sh              # 한 항목만
#
# 왜
#   /mnt/biowulf 는 helix 로의 sshfs 이고 cold read 가 **24 MB/s** 다. 컨테이너를
#   여러 개 띄우면 그 링크를 나눠 쓴다. 순차 rsync 는 36 MB/s 나오지만 병렬은 오히려
#   8 MB/s 로 떨어진다 -- 그래서 이 스크립트는 **순차**다.
#
# 무엇을
#   1) Qwen3-8B                 --lang_encoder_path 로 가리킨다
#   2) vision cache valid.*     --cache_dir 로 가리킨다
#   3) ckpt/adapter 의 *.safetensors  --ckpt / --adapter 로 가리킨다
#      (generate_batched.py 는 --ckpt 에서 model.safetensors 하나, --adapter 에서
#       adapter_model.safetensors 하나만 읽는다. basename 은 유지해야 한다 --
#       line 205 가 캐시 meta 의 checkpoint_tag 와 대조한다.)
#   4) **lz4 valid 캐시 40 GB**  <- 이게 제일 중요하다.
#      CTRATEDataset_Test.__getitem__ 이 프롬프트를 만들면서 CT 볼륨까지 읽는다
#      (샘플당 ~25 MB). generate_batched.py 는 오프라인 vision cache 를 쓰므로 그
#      텐서를 **쓰지 않고 버리는데도** 1,552개 전부를 읽는다. sshfs 에서는 프로세스당
#      38 GB 이고 3개를 병렬로 돌리면 22시간이 된다(실측: 세 프로세스가 D 상태로 정지).
#      Biowulf 에서는 스토리지가 빨라 드러나지 않던 낭비다.
#      run_generate.sh 가 이 로컬 사본을 원래 절대경로 위에 bind 해서 덮으므로
#      **manifest 도 코드도 고치지 않는다.**
#
# 권한
#   컨테이너는 이미지 기본 유저(uid 1000)로 돈다. rsync -a 는 원본의 -rw-r----- 를
#   보존하는데, 그 파일을 safetensors 는 PermissionError 가 아니라
#   **FileNotFoundError** 로 보고한다. 그래서 옮길 때마다 chmod 한다.
# ==============================================================================
set -euo pipefail
SRC=${SRC:-/mnt/biowulf/drdcad/sunggu/0.Challenge/PETRG-3D}
PP=${PP:-/mnt/biowulf/drdcad/sunggu/0.Challenge/pillar-pretrain}
LLM=${LLM:-/mnt/biowulf/drdcad/sunggu/llm_models/Qwen3-8B}
DST=${DST:-$HOME/storm_stage}
ONLY=${ONLY:-}

want(){ [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

one(){   # $1=src file  $2=dst file
  if [ -s "$2" ] && [ "$(stat -c%s "$1")" = "$(stat -c%s "$2")" ]; then
      echo "[stage] skip $(basename "$(dirname "$2")")/$(basename "$2")"; return
  fi
  mkdir -p "$(dirname "$2")"; t0=$SECONDS
  rsync -a --append-verify "$1" "$2"
  chmod a+r "$2"
  echo "[stage] $(basename "$(dirname "$2")")/$(basename "$2")  $(du -h "$2" | cut -f1)  t=$((SECONDS-t0))s"
}

tree(){  # $1=src dir  $2=dst dir
  mkdir -p "$2"; t0=$SECONDS
  rsync -a --append-verify --info=progress2 "$1/" "$2/"
  chmod -R a+rX "$2"
  echo "[stage] $(basename "$2")  $(du -sh "$2" | cut -f1)  t=$((SECONDS-t0))s"
}

adapters(){   # $1=run  $2..=tags
  run=$1; shift
  for t in "$@"; do
    s="$SRC/outputs/$run/checkpoint-$t/adapter_model.safetensors"
    if [ ! -s "$s" ]; then echo "[stage] (아직 없음) $run/checkpoint-$t"; continue; fi
    one "$s" "$DST/adapters/$run/checkpoint-$t/adapter_model.safetensors"
  done
}

# ---- 1) lz4 valid 캐시 40 GB -- 가장 큰 이득 ---------------------------------
if want lz4; then tree "$PP/data/ctrate/ctrate_cache_1.25mm_lz4/valid" "$DST/lz4_valid"; fi

# ---- 2) LLM ------------------------------------------------------------------
if want llm; then tree "$LLM" "$DST/Qwen3-8B"; fi

# ---- 3) vision cache (valid 쪽만 -- 채점 생성은 valid 만 쓴다) ---------------
if want cache; then
  for pair in "newencB_ep15_ckpt7455:cache_newencB" "checkpoint-16412:cache_ckpt16412"; do
    s=${pair%%:*}; d=${pair#*:}
    for f in valid.npy valid_index.json valid_meta.json; do
      one "$SRC/data/vision_cache/$s/$f" "$DST/$d/$f"
    done
  done
fi

# ---- 4) rlB 가 필요로 하는 체크포인트/어댑터 ---------------------------------
if want rlB; then
  one "$SRC/scripts/outputs/preserved/newencB_ep15_ckpt7455/model.safetensors" \
      "$DST/ckpt/newencB_ep15_ckpt7455/model.safetensors"
  adapters grpo_newencB ${NEWENCB_TAGS:-100 200 300 400 500 600 700}
  # storm 이 학습한 변형들 -- ADAPT_RUNS 로 고른다 (기본은 없음, 기존 동작 불변)
  for _r in ${ADAPT_RUNS:-}; do adapters "$_r" ${ADAPT_TAGS:-700}; done
  chmod -R a+rX "$DST"
  echo "[stage] rlB READY"
fi

# ---- 4.5) GRPO 학습(RL)용 -- train vision cache 25 GB + 참조 산출물 ----------
#   RL 은 rollout 마다 train.npy 의 임의 행을 읽는다(memmap 랜덤 액세스). sshfs 에서
#   하면 학습이 I/O 에 묶이므로 반드시 로컬에 둔다. 프롬프트 쪽 CT 볼륨(~300 GB)은
#   --prompts_only 로 아예 읽지 않으므로 lz4 train 캐시는 필요 없다.
if want grpo; then
  for f in train.npy train_index.json train_meta.json; do
    one "$SRC/data/vision_cache/newencB_ep15_ckpt7455/$f" "$DST/cache_newencB/$f"
  done
  for f in subset12000.json reference.json; do
    one "$SRC/data/grpo_ref_newencB/$f" "$DST/grpo_ref_newencB/$f"
  done
  chmod -R a+rX "$DST/cache_newencB" "$DST/grpo_ref_newencB"
  echo "[stage] grpo READY"
fi

# ---- 5) ab_* (ep15best, 17 GB -- 마감이 걸린 쪽 뒤로 미룬다) -----------------
if want ab; then
  one "$SRC/scripts/outputs/ep15/ctrg3d-qwen3-8B-pillar0os-lz4-vt128/checkpoint-16412/model.safetensors" \
      "$DST/ckpt/checkpoint-16412/model.safetensors"
  adapters grpo_ab_gamma2 ${AB_TAGS:-800 1000}
  adapters grpo_ab_g16    ${AB_TAGS:-800 1000}
fi

chmod -R a+rX "$DST"
echo "[stage] ALL DONE  $(du -sh "$DST" | cut -f1)"
