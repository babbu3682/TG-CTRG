#!/usr/bin/env bash
# storm 에서 GRPO 변형 하나를 GPU 한 장에 돌린다.
#
#   bash ~/storm_bin/run_grpo.sh <variant> [GPU]
#   변형: nobleu | a2 | g3 | a2g3        (아래 표)
#
# 설계 근거 (docs/2026-08-26-session-handoff.md 참조)
#   * lambda_bleu = 0 을 네 변형 모두 고정한다. 사용자가 "BLEU 는 충분"이라고 했고,
#     실측상 BLEU 항이 보상의 1/3 을 먹으면서 정책을 짧고 안전한 쪽으로 밀고 있다.
#     그 예산을 임상 지표로 돌린다.
#   * **비율만 의미가 있다.** GRPO 는 그룹 내 std 로 advantage 를 정규화하므로 보상에
#     상수를 곱해도 같은 실험이다. 따라서 alpha:gamma:lambda 의 *비* 로 변형을 나눈다
#     (alpha=3,gamma=1.5 는 alpha=2,gamma=1 과 같은 실험이라 넣을 값어치가 없다).
#   * 반유병률-게이밍 조건 gamma/alpha > 0.401 을 네 칸 모두 만족한다(0.5 / 1.0 / 3.0).
#     이 조건을 깨는 alpha:gamma = 4:1 은 넣지 않았다 -- 흔한 라벨 남발이 순이득이 된다.
#   * 네 번째 축은 rollout 온도다. 기존 채점 산출물 3개를 라벨 투표로 합치면
#     recall 0.586 -> **0.758**, CRG 0.5052 -> **0.5403** 이 나온다. 즉 맞는 소견이
#     이미 표본 분포 안에 있는데 draw 마다 흩어져 있다. 온도를 올리면 그 분포를 더
#     넓게 보여 주므로, CRG 항이 고recall 표본을 집어 올릴 여지가 커진다.
#
# Biowulf 대비 confound 하나를 명시한다: 40GB 카드라 generation batch 가 작아
# 스텝당 프롬프트 수가 다를 수 있다. 네 변형끼리는 동일하므로 서로 비교는 유효하고,
# Biowulf 런과는 절대값이 아니라 추세로 비교한다.
set -euo pipefail

VARIANT=${1:?variant: nobleu | a2 | g3 | a2t12}
GPU=${2:-0}
TEMP=1.0
case "$VARIANT" in
  nobleu) ALPHA=1.0; GAMMA=1.0 ;;                 # 대조: Biowulf 대비 lambda 만 다름
  a2)     ALPHA=2.0; GAMMA=1.0 ;;                 # CRG 쪽. gamma/alpha=0.5 (게이밍 하한 0.401)
  g3)     ALPHA=1.0; GAMMA=3.0 ;;                 # macro-F1 쪽 (희귀 라벨)
  a2t12)  ALPHA=2.0; GAMMA=1.0; TEMP=1.2 ;;       # a2 + rollout 다양성
  # --- phase 2: Biowulf ckpt-600(현재 best) 에서 이어받는 branch ----------------
  # 근거: ckpt-600 의 per-label 실측에서 macro-F1 의 병목이 희귀 라벨 recall 이고
  #       (Peribronchial thickening R 0.173 dR -0.133, Bronchiectasis R 0.214),
  #       alpha=gamma=1 런이 거기서 지고 있다. gamma 는 라벨 동등가중이라 그 축을
  #       직접 민다. lambda=0 은 BLEU 여유(0.3536 vs 문턱 0.3194 = 5 sigma)를 쓴다.
  #       gamma/alpha 는 둘 다 0.401 위라 유병률 게이밍은 여전히 순손실이다.
  #       출력 디렉터리를 ckpt-600 사본으로 미리 씨딩해야 auto-resume 이 잡는다.
  g3from600)  ALPHA=1.0; GAMMA=3.0 ;;
  g6from600)  ALPHA=1.0; GAMMA=6.0 ;;
  # --- phase 3: ckpt-700 에서 이어받는 branch (2026-08-29) --------------------
  # gamma 축은 반증됐다. storm 동일스텝 대조(nobleu g=1 F1 0.5595 vs g3 g=3 0.5545)에서
  # 지고, 겨냥한 희귀 라벨에서 오히려 recall 을 잃었다(Bronchiectasis dR -0.047).
  # 원인: macro-F1 marginal 은 recall 뿐 아니라 **precision 도** 보상하므로 gamma 를
  # 키우면 정책이 "확신하는 것만 말하기"로 간다 -- 희귀 라벨이 가장 먼저 버려진다.
  # 그리고 lambda=0 이 길이 가드까지 없애 g6 은 완성 233 토큰 / reward_bleu 0.196 으로
  # 무너졌다. 그래서 phase 3 은 (a) gamma 를 올리지 않고 (b) lambda 가드를 남긴다.
  lam05from700) ALPHA=1.0; GAMMA=1.0; LAMBDA=${LAMBDA:-0.5} ;;   # BLEU 예산 절반만 임상으로
  t07from700)   ALPHA=1.0; GAMMA=1.0; LAMBDA=${LAMBDA:-1.0}; TEMP=0.7; TOPP=0.9 ;;  # rollout 을 eval 샘플러와 일치
  *) echo "알 수 없는 variant: $VARIANT"; exit 2 ;;
esac
LAMBDA=${LAMBDA:-0.0}
TOPP=${TOPP:-1.0}
# 40GB 카드에서 ckpt-700 정책(완성이 길다)은 pd>=num_generations 제약 때문에 pd8 이
# OOM 난다. NGEN 을 낮추면 pd 를 낮출 수 있고, pd x accum 을 32 로 유지하면
# **스텝당 프롬프트 수는 그대로**여서 Biowulf 와 비교 가능성이 남는다.
NGEN=${NGEN:-8}

IMAGE=${IMAGE:-reportgen-pillar-ctrg3d:latest}
MNT=${MNT:-/mnt/biowulf}
ENV_HOST=${ENV_HOST:-$HOME/storm_grpoenv}
STAGE=${STAGE:-$HOME/storm_stage}
HREPO=$MNT/drdcad/sunggu/0.Challenge/PETRG-3D
CREPO=/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D
PY=/env/py/cpython-3.10-linux-x86_64-gnu/bin/python3.10
MAX_STEPS=${MAX_STEPS:-1500}
OUT_HOST=$HREPO/outputs/grpo_B_$VARIANT
OUT_GUEST=$CREPO/outputs/grpo_B_$VARIANT

echo "=============================================================="
echo " storm GRPO  variant=$VARIANT  alpha=$ALPHA gamma=$GAMMA lambda=$LAMBDA temp=$TEMP top_p=$TOPP G=$NGEN"
echo "   GPU(container)=$GPU  max_steps=$MAX_STEPS  $(date '+%F %T')"
echo "=============================================================="

# ---- 사전검사: 여기서 크게 실패하는 편이 30시간 뒤에 죽는 것보다 낫다 --------
fail(){ echo "  FAIL $1"; exit 1; }
ok(){ echo "  OK   $1"; }
[ -d "$ENV_HOST/lib/trl" ] || fail "py3.10 GRPO 스택 없음 -- bash storm/build_grpo_env.sh"; ok "grpo env"
for f in Qwen3-8B/config.json cache_newencB/train.npy cache_newencB/train_meta.json \
         ckpt/newencB_ep15_ckpt7455/model.safetensors \
         grpo_ref_newencB/subset12000.json grpo_ref_newencB/reference.json; do
  [ -s "$STAGE/$f" ] || fail "stage 미완: $f  (ONLY=grpo bash storm/stage.sh)"
done; ok "stage 6종"
n=$(find "$STAGE/cache_newencB" "$STAGE/grpo_ref_newencB" "$STAGE/ckpt" -type f ! -perm -o+r | wc -l)
[ "$n" = 0 ] || fail "stage 에 컨테이너가 못 읽는 파일 $n 개 -- chmod -R a+rX $STAGE"; ok "stage 읽기 권한"
grep -q '"--prompts_only"' "$HREPO/scripts/grpo_newencB.py" || fail "--prompts_only 없음 (구 드라이버)"; ok "--prompts_only 지원"
# 보상의 RadBERT 체크포인트. 코드 기본값은 /data/MIDS/... 인데 storm 에는 MIDS 마운트가
# 없다(sshfs 는 drdcad 만 보여 준다). drdcad 사본을 그 경로 위에 bind 해서 코드도
# 경로도 건드리지 않는다 -- 채점기와 같은 체크포인트여야 보상과 채점이 갈리지 않는다.
RADBERT=${RADBERT:-$HREPO/data/radbert/RadBertClassifier.pth}
[ -s "$RADBERT" ] || fail "RadBERT 사본 없음: $RADBERT
   Biowulf 에서: cp -p /data/MIDS/datasets/CT-RATE/models/RadBertClassifier.pth \\
                    $HREPO/data/radbert/  &&  chgrp drdcad ... && chmod 0440 ..."
ok "RadBERT ($(du -h "$RADBERT" | cut -f1))"
mkdir -p "$OUT_HOST" || fail "출력 디렉터리"; ok "출력 쓰기"

# ---- rung: 1차에 정하고 재개 때는 그대로 읽는다 -----------------------------
#   generation batch(=pd x accum) 가 바뀌면 스텝당 프롬프트 수가 바뀌어
#   이어받은 뒤부터 다른 실험이 된다.
RUNGFILE=$OUT_HOST/.rung
RESUME=""
LATEST=$(ls -1d "$OUT_HOST"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
# 컨테이너에는 OUT_GUEST 경로를 넘겨야 한다 -- $OUT_HOST 는 /mnt/biowulf/... 로
# 시작하는 호스트 경로이고 컨테이너 안에는 그 마운트가 없다(/vf/users, /data 로 붙는다).
# 지금까지 이 버그가 안 드러난 이유는 네 변형 모두 처음부터 시작해서 RESUME 이
# 한 번도 비어 있지 않은 적이 없었기 때문이다.
[ -n "$LATEST" ] && RESUME="--resume_from_checkpoint $OUT_GUEST/$(basename "$LATEST")" \
    && echo ">>> 재개: $(basename "$LATEST")  (guest: $OUT_GUEST/$(basename "$LATEST"))"

run() {   # $@ = 추가 인자
    docker run --rm -i --gpus all --ipc=host \
        -v "$MNT":/vf/users -v "$MNT":/data -v "$ENV_HOST":/env -v "$STAGE":/stage \
        -v "$RADBERT":/data/MIDS/datasets/CT-RATE/models/RadBertClassifier.pth:ro \
        -e CUDA_VISIBLE_DEVICES="$GPU" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
        -e TOKENIZERS_PARALLELISM=false -e WANDB_MODE=offline \
        -e WANDB_DIR=/tmp/wandb -e WANDB_CACHE_DIR=/tmp/wandb-cache \
        -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/.cache \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e PYTHONPATH=/env/lib:/opt/app/rad-vision-engine \
        -w "$CREPO" --entrypoint "$PY" "$IMAGE" scripts/grpo_newencB.py \
            --ckpt /stage/ckpt/newencB_ep15_ckpt7455 \
            --cache_dir /stage/cache_newencB \
            --lang_encoder_path /stage/Qwen3-8B \
            --subset_json /stage/grpo_ref_newencB/subset12000.json \
            --reference_json /stage/grpo_ref_newencB/reference.json \
            --output_dir "$OUT_GUEST" \
            --alpha "$ALPHA" --gamma "$GAMMA" --lambda_bleu "$LAMBDA" \
            --temperature "$TEMP" --top_p "$TOPP" --num_generations "$NGEN" \
            --prompts_only "$@"
}

if [ -s "$RUNGFILE" ]; then
    read -r PD ACC < "$RUNGFILE"
    echo ">>> rung 재사용: pd $PD x accum $ACC (= $(( PD * ACC )) completions/step)"
else
    # RUNGS 는 'pd:accum' 을 공백으로 나열한다 (예: RUNGS='8:4 4:8').
    # 예전 코드는 ${RUNGS:-...} 를 표시용으로만 평가하고 루프는 하드코딩이라
    # override 가 조용히 무시됐다.
    PD=""; ACC=""
    for rung in ${RUNGS:-16:2 8:4 8:2}; do
        set -- ${rung//:/ }
        echo "=============== smoke: pd $1 x accum $2 = $(( $1 * $2 )) completions/step ==============="
        LOG=$(mktemp)
        if run --smoke --per_device_train_batch_size "$1" --gradient_accumulation_steps "$2" \
               --max_steps 2 2>&1 | tee "$LOG" | tail -25; then
            PD=$1; ACC=$2; rm -f "$LOG"; break
        fi
        if grep -qiE "out of memory|CUDA error: out of memory" "$LOG"; then
            echo "   -> OOM. 다음 rung."; rm -f "$LOG"; continue
        fi
        echo "   -> OOM 이 아닌 실패다. 사다리를 태우지 않고 멈춘다."; rm -f "$LOG"; exit 1
    done
    [ -n "$PD" ] || { echo "ERROR: 모든 rung 실패"; exit 1; }
    [ "$(( PD * ACC ))" = 32 ] || echo "WARNING: generation batch $(( PD * ACC )) -- 다른 변형과 맞는지 확인할 것"
    printf '%s %s\n' "$PD" "$ACC" > "$RUNGFILE"
fi

echo "=============== train: $MAX_STEPS steps @ pd $PD x accum $ACC ==============="
run --per_device_train_batch_size "$PD" --gradient_accumulation_steps "$ACC" \
    --max_steps "$MAX_STEPS" $RESUME
echo "=============== done  $VARIANT  $(date '+%F %T') ==============="
