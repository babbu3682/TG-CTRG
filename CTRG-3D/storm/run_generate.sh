#!/bin/bash
# ==============================================================================
# storm(로컬 DGX, A100-40GB x4) 에서 docker 로 리포트를 **생성만** 한다.
# 채점은 Biowulf CPU 에서 한다 (run_sbatch_score_cpu) -- 이유는 아래.
#
#   bash storm/run_generate.sh rlB                    # base 100 200
#   GPUS="3" bash storm/run_generate.sh ab_gamma2      # 800 1000
#   TAGS="300 400" GPUS="0 1" bash storm/run_generate.sh rlB
#
# storm 전용이다. Biowulf 쪽 런처/스크립트에는 storm 분기를 넣지 않는다.
# 준비: bash storm/stage.sh  (자세한 이유는 storm/README.md)
#
# 왜 생성만 하나
# --------------
# 이미지에 pycocoevalcap / nltk / java 가 없어 evaluate_reportgen.py 가 돌지 않는다.
# 반면 채점은 GPU 가 필요 없고(infer.py 가 cuda-if-available), Biowulf 의 norm
# 파티션은 CPU 가 수천 개 놀고 있어 대기가 사실상 0 이다. 그래서 무거운 쪽(생성)만
# storm 이 맡는다. 덤으로 Biowulf 호스트에는 java 가 있어 METEOR 까지 나온다.
#
# 왜 이 이미지인가
# ----------------
# repo 의 venv 는 torch 2.13.0+cu130 이라 드라이버 >= 580 을 요구하는데 storm 은
# 535.129.03 이다. 반면 이 프로젝트용으로 이미 만들어져 있던
# reportgen-pillar-ctrg3d:latest 는 torch 2.8.0+cu126 / transformers 4.57.0
# (Biowulf 와 동일 버전) 이고 A100 4장을 정상 인식한다. trl 은 grpo_crg.py 의
# 함수 안에서만 import 되므로 없어도 된다.
#
# 왜 -u 로 uid 를 넘기지 않나
# ---------------------------
# 이미지의 rad-vision-engine 이 editable install 이고 소스가
# /opt/app/rad-vision-engine/rve (drwxrws--- runner:runner) 에 있다. 호스트 uid 로
# 실행하면 `import rve` 가 PermissionError 로 죽는다 -- 캐시 특징을 쓰므로 rve 를
# 실제로 호출하지는 않지만 ctrate_dataset_train 이 모듈 최상단에서 import 한다.
# 이미지 기본 유저(runner, uid 1000)가 그 디렉터리의 소유자다.
# uid 를 넘기지 않아도 출력 소유권은 문제되지 않는다: /mnt/biowulf 는 sshfs 라
# 컨테이너 uid 와 무관하게 원격 ssh 사용자(mathaits:drdcad, -rw-r-----)로 떨어지고,
# results/<sub>/ 가 drwxrws--- kyungs2:drdcad 라 storm 은 쓸 수 있고 kyungs2 는
# 읽을 수 있다(채점에 필요한 것은 읽기다).
#
# 왜 stage 하나
# -------------
# /mnt/biowulf 는 helix 로의 sshfs 라 단일 스트림 23.8 MB/s 다. Qwen3-8B 16GB 를
# 매번 읽으면 ~11분이다. --lang_encoder_path 와 --cache_dir 는 둘 다 CLI 플래그라
# 로컬 /home 사본을 가리키게만 하면 된다 -- 코드는 건드리지 않는다.
# ==============================================================================
set -euo pipefail

MNT=${MNT:-/mnt/biowulf}
STAGE=${STAGE:-$HOME/storm_stage}
# lz4 valid 캐시의 로컬 사본. 컨테이너 안에서 **원래 절대경로 위에 덮어씌운다** --
# manifest 의 image_cache_path 가 절대경로라 그래야 코드도 manifest 도 안 고친다.
LZ4_HOST=${LZ4_HOST:-$STAGE/lz4_valid}
LZ4_GUEST=/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/valid
IMAGE=${IMAGE:-reportgen-pillar-ctrg3d:latest}
BATCH=${BATCH:-24}            # A100 **40GB**. 80GB 카드에서 batch 32 가 32.4GB 였다
NGPU=${NGPU:-4}
# 쓸 컨테이너 GPU 인덱스(공백 구분). 두 MODE 를 동시에 돌릴 때 카드를 나눈다.
#   예) rlB 는 GPUS="0 1 2", ab_gamma2 는 GPUS="3"
# 호스트 nvidia-smi 는 GPU 5장(3번이 DGX Display)을 보고하지만 컨테이너 안 CUDA 는
# 디스플레이를 빼고 0..3 을 A100 4장에 매핑한다 -- 여기 쓰는 것은 후자다.
GPUS=${GPUS:-$(seq -s" " 0 $((NGPU-1)))}
DOCKER=${DOCKER:-docker}

CREPO=/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D
HREPO=$MNT/drdcad/sunggu/0.Challenge/PETRG-3D

MODE="${1:-}"
case "$MODE" in
  rlB)       SUB=grpo_newencB;   RUN_SUB=grpo_newencB;  DEF_TAGS="base 100 200"; STCACHE=cache_newencB;   CKPT_NAME=newencB_ep15_ckpt7455 ;;
  ab_gamma2) SUB=grpo_ab_gamma2; RUN_SUB=grpo_ab_gamma2; DEF_TAGS="800 1000";    STCACHE=cache_ckpt16412; CKPT_NAME=checkpoint-16412 ;;
  ab_g16)    SUB=grpo_ab_g16;    RUN_SUB=grpo_ab_g16;    DEF_TAGS="800 1000";    STCACHE=cache_ckpt16412; CKPT_NAME=checkpoint-16412 ;;
  # 샘플러 스윕 -- 어댑터는 ckpt-600 하나로 고정하고 디코딩만 바꾼다. 학습 없이
  # CRG/macro-F1 을 올릴 수 있는지 보는 실험이라 delta 기준선은 ckpt-600 자신이다.
  # 가설: repetition_penalty 1.05 가 지금 잃고 있는 라벨을 직접 억제한다 --
  #   Peribronchial thickening(R 0.173) / Bronchiectasis / Interlobular septal
  #   thickening 이 "thickening"/"bronch" 토큰을 공유해서, 한 번 쓰고 나면
  #   두 번째 언급의 확률이 깎인다.
  sampler)   SUB=grpo_newencB;   RUN_SUB=grpo_newencB;  DEF_TAGS="600_rep100 600_t10p95"; STCACHE=cache_newencB; CKPT_NAME=newencB_ep15_ckpt7455
             SAMPLER_TAG=${SAMPLER_TAG:-600} ;;
  # sampler:<run> -- 임의 런의 어댑터 하나를 고정하고 디코딩만 바꾼다.
  #   MODE=sampler 는 grpo_newencB 에 고정돼 있어 최고 BLEU 정책(a2from700)의 샘플러
  #   변형을 만들 수 없었다. 접미사 규칙(tag_flags)은 그대로 쓴다.
  #   예) SAMPLER_TAG=900 TAGS="900_t10p95" bash storm/run_generate.sh sampler:grpo_B_a2from700
  sampler:*) _sr=${MODE#sampler:}; SUB=$_sr; RUN_SUB=$_sr; DEF_TAGS="900_t10p95"
             STCACHE=cache_newencB; CKPT_NAME=newencB_ep15_ckpt7455
             SAMPLER_TAG=${SAMPLER_TAG:-900} ;;
  # storm 에서 학습한 newenc B 계열 변형 (nobleu / g3 / a2 / a2t12 / *from600).
  # rlB 와 같은 인코더·캐시·split 이고 출력 디렉터리만 다르다.
  grpo_B_*)  SUB=$MODE; RUN_SUB=$MODE; DEF_TAGS="700"; STCACHE=cache_newencB; CKPT_NAME=newencB_ep15_ckpt7455 ;;
  *) echo "usage: bash $0 {rlB|ab_gamma2|ab_g16|sampler|sampler:<run>|grpo_B_<변형>}" >&2; exit 1 ;;
esac
TAGS="${TAGS:-$DEF_TAGS}"
OUT=$CREPO/results/$SUB
HOUT=$HREPO/results/$SUB

# rlB 만 새 인코더라 추가 플래그가 붙는다
# --ckpt / --adapter 는 **staged 로컬 사본**을 가리킨다. 둘 다 sshfs 위에서 읽으면
# model.safetensors 17GB 를 24 MB/s 로 매번 읽어 컨테이너당 12분이 날아가고, 태그를
# 병렬로 돌리면 그 링크를 나눠 쓴다. basename 은 유지해야 한다 --
# generate_batched.py:205 가 캐시 meta 의 checkpoint_tag 와 대조한다.
case "$MODE" in rlB|sampler|sampler:*|grpo_B_*) _NEWENC=1 ;; *) _NEWENC=0 ;; esac
if [ "$_NEWENC" = 1 ]; then
  GEN_EXTRA="--ckpt /stage/ckpt/$CKPT_NAME \
             --model_config_yaml $CREPO/src/Model/ctrate_pillar0_chest_lapa.yaml \
             --vis_dim 384 --split valid_newenc"
else
  GEN_EXTRA="--ckpt /stage/ckpt/$CKPT_NAME"
fi

echo "=============================================================="
echo " storm generate  MODE=$MODE  tags=$TAGS  batch=$BATCH  $(date '+%F %T')"
echo "   image : $IMAGE"
echo "   stage : $STAGE  (Qwen3-8B, $STCACHE, ckpt/$CKPT_NAME)"
echo "   gpus  : $GPUS"
echo "   out   : $HOUT"
echo "=============================================================="

# ---- 사전검사 ----------------------------------------------------------------
fail=0; say(){ printf "  %-4s %s\n" "$1" "$2"; [ "$1" = NO ] && fail=1; return 0; }
$DOCKER image inspect "$IMAGE" >/dev/null 2>&1 && say OK "image" || say NO "image 없음: $IMAGE"
[ -s "$STAGE/Qwen3-8B/config.json" ] && say OK "stage Qwen3-8B" || say NO "stage 미완: $STAGE/Qwen3-8B (bash ~/storm_stage.sh)"
[ -s "$STAGE/$STCACHE/valid.npy" ]   && say OK "stage $STCACHE" || say NO "stage 미완: $STAGE/$STCACHE"
# 컨테이너는 이미지 기본 유저(uid 1000)로 돌므로 stage 가 world-readable 이어야 한다.
# rsync -a 가 원본의 -rw-r----- 를 보존하는데, 그 파일을 safetensors 는
# **FileNotFoundError** 로 보고한다(PermissionError 가 아니다) -- 원인을 찾기 어려우니
# 깊이 제한 없이 전수로 검사한다.
# lz4 로컬 사본이 없으면 컨테이너가 sshfs 에서 1,552 볼륨(38 GB)을 읽는다 --
# 프롬프트를 만들 때 CTRATEDataset_Test 가 CT 볼륨까지 로드하기 때문이다(그리고
# generate_batched 는 그 텐서를 쓰지 않는다). 병렬로 돌리면 D 상태로 사실상 멈춘다.
_lz4n=$(ls -1 "$LZ4_HOST"/*.tar.lz4 2>/dev/null | wc -l)
[ "$_lz4n" -ge 1552 ] && say OK "stage lz4_valid ($_lz4n)" \
    || say NO "stage 미완: lz4_valid ($_lz4n/1552) -- bash storm/stage.sh"
_nr=$(find "$STAGE" -type f ! -perm -o+r 2>/dev/null | wc -l)
_nx=$(find "$STAGE" -type d ! -perm -o+x 2>/dev/null | wc -l)
if [ "$_nr" -eq 0 ] && [ "$_nx" -eq 0 ]; then say OK "stage 읽기 권한"
else say NO "stage 에 uid 1000 이 못 읽는 파일 $_nr / 못 여는 디렉터리 $_nx -- chmod -R a+rX $STAGE"; fi
[ -s "$STAGE/ckpt/$CKPT_NAME/model.safetensors" ] && say OK "stage ckpt $CKPT_NAME" \
    || say NO "stage 미완: $STAGE/ckpt/$CKPT_NAME  (bash storm/stage.sh)"
CHECK_TAGS="$TAGS"
case "$MODE" in sampler|sampler:*) CHECK_TAGS="$SAMPLER_TAG" ;; esac
for t in $CHECK_TAGS; do
  [ "$t" = base ] && continue
  [ -s "$STAGE/adapters/$RUN_SUB/checkpoint-$t/adapter_model.safetensors" ] \
      && say OK "stage adapter $t" \
      || say NO "stage 미완: adapter checkpoint-$t  (bash storm/stage.sh)"
done
mkdir -p "$HOUT" 2>/dev/null || true
# 계정이 둘인 파일시스템이다 -- Biowulf(kyungs2)가 먼저 만든 디렉터리는 0750 으로
# 떨어져 storm(sshfs 원격 유저 mathaits)이 못 쓴다. 이 세션에만 네 번 났다.
chmod 2770 "$HOUT" 2>/dev/null || true
touch "$HOUT/.w" 2>/dev/null && { say OK "출력 쓰기"; rm -f "$HOUT/.w"; } \
    || say NO "출력에 쓸 수 없다: $HOUT (소유자 $(stat -c %U "$HOUT" 2>/dev/null))
       Biowulf 에서:  chmod 2770 results/$SUB"
[ "$fail" -eq 0 ] || { echo "사전검사 실패" >&2; exit 1; }

# 태그 하나가 필요로 하는 --adapter 와 샘플러 플래그를 돌려준다.
# sampler 모드에서만 디코딩이 바뀌고, 나머지 모드는 기존 동작 그대로다.
tag_flags() {   # $1 = tag
    if [ "$MODE" = sampler ] || [ "${MODE%%:*}" = sampler ]; then
        printf -- '--adapter /stage/adapters/%s/checkpoint-%s ' "$RUN_SUB" "$SAMPLER_TAG"
        # 접미사로 매칭한다 -- SAMPLER_TAG 를 바꾸면 어느 체크포인트에도 그대로 쓴다.
        #   예) SAMPLER_TAG=800 TAGS="800_t10p95" bash storm/run_generate.sh sampler
        case "$1" in
          *_rep100_t10p95_s[0-9]|*_rep100_t10p95_s[0-9][0-9]) printf -- '--repetition_penalty 1.00 --temperature 1.0 --top_p 0.95 --seed %s' "${1##*_s}" ;;
          *_rep100_s[0-9]|*_rep100_s[0-9][0-9]) printf -- '--repetition_penalty 1.00 --seed %s' "${1##*_s}" ;;
          *_t10p95_s[0-9]|*_t10p95_s[0-9][0-9]) printf -- '--temperature 1.0 --top_p 0.95 --seed %s' "${1##*_s}" ;;
          *_s[0-9]|*_s[0-9][0-9]) printf -- '--seed %s' "${1##*_s}" ;;
          *_rep100_t10p95) printf -- '--repetition_penalty 1.00 --temperature 1.0 --top_p 0.95' ;;
          *_rep100) printf -- '--repetition_penalty 1.00' ;;
          *_t10p95) printf -- '--temperature 1.0 --top_p 0.95' ;;
          *_default) : ;;
          600_rep100_LEGACY) printf -- '--repetition_penalty 1.00' ;;
          # 두 축을 합친 것. t10p95 실측에서 최악 두 라벨이 크게 올랐고
          # (Peribronchial thickening R 0.173->0.341, Bronchiectasis 0.214->0.298)
          # rep 1.00 은 그 라벨들이 공유하는 토큰의 억제를 직접 푼다 -- 기전이 다르므로
          # 겹칠 여지가 있다. 대신 중간 라벨 precision 손실이 커질 수 있어 셋을 다 잰다.
          *) echo "알 수 없는 sampler 태그: $1" >&2; return 1 ;;
        esac
    elif [ "$1" != base ]; then
        printf -- '--adapter /stage/adapters/%s/checkpoint-%s' "$RUN_SUB" "$1"
    fi
}

gen() {   # $1 = container GPU index, $2 = tag, $3 = extra flags
    $DOCKER run --rm --gpus all --ipc=host \
        -v "$MNT":/vf/users -v "$MNT":/data -v "$STAGE":/stage \
        -v "$LZ4_HOST":"$LZ4_GUEST":ro \
        -e CUDA_VISIBLE_DEVICES="$1" \
        -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e TOKENIZERS_PARALLELISM=false \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -w "$CREPO" --entrypoint bash "$IMAGE" -lc \
        "python scripts/generate_batched.py $GEN_EXTRA \
             --lang_encoder_path /stage/Qwen3-8B --cache_dir /stage/$STCACHE \
             --batch_size $BATCH $3"
}

# SKIP_SMOKE=1 은 같은 MODE 의 smoke 가 이미 통과했을 때만 쓴다. 태그/캐시/체크포인트가
# 바뀌면 반드시 다시 돌린다 -- smoke 는 5분이고, 잘못된 생성은 4시간이다.
# ---- smoke: 32볼륨 (GPU 0) ---------------------------------------------------
FIRST=${TAGS%% *}
SMOKE_ADPT="$(tag_flags "$FIRST")" || exit 1
echo "=============== smoke: 32 volumes ($FIRST) ==============="
set -- $GPUS; SMOKE_GPU=$1
if [ "${SKIP_SMOKE:-0}" = 1 ]; then
    echo "   [smoke] SKIP_SMOKE=1 -- 건너뜀"
else
gen "$SMOKE_GPU" "$FIRST" "$SMOKE_ADPT --n 32 --out $OUT/_smoke_storm_${MODE}_$$.csv"
python3 - "$HOUT/_smoke_storm_${MODE}_$$.csv" <<'PY'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1])))
assert len(rows)==32, f"smoke rows {len(rows)}"
e=sum(1 for r in rows if not r["Pred_report"].strip())
print(f"[smoke] {len(rows)} rows, {e} empty")
assert e < len(rows), "smoke produced only empty predictions"
PY
rm -f "$HOUT/_smoke_storm_${MODE}_$$.csv"
fi

# ---- 태그를 GPU 에 라운드로빈으로 붙여 병렬 실행 ------------------------------
i=0; pids=""; names=""
for tag in $TAGS; do
    if [ -s "$HOUT/${tag}.csv" ]; then echo "이미 있음, 건너뜀: ${tag}.csv"; continue; fi
    set -- $GPUS; shift $(( i % $# )); g=$1; i=$(( i + 1 ))
    adpt="$(tag_flags "$tag")" || exit 1
    echo "=============== generate: $tag  -> GPU $g ==============="
    gen "$g" "$tag" "$adpt --out $OUT/${tag}.csv" > "$HOME/storm_gen_${MODE}_${tag}.log" 2>&1 &
    pids="$pids $!"; names="$names $tag"
    sleep 20          # 모델 로딩 피크(CPU RAM)가 겹치지 않게 어긋나게 띄운다
done

rc=0; set -- $names
for p in $pids; do
    if wait "$p"; then echo "  done: $1"; else echo "  FAILED: $1 (see ~/storm_gen_${MODE}_$1.log)"; rc=1; fi
    shift
done

echo "=============== 결과 ==============="
for tag in $TAGS; do
    f="$HOUT/${tag}.csv"
    if [ -s "$f" ]; then printf "  %-6s %8s rows  %s\n" "$tag" "$(( $(wc -l < "$f") - 1 ))" "$f"
    else printf "  %-6s (없음)\n" "$tag"; fi
done
echo
echo "다음: Biowulf 에서 채점 (CPU only, GPU 대기 없음)"
echo "  sbatch run_sbatch_score_cpu $SUB $TAGS"
exit $rc
