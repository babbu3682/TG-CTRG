#!/bin/bash
# ==============================================================================
# interactive **job 자체가 끝났을 때** GPU 할당을 다시 잡아 RL 갈래를 이어 돌린다.
#
#   sbatch run_sbatch_keepalive_watch          # 인자 없음
#
# 기존 scripts/watch_kyungs2_gpus.sh 와 역할이 다르다 -- 겹치지 않는다:
#   watch_kyungs2_gpus.sh : job 은 살아 있는데 **갈래만** 죽었을 때 그 job 의 유휴
#                           GPU 에 재부착한다 (srun --overlap).
#   이 스크립트          : **job 이 통째로 끝났을 때** 새 interactive 할당을 잡는다.
#
# 왜 필요한가 (2026-08-30 실측)
#   28608544(OnDemand VSCode 세션, cn0093, a2 갈래)  -> 08-30 13:12:30 만료
#   28631449(ctrg-grpoB-agg, cn0097, base+a2lam05)   -> 08-30 13:25:39 만료
#   둘 다 오늘 낮에 13분 간격으로 죽는데, 그때 A100 세 장의 RL 이 같이 멈춘다.
#   watch_kyungs2_gpus.conf 는 JOBID 가 박혀 있어 죽은 job 을 "건너뜀" 으로 흘린다.
#   즉 **아무도 재할당을 하지 않는다** -- 그 구멍을 메우는 것이 이 스크립트다.
#
# 두 가지 안전 규칙
#   1. 생사 판정이 **불확실하면 살아 있는 것으로 본다.** 오판으로 같은 output_dir 에
#      두 프로세스가 붙으면 체크포인트가 조용히 깨진다. 늦게 재개하는 쪽이 싸다.
#   2. 내가 낸 job 이 PD/R 인 동안 그 갈래는 "덮여 있음" 으로 본다. 모델 로딩에
#      ~3분 걸려 그 창에서는 프로세스가 안 보이는데, 그때 또 내면 두 개가 뜬다.
#
# 슬롯 제약 (sacctmgr 실측): interactive QoS  MaxJobsPU 2  MaxSubmitPU 2
#                            MaxTRESPU cpu=64, gres/gpu=4   MaxWall 1-12:00:00
#   -> 두 job 이 살아 있는 동안에는 **미리 낼 수 없다.** 죽기를 기다렸다가 낸다.
# ==============================================================================
set -uo pipefail

REPO=/data/drdcad/sunggu/0.Challenge/PETRG-3D
CONF=${KEEPALIVE_CONF:-$REPO/scripts/keepalive_branches.conf}
GPUWATCH_CONF=${GPUWATCH_CONF:-$REPO/scripts/watch_kyungs2_gpus.conf}
LAUNCHER=$REPO/run_sinteractive_grpoB_keepalive
LOGDIR=${LOGDIR:-/data/drdcad/sunggu/logs}
STATE=${STATE:-$LOGDIR/.keepalive_state}

INTERVAL=${INTERVAL:-60}
WALL=${KEEPALIVE_WALL:-1-00:00:00}      # interactive MaxWall 은 1-12:00:00. 짧을수록 backfill 이 잘 된다
CPU_PER_GPU=${CPU_PER_GPU:-16}
MEM_PER_GPU=${MEM_PER_GPU:-60}
JOB_QUOTA=${JOB_QUOTA:-2}
GPU_QUOTA=${GPU_QUOTA:-4}
CPU_QUOTA=${CPU_QUOTA:-64}
MIN_SUBMIT_GAP=${MIN_SUBMIT_GAP:-180}   # 연속 제출 사이 최소 간격(초)
MAX_SUBMITS=${MAX_SUBMITS:-25}          # 갈래당 총 제출 한도 -- 즉사 루프 방지
MAX_STEPS=${MAX_STEPS:-1500}            # 런처와 같아야 한다. 이 스텝에 도달하면 "완료"다
CONSOLIDATE_MIN=${CONSOLIDATE_MIN:-30}  # 내 다른 interactive job 이 N분 안에 끝나면 기다렸다 한 번에 낸다
STOP_FILE=${STOP_FILE:-$LOGDIR/.keepalive_STOP}

log(){ echo "[$(date '+%F %T')] $*"; }
declare -A SUBMITS=()
LAST_SUBMIT=0

# squeue 가 일시적으로 실패하는 것과 job 이 정말 끝난 것을 구분한다. 구분하지 않으면
# 통신 한 번 튄 것을 "끝났다" 로 읽고 살아 있는 갈래 위에 두 번째를 띄운다.
#   RUNNING/PENDING/... : 그 상태
#   ""                  : squeue 성공 + 결과 없음 = 정말 끝났다
#   UNKNOWN             : squeue 자체가 실패 -- 판단하지 않는다
job_state(){
  local out rc
  out=$(squeue -h -j "$1" -o "%T" 2>/dev/null); rc=$?
  if [ "$rc" -ne 0 ]; then
    # squeue 는 "job 이 없다" 에도 비-0 을 낼 수 있다. sacct 로 한 번 더 본다.
    if sacct -n -X -j "$1" -o State 2>/dev/null | grep -qE '.'; then echo ""; else echo UNKNOWN; fi
    return
  fi
  printf '%s\n' "$out" | head -1
}

# squeue 의 TIME_LEFT("D-HH:MM:SS" / "HH:MM:SS" / "MM:SS")를 분으로. 못 읽으면 큰 값.
mins_left(){
  local t d h m s
  t=$(squeue -h -j "$1" -o "%L" 2>/dev/null | head -1 | tr -d ' ')
  case "$t" in ''|UNLIMITED|INVALID|NOT_SET) echo 99999; return;; esac
  d=0; case "$t" in *-*) d="${t%%-*}"; t="${t#*-}";; esac
  case "$t" in
    *:*:*) h="${t%%:*}"; m="${t#*:}"; m="${m%%:*}"; s="${t##*:}";;
    *:*)   h=0; m="${t%%:*}"; s="${t##*:}";;
    *)     h=0; m=0; s="$t";;
  esac
  echo $(( 10#${d:-0}*1440 + 10#${h:-0}*60 + 10#${m:-0} + (10#${s:-0} > 0 ? 1 : 0) )) 2>/dev/null || echo 99999
}

mkdir -p "$LOGDIR"
touch "$STATE" 2>/dev/null || { log "STATE 를 쓸 수 없다: $STATE"; exit 1; }

log "keepalive 감시 시작"
log "  conf=$CONF  launcher=$(basename "$LAUNCHER")  wall=$WALL  interval=${INTERVAL}s"
log "  쿼터: job<=$JOB_QUOTA  gpu<=$GPU_QUOTA  cpu<=$CPU_QUOTA (interactive QoS)"

# ---- conf 읽기 -----------------------------------------------------------
read_conf(){
  NAMES=(); DIRS=(); BASES=()
  while IFS= read -r l; do
    case "$l" in ''|'#'*) continue;; esac
    local nm rest d
    nm="${l%%|*}"; rest="${l#*|}"; d="${rest%|*}"
    NAMES+=("$nm"); DIRS+=("$d"); BASES+=("$(basename "$d")")
  done < "$CONF"
}

# ---- 내 interactive job 들 ------------------------------------------------
# 출력: INT_IDS(전체 R+PD)  INT_RUN(R만)  USED_GPU  USED_CPU
scan_jobs(){
  INT_IDS=(); INT_RUN=(); USED_GPU=0; USED_CPU=0
  local line id st part cpus tres g
  while IFS='|' read -r id st part cpus tres; do
    [ -n "${id:-}" ] || continue
    [ "$part" = interactive ] || continue
    case "$st" in RUNNING|PENDING|CONFIGURING|COMPLETING) ;; *) continue;; esac
    INT_IDS+=("$id")
    [ "$st" = RUNNING ] && INT_RUN+=("$id")
    USED_CPU=$(( USED_CPU + ${cpus:-0} ))
    case "$tres" in
      gres/gpu*) g="${tres##*:}"; [ "$g" -eq "$g" ] 2>/dev/null && USED_GPU=$(( USED_GPU + g )) || USED_GPU=$(( USED_GPU + 1 ));;
    esac
  done < <(squeue -h -u "$USER" -o "%i|%T|%P|%C|%b" 2>/dev/null)
}

# ---- 살아 있는 갈래 basename 수집 ----------------------------------------
# 프로브가 실패했는지 성공했는지 구분해야 한다. 빈 결과가 "갈래 없음" 일 수도,
# "프로브 실패" 일 수도 있기 때문이다. 그래서 마커를 먼저 찍게 한다.
# ps 를 sed 로 훑으므로 프로브 자신의 커맨드라인도 걸리는데, 거기서 나오는
# basename 은 실제 갈래 이름과 절대 같지 않아 무해하다(멤버십 검사만 한다).
PROBE_CMD='echo __PROBE_OK__; ps -u "$USER" -o args= | sed -n "s/.*--output_dir \([^ ]*\).*/\1/p" | sed "s#.*/##" | sort -u'
scan_alive(){
  ALIVE=" "; PROBE_FAIL=0
  local j out
  for j in "${INT_RUN[@]:-}"; do
    [ -n "$j" ] || continue
    out=$(timeout 120 srun --overlap --jobid="$j" bash -c "$PROBE_CMD" 2>/dev/null)
    if ! printf '%s\n' "$out" | grep -q '__PROBE_OK__'; then
      log "  프로브 실패: job $j -- 이번 회차는 보수적으로 간다"
      PROBE_FAIL=1; continue
    fi
    while IFS= read -r b; do
      [ -n "$b" ] && [ "$b" != "__PROBE_OK__" ] && ALIVE="$ALIVE$b "
    done <<< "$out"
  done
}

# ---- 내가 낸 job 중 아직 살아 있는 것 (덮여 있음 판정) --------------------
# STATE 는 "name jobid" 한 줄씩. job 이 끝나면 줄을 지운다.
prune_state(){
  COVERED=" "; local nm j st tmp
  tmp=$(mktemp "$LOGDIR/.kastate.XXXXXX") || return 0
  while read -r nm j; do
    [ -n "${nm:-}" ] && [ -n "${j:-}" ] || continue
    st=$(job_state "$j")
    case "$st" in
      # UNKNOWN 도 남긴다 -- 확인이 안 되면 "덮여 있다" 로 본다(중복 제출 방지)
      RUNNING|PENDING|CONFIGURING|UNKNOWN) printf '%s %s\n' "$nm" "$j" >> "$tmp"; COVERED="$COVERED$nm ";;
      *) log "  제출분 정리: $nm job $j 상태 '${st:-종료}'";;
    esac
  done < "$STATE"
  mv -f "$tmp" "$STATE"
}

# ---- 새 job 이 뜨면 gpuwatch conf 에 등록 ---------------------------------
# 갈래-GPU 매핑은 추측하지 않고 런처가 찍은 '>>> [nm] GPU g' 를 읽는다(정본).
sync_gpuwatch_conf(){
  local tmp nm j g d ex line changed=0
  [ -f "$GPUWATCH_CONF" ] || return 0
  tmp=$(mktemp "$LOGDIR/.kaconf.XXXXXX") || return 0
  # 살아 있는 job 의 줄만 남긴다
  while IFS= read -r line; do
    case "$line" in ''|'#'*) printf '%s\n' "$line" >> "$tmp"; continue;; esac
    j="${line%%:*}"
    case "$(job_state "$j")" in
      RUNNING|PENDING|CONFIGURING|UNKNOWN) printf '%s\n' "$line" >> "$tmp";;
      *) changed=1;;
    esac
  done < "$GPUWATCH_CONF"
  # 내 job 의 갈래를 추가한다 (이미 있으면 건너뜀)
  while read -r nm j; do
    [ -n "${nm:-}" ] && [ -n "${j:-}" ] || continue
    [ "$(job_state "$j")" = RUNNING ] || continue
    grep -q "^$j:[0-9]*:$nm:" "$tmp" 2>/dev/null && continue
    g=$(grep -oE ">>> \[$nm\] GPU [0-9]+" "$LOGDIR/ctrg-keepalive-$j.out" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    [ -n "$g" ] || continue          # 아직 안 찍혔으면 다음 회차에
    d=""; ex=""
    for i in "${!NAMES[@]}"; do
      [ "${NAMES[$i]}" = "$nm" ] || continue
      d="${DIRS[$i]}"
      ex=$(grep -E "^$nm\|" "$CONF" | head -1); ex="${ex##*|}"
    done
    [ -n "$d" ] || continue
    printf '%s:%s:%s:%s:%s\n' "$j" "$g" "$nm" "$d" "$ex" >> "$tmp"
    log "  gpuwatch 등록: $j:$g:$nm"
    changed=1
  done < "$STATE"
  if [ "$changed" = 1 ]; then mv -f "$tmp" "$GPUWATCH_CONF"; else rm -f "$tmp"; fi
}

# ---- 갈래 단위 감시자(ctrg-gpuwatch)가 항상 하나는 돌게 한다 -------------
# 왜 여기서 챙기나: 2026-08-30 04:15 에 다른 Claude 세션이 내가 걸어 둔 PD 후속
# (afterany 승계)을 **중복 제출로 보고 취소**했다. 이 계정의 큐를 두 주체가 함께
# 관리하므로 "오래 대기하는 PD job" 에 기대는 승계는 취약하다. 대신 매 회차
# "R/PD 인 gpuwatch 가 하나도 없으면 그때 하나 낸다" 로 바꾼다 -- 대기 창이 없다.
# 둘이 동시에 돌면 같은 갈래를 두 번 재부착하므로 **없을 때만** 낸다.
ensure_gpuwatch(){
  local n
  n=$(squeue -h -u "$USER" -n ctrg-gpuwatch -o "%T" 2>/dev/null | grep -cE 'RUNNING|PENDING|CONFIGURING')
  [ "${n:-1}" -eq 0 ] 2>/dev/null || return 0
  log ">> ctrg-gpuwatch 가 없다 -- 하나 띄운다(갈래 단위 복구가 끊기지 않게)"
  local j
  j=$(cd "$REPO" && sbatch --parsable --time 2-00:00:00 \
        --output /data/drdcad/sunggu/logs/%x-%A.out run_sbatch_gpuwatch 2>&1)
  log "   -> $j"
}

# ---- 자기 walltime 이 얼마 안 남았을 때만 후속을 낸다 ---------------------
# 시작하자마자 afterany 로 걸어 두면 몇 시간씩 PD 로 남아 다른 주체에게 "중복" 으로
# 보인다(실제로 취소당했다). 끝나기 직전에만 내면 PD 창이 몇 분이라 그럴 일이 없다.
# 대가: 크래시로 죽으면 후속이 없다 -- 그건 로컬 세션의 30분 점검 루프가 맡는다.
SUCC_DONE=0
self_succeed(){
  [ "$SUCC_DONE" = 0 ] || return 0
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  local m; m=$(mins_left "$SLURM_JOB_ID")
  [ "$m" -le "${SUCC_LEAD_MIN:-45}" ] 2>/dev/null || return 0
  local j
  j=$(cd "$REPO" && KA_WATCH_GEN=1 sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID" run_sbatch_keepalive_watch 2>&1)
  log ">> walltime ${m}분 남음 -- 후속 제출: $j"
  SUCC_DONE=1
}

# ==========================================================================
while true; do
  read_conf
  scan_jobs
  scan_alive
  prune_state
  sync_gpuwatch_conf

  if [ -f "$STOP_FILE" ]; then
    log "STOP 파일이 있다($STOP_FILE) -- 감시를 끝낸다. 재개하려면 파일을 지우고 다시 sbatch."
    exit 0
  fi

  ensure_gpuwatch
  self_succeed

  DEAD=()
  for i in "${!NAMES[@]}"; do
    nm="${NAMES[$i]}"; b="${BASES[$i]}"
    case "$ALIVE" in *" $b "*) continue;; esac      # 프로세스가 보인다
    case "$COVERED" in *" $nm "*) continue;; esac   # 내가 낸 job 이 아직 PD/R
    # 학습이 끝난 갈래를 다시 띄우면 즉시 종료 -> 재제출을 반복한다. max_steps 에
    # 도달한 체크포인트가 있으면 "완료" 로 보고 손대지 않는다.
    last=$(ls -1d "${DIRS[$i]}"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    if [ -n "$last" ] && [ "$last" -ge "$MAX_STEPS" ] 2>/dev/null; then
      log "  완료: $nm (checkpoint-$last >= max_steps $MAX_STEPS) -- 재제출하지 않는다"
      continue
    fi
    DEAD+=("$nm")
  done

  if [ "${#DEAD[@]}" -eq 0 ]; then
    log "정상: 갈래 ${#NAMES[@]}개 전부 살아 있음/덮임  (내 interactive job ${#INT_IDS[@]}개, gpu ${USED_GPU}/${GPU_QUOTA})"
    sleep "$INTERVAL"; continue
  fi

  log "끊긴 갈래: ${DEAD[*]}   (interactive job ${#INT_IDS[@]}/${JOB_QUOTA}, gpu ${USED_GPU}/${GPU_QUOTA}, cpu ${USED_CPU}/${CPU_QUOTA})"

  if [ "$PROBE_FAIL" = 1 ]; then
    log "  -> 프로브가 실패한 job 이 있다. 제출하지 않는다(중복 실행 방지)."
    sleep "$INTERVAL"; continue
  fi
  if [ "${#INT_IDS[@]}" -ge "$JOB_QUOTA" ]; then
    log "  -> interactive 슬롯이 없다(${#INT_IDS[@]}/${JOB_QUOTA}). 기존 job 이 끝나기를 기다린다."
    sleep "$INTERVAL"; continue
  fi
  # 곧 끝날 job 이 있으면 기다렸다가 **한 번에** 낸다. 이유는 슬롯이다: 갈래를 두 번에
  # 나눠 내면 interactive job 2개를 다 써서 사용자가 OnDemand VSCode 세션을 띄울 자리가
  # 없어진다. 한 job 에 3장을 담으면 슬롯 1개 + GPU 1장 + CPU 16 이 남는다.
  # 대기는 유한하다 -- 그 job 은 CONSOLIDATE_MIN 분 안에 반드시 끝난다. 그리고 갈래는
  # 어차피 마지막 체크포인트에서 이어받으므로 이 대기로 잃는 학습량은 없다.
  soon=""
  for j in "${INT_IDS[@]:-}"; do
    [ -n "$j" ] || continue
    grep -q " $j\$" "$STATE" 2>/dev/null && continue     # 내가 낸 job 은 제외
    m=$(mins_left "$j")
    [ "$m" -le "$CONSOLIDATE_MIN" ] 2>/dev/null && soon="$soon $j(${m}m)"
  done
  if [ -n "$soon" ]; then
    log "  -> 곧 끝나는 job 이 있다:$soon. 한 번에 내려고 기다린다(슬롯 절약)."
    sleep "$INTERVAL"; continue
  fi

  now=$(date +%s)
  if [ "$(( now - LAST_SUBMIT ))" -lt "$MIN_SUBMIT_GAP" ]; then
    log "  -> 직전 제출 후 $(( now - LAST_SUBMIT ))s. ${MIN_SUBMIT_GAP}s 는 기다린다."
    sleep "$INTERVAL"; continue
  fi

  n=${#DEAD[@]}
  [ "$n" -gt "$(( GPU_QUOTA - USED_GPU ))" ] && n=$(( GPU_QUOTA - USED_GPU ))
  [ "$n" -gt "$(( (CPU_QUOTA - USED_CPU) / CPU_PER_GPU ))" ] && n=$(( (CPU_QUOTA - USED_CPU) / CPU_PER_GPU ))
  if [ "$n" -lt 1 ]; then
    log "  -> GPU/CPU 쿼터가 남지 않았다. 기다린다."
    sleep "$INTERVAL"; continue
  fi

  SEL=(); for ((k=0;k<n;k++)); do
    nm="${DEAD[$k]}"
    if [ "${SUBMITS[$nm]:-0}" -ge "$MAX_SUBMITS" ]; then
      log "  !! $nm 제출 한도(${MAX_SUBMITS}) 초과 -- 사람이 봐야 한다"; continue
    fi
    SEL+=("$nm")
  done
  [ "${#SEL[@]}" -ge 1 ] || { sleep "$INTERVAL"; continue; }
  n=${#SEL[@]}

  log ">> 제출: 갈래='${SEL[*]}'  gpu=$n  cpu=$(( CPU_PER_GPU * n ))  mem=$(( MEM_PER_GPU * n ))g  time=$WALL"
  jid=$(cd "$REPO" && BRANCHES_ONLY="${SEL[*]}" sbatch --parsable \
          --partition interactive --time "$WALL" \
          --gres=gpu:a100:$n --cpus-per-task=$(( CPU_PER_GPU * n )) --mem $(( MEM_PER_GPU * n ))g \
          --job-name ctrg-keepalive "$LAUNCHER" 2>&1)
  if printf '%s' "$jid" | grep -qE '^[0-9]+$'; then
    log "   -> JobID $jid"
    for nm in "${SEL[@]}"; do
      printf '%s %s\n' "$nm" "$jid" >> "$STATE"
      SUBMITS[$nm]=$(( ${SUBMITS[$nm]:-0} + 1 ))
    done
    LAST_SUBMIT=$(date +%s)
  else
    log "   !! 제출 실패: $jid"
    LAST_SUBMIT=$(date +%s)
  fi
  sleep "$INTERVAL"
done
