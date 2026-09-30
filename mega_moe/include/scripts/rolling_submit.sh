#!/bin/bash
# Rolling sweep submitter: at most 4 of OUR tasks queued/running at any time.
# - fills the queue back to 4 as soon as a slot frees (rolling, no batches)
# - tracks each run in $SW/STATUS.md (PENDING/ACTIVE/DONE/FAILED/GAVEUP)
# - resubmits silently-lost runs up to 3 times
# - parses finished runs; aggregates cost_model.xlsx when all terminal
# Usage: setsid nohup bash rolling_submit.sh <sweep_root> RUN... &
SW=$1; shift
PY=/mnt/s00951640/envs/wcf_py310/bin/python
SCRIPTS=/mnt/s00951640/ops-transformer-profs-demo-megamoe-profile/megamoe_profile/scripts
LOG=$SW/controller.log
STATUS=$SW/STATUS.md
MAXQ=4
MAXTRIES=3
mkdir -p $SW/pids

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

pid_alive() {  # pid -> 0 alive
  [ -n "$1" ] && [ -d /proc/$1 ] && tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -q "qrun"
}

state_of() {  # name -> DONE | FAILED | ACTIVE | LOST | NEW
  local name=$1 r=$SW/$name
  local bins
  bins=$(ls $r/raw/prof_rank*.bin 2>/dev/null | wc -l)
  [ "$bins" -ge 4 ] && { echo DONE; return; }
  if grep -q "退出码\|超时\|SIGTERM\|SIGKILL\|排队已取消" $r/qrun.log 2>/dev/null; then
    echo FAILED; return
  fi
  local p
  p=$(cat $SW/pids/$name.pid 2>/dev/null)
  if pid_alive "$p"; then echo ACTIVE; return; fi
  if [ -f "$SW/pids/$name.tries" ]; then echo LOST; else echo NEW; fi
}

submit() {  # name
  local r=$SW/$1
  rm -f $r/qrun.log
  setsid nohup qrun -n s00951640-cost-$1 -t 300 "bash $r/run_one.sh" \
    > $r/qrun.log 2>&1 < /dev/null &
  local p=$!
  disown 2>/dev/null
  echo $p > $SW/pids/$1.pid
  echo $(( $(cat $SW/pids/$1.tries 2>/dev/null || echo 0) + 1 )) > $SW/pids/$1.tries
  log "SUBMIT $1 (try $(cat $SW/pids/$1.tries), pid $p)"
}

parse_one() {  # name
  local r=$SW/$1
  (cd $r/raw && $PY $SCRIPTS/parse_prof.py --n-cores 84 --aiv-cores 56 \
     --event-names-from $r/build/generated/profile_events_generated.h \
     --pair-map $r/build/generated/pair_map.json \
     prof_rank0.bin prof_rank1.bin prof_rank2.bin prof_rank3.bin \
     > $r/parse.log 2>&1) && log "PARSE_OK $1"
}

write_status() {
  local done_n=0 act_n=0 pend_n=0 fail_n=0
  local s_done="" s_act="" s_pend="" s_fail=""
  for name in "${ALL[@]}"; do
    case $(state_of $name) in
      DONE)  done_n=$((done_n+1)); s_done+="- $name ✓\n";;
      ACTIVE) act_n=$((act_n+1)); s_act+="- $name\n";;
      FAILED) fail_n=$((fail_n+1)); s_fail+="- $name（已执行但无产物/超时）\n";;
      LOST)   pend_n=$((pend_n+1)); s_pend+="- $name（曾入队但丢失，待重提 try=$(cat $SW/pids/$name.tries 2>/dev/null||echo 0)/$MAXTRIES）\n";;
      *)      pend_n=$((pend_n+1)); s_pend+="- $name（未提交）\n";;
    esac
  done
  {
    echo "# Cost sweep 状态（$(date '+%m-%d %H:%M:%S') 自动更新）"
    echo ""
    echo "规则：同时在队 ≤ $MAXQ；完成 1 个自动补位 1 个"
    echo ""
    echo "## 汇总：完成 $done_n / 共 ${#ALL[@]}，在队 $act_n/$MAXQ，待提交 $pend_n，失败 $fail_n"
    echo ""
    echo "## 已完成 ($done_n)"; printf '%b' "$s_done"
    echo "## 在队/执行中 ($act_n)"; printf '%b' "$s_act"
    echo "## 待提交 ($pend_n)"; printf '%b' "$s_pend"
    echo "## 失败 ($fail_n)"; printf '%b' "$s_fail"
  } > $STATUS
}

ALL=("$@")
log "==== rolling controller start: ${#ALL[@]} runs, maxq=$MAXQ ===="
while :; do
  # settle: parse new DONE runs
  for name in "${ALL[@]}"; do
    if [ "$(state_of $name)" = DONE ] && [ ! -f $SW/pids/$name.parsed ]; then
      parse_one $name && touch $SW/pids/$name.parsed
    fi
  done
  # fill queue to MAXQ from NEW then LOST (retryable)
  active=0
  for name in "${ALL[@]}"; do
    [ "$(state_of $name)" = ACTIVE ] && active=$((active+1))
  done
  for name in "${ALL[@]}"; do
    [ $active -ge $MAXQ ] && break
    st=$(state_of $name)
    if [ "$st" = NEW ]; then
      submit $name; active=$((active+1))
    elif [ "$st" = LOST ] && [ "$(cat $SW/pids/$name.tries 2>/dev/null || echo 0)" -lt $MAXTRIES ]; then
      submit $name; active=$((active+1))
    fi
  done
  write_status
  # all terminal?
  finished=1
  for name in "${ALL[@]}"; do
    case $(state_of $name) in DONE|FAILED) ;; *) finished=0;; esac
  done
  [ $finished -eq 1 ] && break
  sleep 20
done
write_status
runs=()
for name in "${ALL[@]}"; do [ -f $SW/$name/raw/prof_paired.csv ] && runs+=("$SW/$name"); done
log "AGGREGATE ${#runs[@]} runs"
$PY $SCRIPTS/aggregate_cost.py --freq 1000 --output $SW/cost_model.xlsx "${runs[@]}" >> $LOG 2>&1 \
  && log "XLSX_OK $SW/cost_model.xlsx"
log "==== rolling controller finished ===="
