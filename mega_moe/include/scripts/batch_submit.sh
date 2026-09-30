#!/bin/bash
# Batched cost-sweep submitter: <=5 queued at a time; when all 5 settle, submit next 5.
# Also parses each run after settle and aggregates the final Excel.
# Usage: setsid nohup bash batch_submit.sh <sweep_root> RUN_NAME... &
SW=$1; shift
PY=/mnt/s00951640/envs/wcf_py310/bin/python
SCRIPTS=/mnt/s00951640/ops-transformer-profs-demo-megamoe-profile/megamoe_profile/scripts
LOG=$SW/batch_controller.log
BATCH=5

log() { echo "[$(date +%H:%M:%S)] $*" >> $LOG; }

settled() {  # RUN -> 0 running/queued, 1 ok, 2 failed-final
  local r=$SW/$1
  local bins=$(ls $r/raw/prof_rank*.bin 2>/dev/null | wc -l)
  [ "$bins" -ge 4 ] && return 1
  grep -q "退出码" $r/qrun.log 2>/dev/null && return 2   # ran, no bins -> failed
  grep -q "超时" $r/qrun.log 2>/dev/null && return 2
  return 0
}

resubmit() {  # RUN
  local r=$SW/$1
  setsid nohup qrun -n s00951640-cost-$1 -t 300 "bash $r/run_one.sh" \
    > $r/qrun.log 2>&1 < /dev/null &
  disown
}

parse_one() {  # RUN
  local r=$SW/$1
  (cd $r/raw && $PY $SCRIPTS/parse_prof.py --n-cores 84 --aiv-cores 56 \
     --event-names-from $r/build/generated/profile_events_generated.h \
     --pair-map $r/build/generated/pair_map.json \
     prof_rank0.bin prof_rank1.bin prof_rank2.bin prof_rank3.bin \
     > $r/parse.log 2>&1) && log "PARSE_OK $1"
}

declare -A TRIES
log "==== batch controller start: $* ===="
while [ $# -gt 0 ]; do
  # take next batch of BATCH names
  batch=(); while [ $# -gt 0 ] && [ ${#batch[@]} -lt $BATCH ]; do batch+=("$1"); shift; done
  log "SUBMIT_BATCH: ${batch[*]}"
  for r in "${batch[@]}"; do
    if ! settled $r; then TRIES[$r]=$(( ${TRIES[$r]:-0} + 1 )); resubmit $r; log "SUBMIT $r (try ${TRIES[$r]})"; fi
  done
  # wait until whole batch settles
  while :; do
    all=1
    for r in "${batch[@]}"; do
      settled $r; s=$?
      if [ $s -eq 1 ]; then
        [ -f $SW/$r/raw/prof_paired.csv ] || parse_one $r
      elif [ $s -eq 0 ]; then
        # still queued/running: detect silent loss (qrun process gone, no verdict)
        if ! pgrep -f "cost-$r\b" >/dev/null 2>&1; then
          sleep 20
          pgrep -f "cost-$r\b" >/dev/null 2>&1 || { settled $r; [ $? -ne 0 ] && continue; \
            if [ ${TRIES[$r]:-0} -lt 3 ]; then TRIES[$r]=$(( ${TRIES[$r]:-0} + 1 )); resubmit $r; log "RESUBMIT $r (lost, try ${TRIES[$r]})"; else log "GIVEUP $r"; fi; }
        fi
        all=0
      fi
    done
    [ $all -eq 1 ] && break
    sleep 30
  done
  log "BATCH_DONE: ${batch[*]}"
done
# aggregate final Excel over all runs with parsed CSVs
runs=()
for d in $SW/bs*/; do [ -f "$d/raw/prof_paired.csv" ] && runs+=("${d%/}"); done
log "AGGREGATE: ${runs[*]}"
$PY $SCRIPTS/aggregate_cost.py --freq 1000 --output $SW/cost_model.xlsx "${runs[@]}" >> $LOG 2>&1
log "==== batch controller finished: $SW/cost_model.xlsx ===="
