#!/bin/bash
# Sequential qrun driver for the repeats sweep: ONE config in the queue at a time.
# (Deliberately queue-polite; qrun itself executes FIFO-serial on hardware.)
# Usage: nohup bash run_sweep_seq.sh <sweep_root> NAME... &
# Progress -> <sweep_root>/progress.log; completion marker -> <sweep_root>/ALL_DONE
SW=$1; shift
LOG=$SW/progress.log
mkdir -p "$SW"
for NAME in "$@"; do
  R=$SW/$NAME
  echo "[$(date '+%m-%d %H:%M:%S')] SUBMIT $NAME" >> "$LOG"
  qrun -n s00951640-reps-$NAME -t 420 "bash $R/run_one.sh" > $R/qrun.log 2>&1
  code=$?
  echo "[$(date '+%m-%d %H:%M:%S')] EXIT $NAME code=$code" >> "$LOG"
done
echo "[$(date '+%m-%d %H:%M:%S')] ALL_DONE" >> "$LOG"
touch $SW/ALL_DONE
