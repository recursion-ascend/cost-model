#!/bin/bash
# Cost-model sweep: per-config build (direct, no qrun) + ONE qrun task per config (<=5min each).
# Usage: bash sweep_cost.sh <sweep_root> NAME:BS[:MGW] ...
#   NAME:BS      -> auto mGroupsPerWave (tiered formula)
#   NAME:BS:MGW  -> override mGroupsPerWave via MEGAMOE_MGW_OVERRIDE env (recorded in config)
# Same-BS runs share one build (override is runtime-only; build copied between RUN dirs).
set -e
REPO=/mnt/s00951640/ops-transformer-profs-demo-megamoe-profile
PY=/mnt/s00951640/envs/wcf_py310/bin/python
CACHE_SRC=$REPO/prof_runs/20260914_225500_global_wave/torch_ext_cache
SWEEP_ROOT=$1; shift
mkdir -p $SWEEP_ROOT
declare -A BS_BUILD   # bs -> reference build dir

for spec in "$@"; do
  IFS=':' read -r NAME BS MGW <<< "$spec"
  RUN=$SWEEP_ROOT/$NAME
  mkdir -p $RUN/raw
  # config: tokens (+ mgw override recorded in a non-"case" section so the build stays identical)
  sed "s/\"tokens\": 8192,/\"tokens\": $BS,/" \
    $REPO/megamoe_profile/configs/four_card_v4_noshared.json5 > $RUN/config.json5
  grep -q "\"tokens\": $BS," $RUN/config.json5
  if [ -n "${MGW:-}" ]; then
    python3 - "$RUN/config.json5" "$MGW" <<'PYEOF'
import sys
path, mgw = sys.argv[1], sys.argv[2]
text = open(path).read()
i = text.rfind('}')
text = text[:i] + f'  ,\n  "cost_sweep": {{"mGroupsPerWaveOverride": {mgw}}}\n' + text[i:]
open(path, 'w').write(text)
PYEOF
  fi
  cp -a $CACHE_SRC $RUN/torch_ext_cache 2>/dev/null || true

  # build once per bs, copy to sibling runs of the same bs
  if [ -z "${BS_BUILD[$BS]:-}" ]; then
    if [ ! -f "$RUN/build/lib/libmegamoe_kernel.so" ]; then
      if [ -f "$SWEEP_ROOT/bs$BS/build/lib/libmegamoe_kernel.so" ]; then
        BS_BUILD[$BS]="$SWEEP_ROOT/bs$BS/build"   # reuse an existing same-bs build
      fi
    else
      BS_BUILD[$BS]=$RUN/build
    fi
  fi
  if [ -n "${BS_BUILD[$BS]:-}" ] && [ "${BS_BUILD[$BS]}" != "$RUN/build" ]; then
    cp -r "${BS_BUILD[$BS]}" $RUN/build
    echo "BUILD_COPY $NAME (bs=$BS, shared build)"
  elif [ -n "${BS_BUILD[$BS]:-}" ]; then
    echo "BUILD_SKIP $NAME (bs=$BS, is the reference build)"
  else
    mkdir -p $RUN/build
    source /usr/local/Ascend/cann-9.1.0/set_env.sh
    export CC=gcc-11 CXX=g++-11 MAX_JOBS=2
    cmake -S $REPO/megamoe_profile -B $RUN/build \
      -DPROFILE_CONFIG=$RUN/config.json5 -DPROFILE_PYTHON=$PY \
      -DPython3_EXECUTABLE=$PY -DSOC_VERSION=ascend950pr_957c \
      -DCMAKE_BUILD_TYPE=Release > $RUN/build.log 2>&1
    cmake --build $RUN/build -j 2 >> $RUN/build.log 2>&1
    BS_BUILD[$BS]=$RUN/build
    echo "BUILD_OK $NAME (bs=$BS, fresh)"
  fi

  cat > $RUN/run_one.sh <<EOF
#!/bin/bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export CC=gcc-11 CXX=g++-11 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MAX_JOBS=4
export MEGAMOE_MGW_OVERRIDE=${MGW:-0}
export MEGAMOE_RUN_DIR=$RUN
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_WORK_PATH=$RUN/raw/device_logs
export TORCH_EXTENSIONS_DIR=$RUN/torch_ext_cache
cd $RUN
$PY -m torch.distributed.run --standalone --nproc_per_node=4 \
  $REPO/megamoe_profile/scripts/run_case.py >> $RUN/run.log 2>&1
code=\$?
echo "NPU_STAGE_EXIT=\$code $NAME"
tail -3 $RUN/run.log
exit \$code
EOF
  chmod +x $RUN/run_one.sh
  setsid nohup qrun -n s00951640-cost-$NAME -t 300 "bash $RUN/run_one.sh" \
    > $RUN/qrun.log 2>&1 < /dev/null &
  disown
  echo "SUBMITTED s00951640-cost-$NAME (bs=$BS mgw=${MGW:-auto}, timeout 300s)"
done
echo "$SWEEP_ROOT" > /tmp/opencode/last_sweep_root
