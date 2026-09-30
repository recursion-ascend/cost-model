#!/bin/bash
# p1/p2 wave-policy sweep PREPARATION (compile only, no qrun, no NPU).
#
# Usage: bash sweep_p1p2.sh <sweep_root> <base_config> NAME:P1:P2 ...
#   NAME:P1:P2  -> MEGAMOE_P1_OVERRIDE / MEGAMOE_P2_OVERRIDE for that run
#                  (0 = no override: tier p1 + p2=1, i.e. source default)
#
# - ONE shared build for all runs (same case -> same case_config.h); build dir
#   is copied (cp -a) into each run dir like sweep_cost.sh does.
# - mGroupsPerWave always comes from the formula; MEGAMOE_MGW_OVERRIDE is NOT
#   used in this experiment.
# - Does NOT submit anything. After this script finishes, submit in batches:
#     setsid nohup bash rolling_submit.sh <sweep_root> NAME... &
#   (rolling_submit keeps <= 4 of our tasks in the qrun queue; qrun itself is
#    FIFO-serial on hardware, so timings stay clean and the queue is not hogged)
set -e
REPO=/mnt/s00951640/ops-transformer-profs-demo-megamoe-profile
PY=/mnt/s00951640/envs/wcf_py310/bin/python
SCRIPTS=$REPO/megamoe_profile/scripts
CACHE_SRC=$REPO/prof_runs/20260914_225500_global_wave/torch_ext_cache
# qrun 任务在独立 cwd 执行，所有路径必须绝对化
SWEEP_ROOT=$(readlink -f "$1")
BASE=$(readlink -f "$2")
shift 2
mkdir -p $SWEEP_ROOT
BUILD_SHARED=$SWEEP_ROOT/build_shared
NAMES=()

for spec in "$@"; do
  IFS=':' read -r NAME P1 P2 <<< "$spec"
  RUN=$SWEEP_ROOT/$NAME
  NAMES+=("$NAME")
  mkdir -p $RUN/raw
  # config: identical case for every run; overrides recorded for bookkeeping only
  # (behavior comes from env vars in run_one.sh, so the shared build stays valid)
  cp "$BASE" $RUN/config.json5
  python3 - "$RUN/config.json5" "$P1" "$P2" <<'PYEOF'
import sys
path, p1, p2 = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read()
i = text.rfind('}')
text = text[:i] + f'  ,\n  "cost_sweep": {{"p1Override": {p1}, "p2Override": {p2}}}\n' + text[i:]
open(path, 'w').write(text)
PYEOF
  cp -a $CACHE_SRC $RUN/torch_ext_cache 2>/dev/null || true

  cat > $RUN/run_one.sh <<EOF
#!/bin/bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export CC=gcc-11 CXX=g++-11 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MAX_JOBS=4
export MEGAMOE_P1_OVERRIDE=${P1}
export MEGAMOE_P2_OVERRIDE=${P2}
unset MEGAMOE_MGW_OVERRIDE
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
done

# ---- single shared build (CPU only; no qrun) ----
if [ ! -f "$BUILD_SHARED/lib/libmegamoe_kernel.so" ]; then
  FIRST=$SWEEP_ROOT/${NAMES[0]}
  mkdir -p $BUILD_SHARED
  source /usr/local/Ascend/cann-9.1.0/set_env.sh
  export CC=gcc-11 CXX=g++-11 MAX_JOBS=2
  cmake -S $REPO/megamoe_profile -B $BUILD_SHARED \
    -DPROFILE_CONFIG=$FIRST/config.json5 -DPROFILE_PYTHON=$PY \
    -DPython3_EXECUTABLE=$PY -DSOC_VERSION=ascend950pr_957c \
    -DCMAKE_BUILD_TYPE=Release > $SWEEP_ROOT/build.log 2>&1
  cmake --build $BUILD_SHARED -j 2 >> $SWEEP_ROOT/build.log 2>&1
  echo "BUILD_OK shared ($BUILD_SHARED)"
else
  echo "BUILD_REUSE $BUILD_SHARED"
fi

for NAME in "${NAMES[@]}"; do
  RUN=$SWEEP_ROOT/$NAME
  rm -rf $RUN/build
  cp -a $BUILD_SHARED $RUN/build
done

echo ""
echo "Prepared ${#NAMES[@]} runs under $SWEEP_ROOT"
echo "Submit (batched, <=4 in queue):"
echo "  setsid nohup bash $SCRIPTS/rolling_submit.sh $SWEEP_ROOT ${NAMES[*]} > /dev/null 2>&1 &"
echo "$SWEEP_ROOT" > /tmp/opencode/last_sweep_root
