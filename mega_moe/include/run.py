#!/usr/bin/env python3
"""Compile and run one configured MegaMoE capture, then export Chrome trace JSON."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE / 'scripts'))
from config_io import load_config


def call(command, root, env, cwd=None):
    """Run foreground children; retain compiler/application errors in one log."""
    print(shlex.join(map(str, command)), flush=True)
    with (root / 'run.log').open('ab') as log:
        log.write(('\n$ ' + shlex.join(map(str, command)) + '\n').encode())
        log.flush()
        subprocess.run(list(map(str, command)), env=env, cwd=cwd,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in args.name)
    root = SOURCE.parent / 'prof_runs' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f_') + name)
    root.mkdir(parents=True)
    for directory in ('build', 'raw'):
        (root / directory).mkdir()
    shutil.copy2(args.config, root / 'config.json5')
    print(f'RUN_DIR={root}', flush=True)
    env = os.environ.copy()
    if cfg['runtime'].get('cann_env'):
        with (root / 'run.log').open('ab') as log:
            result = subprocess.check_output(
                ['bash', '-c', 'source "$1" >&2 || exit $?; env -0', 'bash',
                 cfg['runtime']['cann_env']], env=env, stderr=log)
        env.update(item.decode().split('=', 1) for item in result.split(b'\0') if b'=' in item)
    env.update(MEGAMOE_RUN_DIR=str(root), PYTHONUNBUFFERED='1',
               ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, cfg['device']['ids'])),
               ASCEND_WORK_PATH=str(root / 'raw' / 'device_logs'),
               OMP_NUM_THREADS=str(cfg['runtime']['omp_threads']),
               MAX_JOBS=str(cfg['build']['jobs']))
    python = sys.executable
    try:
        call(['cmake', '-S', SOURCE, '-B', root / 'build',
              f'-DPROFILE_CONFIG={root}/config.json5', f'-DPROFILE_PYTHON={python}',
              f'-DPython3_EXECUTABLE={python}', f'-DSOC_VERSION={cfg["build"]["soc"]}',
              f'-DCMAKE_BUILD_TYPE={cfg["build"]["type"]}'], root, env)
        call(['cmake', '--build', root / 'build', '-j', cfg['build']['jobs']], root, env)
        call([python, '-m', 'torch.distributed.run', '--standalone',
              f'--nproc_per_node={cfg["case"]["ep"]}', SOURCE / 'scripts/run_case.py'], root, env)
        device = cfg['device']
        call([python, SOURCE / 'scripts/parse_prof.py',
              '--n-cores', device['aic_cores'] + device['aiv_cores'],
              '--aiv-cores', device['aiv_cores'],
              '--event-names-from', root / 'build/generated/profile_events_generated.h',
              '--pair-map', root / 'build/generated/pair_map.json',
              *[root / f'raw/prof_rank{rank}.bin' for rank in range(cfg['case']['ep'])]], root, env, root / 'raw')
        call([python, SOURCE / 'scripts/to_chrome_trace.py', root / 'raw/prof_paired.csv',
              '--freq', cfg['profiling']['timestamp_frequency_mhz'],
              '--aic-cores', device['aic_cores'], root], root, env)
    except subprocess.CalledProcessError as error:
        print(f'Failed (exit {error.returncode}); see {root}/run.log', file=sys.stderr)
        return 1
    print(f'TRACE={root}/{root.name}_trace_all_ranks.json', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
