#!/usr/bin/env python3
"""预测 DAG 与实测 trace 的结构比对 (步骤 6).

用法:
  python tools/compare_trace_structure.py                 # 全部 run
  python tools/compare_trace_structure.py --run bs128     # 名字包含该串的 run
  python tools/compare_trace_structure.py --rank 0         # 只比某个 rank (缺省 0)
  python tools/compare_trace_structure.py --check          # 有结构问题则退出码 1

**先说能比什么** (详见 validation/compare 的模块说明):
  可比    波数 / 逐专家分布的形状 / 用了几个核 / 逐专家条数比是否一致
  不能直接比条数  实测的 GMM1 标记是每次 RunGmm1Generic 调用一个 (一个 (核,波,专家)
          切片一个), 模型是每个 tile 一个。两个数都对, 口径不同 —— 所以比的是**比值
          在各专家间是否一致**, 一致说明同一个结构的两种刻度。
  不可比  字节 (trace 的 args 里没有任何搬运字节字段)、buffer 生命周期 (只有等待事件
          这个影子)。

模型侧的形状从 data/<run>/config.json5 建, 不从 tiling 真值建 —— raw/ 整个被 gitignore,
六个 run 的 tiling_rank0.bin 都不在仓里, 所以 examples/*.toml 那六个场景在干净克隆上跑不起来。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import moe_cost_model as m                                            # noqa: E402
from moe_cost_model.validation import (check_graph, compare_run,      # noqa: E402
                                       read_run_config, read_trace)

#: 夹具速率 (非标定常数): 结构比对不看时长, 但 Cube 速率没有缺省值, 必须给一个。
CUBE_RATE = 2.7e7


def model_events(cfg, rank: int = 0):
    """按 run 的 config.json5 建模型侧事件图."""
    scenario = m.Scenario(
        workload=m.Workload(tokens=cfg["tokens"], topk=cfg["topk"], world=cfg["world"],
                            local_experts=cfg["local_experts"], routing=cfg["routing"],
                            seed=cfg["seed"], shared_expert_num=cfg["shared_experts"]),
        h=cfg["h"], hidden_dim=cfg["hidden_dim"], aic_num=cfg["aic_cores"],
        profile="megamoe-a8w8",
        calibration=m.Calibration(cube_mac_per_us=CUBE_RATE))
    result = m.simulate(scenario)
    return result["rank_results"][rank]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="", help="只比名字含该串的 run")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="有结构问题则退出码 1")
    ap.add_argument("--data", default=str(PROJ / "data"))
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    runs = [d for d in sorted(Path(args.data).iterdir())
            if d.is_dir() and args.run in d.name
            and list(d.glob(f"*_trace_rank{args.rank}.json"))]
    if not runs:
        print(f"没有可比的 run (data={args.data}, 过滤={args.run!r})")
        return 1

    problems = 0
    for run in runs:
        cfg = read_run_config(run)
        trace = read_trace(next(iter(run.glob(f"*_trace_rank{args.rank}.json"))))
        rank_result = model_events(cfg, args.rank)
        report = compare_run(trace, rank_result["events"], run.name)
        print(report.report())
        if trace.note:
            print(f"  注: {trace.note}")
        violations = check_graph(rank_result["events"])
        if violations:
            print(f"  结构不变量违规 {len(violations)} 条:")
            for v in violations[:5]:
                print(f"    - {v.rule}: {v.detail}")
        problems += len(report.issues()) + len(violations)
        print()
    print(f"# 共 {len(runs)} 个 run, 问题 {problems} 条")
    return 1 if (args.check and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
