"""bs=72 / 8 卡 / topk=6 均匀路由: 扫 Cube 速率, 打印 DAG 调度器排出的时间线.

运行: python examples/run_bs72_8rank.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_cost_model import load_scenario, simulate

HERE = Path(__file__).parent
base = load_scenario(HERE / "bs72_8rank.toml")

# 路由矩阵另存一份, 便于检查或拿去别处用
counts = base.workload.routing_counts()
(HERE / "bs72_8rank_routing.json").write_text(
    json.dumps({"layout": "routing_counts[dst_rank][local_expert][src_rank] = rows",
                "routing_counts": counts}, indent=1), encoding="utf-8")
print(f"路由矩阵: {len(counts)} rank x {len(counts[0])} 专家 x {len(counts[0][0])} 源, "
      f"每格 {counts[0][0][0]} 行, 每专家 {sum(counts[0][0])} 行")

STAGES = ("dispatch_call", "dispatch", "gmm1", "activation", "gmm2", "combine", "epilogue")
# 单位 us (不写 µ: Windows 默认控制台 GBK 无法编码)
for rate in (1.0e7, 2.7e7, 5.0e7):
    res = simulate(base.with_overrides({"calibration.cube_mac_per_us": rate}))
    rank = res["rank_results"][res["slowest_rank"]]
    print(f"\nR_cube = {rate:.1e}  执行时间 (到最后一个 COMBINE 结束) = "
          f"{res['kernel_total_us']:.3f} us, 含尾段 = {res['kernel_dag_end_us']:.3f} us  "
          f"(最慢 rank {res['slowest_rank']}, {rank['wave_count']} 波, "
          f"{len(rank['events'])} 事件)")
    print(f"  {'stage':14s} {'首个开始':>9s} {'最后结束':>9s} {'事件数':>6s} {'单事件时长':>16s}")
    for stage in STAGES:
        durs = [e.end_us - e.start_us for e in rank["events"] if e.meta.get("stage") == stage]
        if not durs:
            continue
        one = f"{min(durs):.3f}" if max(durs) - min(durs) < 1e-9 else \
            f"{min(durs):.3f}~{max(durs):.3f}"
        print(f"  {stage:14s} {rank['stage_first_start_us'][stage]:9.3f} "
              f"{rank['stage_last_end_us'][stage]:9.3f} {len(durs):6d} {one:>16s}")
    print("  关键路径:")
    for step in rank["critical_path"]:
        print(f"    {step['start_us']:8.3f} -> {step['end_us']:8.3f}  "
              f"{step['stage']:14s} {step['name']}")
