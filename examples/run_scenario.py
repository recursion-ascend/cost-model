"""场景文件 → 执行时间 (到最后一个 COMBINE 结束) → 改参数对比.

运行: python examples/run_scenario.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_cost_model import load_scenario, simulate

base = load_scenario(Path(__file__).with_name("scenario_basic.toml"))

variants = {
    "基线": {},
    "GMM2 滞后 1 波": {"policy.gmm2_lag_waves": 1},
    "tile_n = 128": {"kernel.tile_n": 128},
    "均衡打包": {"wave_packing": "balanced_waves"},
    "Layered 编排": {"kernel.topo_urma": True},
}

base_total = None
for label, overrides in variants.items():
    result = simulate(base.with_overrides(overrides))
    total = result["kernel_total_us"]
    if base_total is None:
        base_total = total
    # 各 stage 最后一个事件的结束时刻 (调度器排出的时刻, 不是各核忙碌时长之和)
    ends = result["rank_results"][result["slowest_rank"]]["stage_last_end_us"]
    stages = "  ".join(f"{s}={ends[s]:.1f}"
                       for s in ("gmm1", "activation", "gmm2", "combine") if s in ends)
    # 单位写 us 不写 µ: Windows 默认控制台 (GBK) 无法编码 µ
    print(f"{label:16s} {total:10.1f} us  ({total - base_total:+9.1f})  结束时刻: {stages}")
