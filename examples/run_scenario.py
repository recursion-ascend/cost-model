"""场景文件 → 执行时间 → 改旋钮对比.

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
    busy = result["rank_results"][result["slowest_rank"]]["stage_busy_us"]
    top = sorted(busy.items(), key=lambda kv: -kv[1])[:5]      # 忙碌时长最大的 5 个 stage
    stages = "  ".join(f"{s}={v:.0f}" for s, v in top)
    # 单位写 us 不写 µ: Windows 默认控制台 (GBK) 无法编码 µ
    print(f"{label:16s} {total:10.1f} us  ({total - base_total:+9.1f})  busy: {stages}")
