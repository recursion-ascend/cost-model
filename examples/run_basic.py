"""最小示例: 路由计数 → 执行时间.

运行: python3 examples/run_basic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_cost_model import (
    DispatchMechanisticLatency, InstancePolicy, KernelConfig,
    ModelOptions, build_analytical_costs, simulate_routing_counts,
)

WORLD, LOCAL = 4, 64            # 4 rank, 每 rank 64 专家
TOKENS = 64                     # 每 rank token 数
H, HIDDEN, AIC = 6144, 4096, 28

# 均匀路由: 每个专家从每个源 rank 收同等行数
per = TOKENS * 8 // WORLD // LOCAL
C = [[[per] * WORLD for _ in range(LOCAL)] for _ in range(WORLD)]

kernel = KernelConfig(tile_m=256, tile_n=256)
# Cube 计算速率 (MAC/µs) 没有缺省值, 必须显式给. 这里是示例值, 不是标定常数:
# 换成目标平台的实测速率再看结果.
CUBE_MAC_PER_US = 2.7e7
costs = build_analytical_costs(h=H, kernel=kernel,
                               dispatch_mechanistic=DispatchMechanisticLatency(),
                               cube_mac_per_us=CUBE_MAC_PER_US)

res = simulate_routing_counts(
    routing_counts=C, token_num_per_rank=TOKENS, h=H, hidden_dim=HIDDEN,
    aic_num=AIC, costs=costs, topk=8,
    kernel=kernel,
    policy=InstancePolicy(), p1_override=2, p2_override=1,
)
# 单位写 us 不写 µ: Windows 默认控制台 (GBK) 无法编码 µ
print(f"kernel 总时长: {res['kernel_total_us']:.1f} us (最慢 rank {res['slowest_rank']})")
print(f"事件数/rank: {len(res['rank_results'][0]['events'])}")
