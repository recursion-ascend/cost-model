#!/usr/bin/env python3
"""扫一遍 stage 边的编排选择, 看每个编排选择的代价.

运行: python examples/run_design_space.py

读表的方法:
  Δ            负数是变快。这是收益。
  关键路径变化  收益落在关键路径的哪个 stage 上 —— 说不出来的收益不敢用。
  总忙碌变化    "(工作量不变)" = 省的不是计算量, 是等待/排布。
  最大等待      改完之后卡在什么上: capacity (信号量/容量) / dep (依赖) / res (核被占)。
  访存量差      少搬/多搬多少字节。片上不物化省的就是这一列。
  不变量        "违反" = 这个方案下有就绪的活却有核空闲, 该行时长偏慢, 收益不可比。
                "HBM x%" = 这个方案需要的聚合带宽占该平台规格的多少; 超 100% 就是
                物理上不可能 (模型不建带宽争用, 只能事后核对)。

Cube 速率取规格峰值 (config.platform.cube_mac_per_us("fp8") = 1.35e7 MAC/µs,
A8W8 主路径), 不是反解值 —— 让模型自己判断谁绑定。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import moe_cost_model as m                                       # noqa: E402

L = m.StageLink
OPT = m.ModelOptions

#: gmm1->activation 的同核是硬件强制 (Fixpipe 直给配对 AIV0), 不是编排选择;
#: depth 是 UB 里能同时存几块 GMM1 结果。
def hw(depth=1):
    return L("gmm1", "activation", location="onchip", depth=depth,
             colocated_by_hardware=True)


def edges(depth=1, **act_to_gmm2):
    return (hw(depth), L("activation", "gmm2", **act_to_gmm2))


def main() -> int:
    WORLD, LOCAL, PER = 5, 3, 64        # 5 卡, 每卡 3 个本地专家, 每格 64 行
    rc = [[[0 if s == d else PER for s in range(WORLD)] for _ in range(LOCAL)]
          for d in range(WORLD)]
    tokens = sum(rc[d][e][1] for d in range(WORLD) for e in range(LOCAL)) // 6

    AIC = 28                       # 单卡真实可用核数

    def runner(platform):
        def run(options):
            return m.simulate_routing_counts(
                routing_counts=rc, token_num_per_rank=tokens, h=5120, hidden_dim=9216,
                aic_num=AIC, topk=6, p1_override=1, p2_override=1,
                costs=m.build_analytical_costs(
                    h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(),
                    cube_mac_per_us=m.cube_mac_per_us("fp8"),
                    platform=platform, active_cores=AIC),
                options=options)
        return run

    points = {
        "基线 (最少假设)": OPT(),
        "GMM2 首块先开工": OPT(links=edges(readiness="first_chunk")),
        "GMM2 均分 4 段就绪": OPT(links=edges(readiness=4)),
        "GMM2 逐 K 块就绪": OPT(links=edges(readiness="per_chunk")),
        "UB 深度 2 (交织路径)": OPT(links=edges(2)),
        "UB 不设限 (上界)": OPT(links=edges(0)),
        "ACT 不物化 (留片上)": OPT(links=edges(location="onchip")),
        "波间全核对齐": OPT(barriers=("wave",)),
        "静态分核": OPT(late_bind_pools=()),
        # 事件粒度 (缺口 12): 一个事件覆盖几份工作 —— 同步点密度 <-> 并行度的交换。
        # 粗粒度不是收益开关: 项数少于核数就有核闲着, 这一列会直接看见。
        "GMM1 攒 2 个 tile": OPT(granularity={"gmm1": 2}),
        "GMM2 攒 2 个 tile": OPT(granularity={"gmm2": 2}),
        "combine 攒 2 个 tile": OPT(granularity={"combine": 2}),
        "combine 逐专家 (整片)": OPT(granularity={"combine": 0}),
        # ACT 粒度要"喂它的 GMM1 tile 同核且 n 相邻" (Fixpipe 强制同核), 这个夹具用
        # 轮转分核, 相邻 n-tile 落在不同核上 -> **这一行与"UB 深度 2"完全相同, 是空操作**。
        # 留在表里是为了让这件事看得见: 粒度参数会静默无效, 要配按块分核才生效。
        "ACT 攒 2 个 (轮转下空操作)": OPT(granularity={"activation": 2},
                                          links=edges(2)),
        "那份实现 (MEGAMOE_A8W8)": m.MEGAMOE_A8W8.options,
    }
    for platform in (m.ASCEND_950PR, m.ASCEND_950DT):
        rows = m.design_space(runner(platform), points, platform=platform)
        print(f"== {platform.name}  (聚合 HBM {platform.hbm_bytes_per_us / 1e6:.1f} TB/s, "
              f"{AIC} 核)")
        print(m.format_design_space(rows))
        ok = [r for r in rows if r["invariant_ok"] and r["bandwidth_ok"]]
        best = min(ok, key=lambda r: r["total_us"])
        print(f"   不变量成立的最快方案: {best['name']}  {best['total_us']:.2f} us "
              f"({best['delta_pct']:+.1f}%), 需要聚合带宽 "
              f"{best['gm_bw_needed'] / 1e6:.2f} TB/s = 规格的 {best['hbm_pct']:.0f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
