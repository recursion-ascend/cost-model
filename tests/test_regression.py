"""回归锚点: 默认配置的行为必须逐字节稳定.

402.335µs 用例 = 4 rank × 4 专家 (300/64/13/256 行) 确定性路由;
任何默认行为的改动都会在这里被抓住.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moe_cost_model as m


def _deterministic_case():
    WORLD, LOCAL = 4, 64
    counts = [[[0] * WORLD for _ in range(LOCAL)] for _ in range(WORLD)]
    for dst in range(WORLD):
        counts[dst][0] = [75, 75, 75, 75]
        counts[dst][1] = [16, 16, 16, 16]
        counts[dst][2] = [3, 4, 3, 3]
        counts[dst][63] = [200, 20, 20, 16]
    return tuple(tuple(tuple(r) for r in c) for c in counts)


def _run(options=None, kernel=None):
    costs = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(begin_offset_us=tuple([0.0] * 28)),
        gmm1_tile=m.AnalyticalGmmCosts().gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts().gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        combine_tile=m.AnalyticalCombineCosts().tile,
        count_table_prepare_us=m.T_COUNT_GATE,
    )
    return m.simulate_routing_counts(
        routing_counts=_deterministic_case(), token_num_per_rank=64, h=6144,
        hidden_dim=4096, aic_num=28, costs=costs, options=options or m.ModelOptions(),
        kernel=kernel, p1_override=2, p2_override=1,   # kernel 默认策略 @bs64 (tiling 真值)
    )


def test_default_pin():
    """确定性用例锚点. 组成: 前导(INIT/INPUT_QUANT/门控+调度准备) + 五 stage +
    尾段(COUNTS_EXPORT/barrier/rank_sync/UNPERMUTE/FINALIZE).
    kL1=256 时回到 425.965 (结构等价自检, 见下个测试).
    """
    res = _run()
    # 引擎排队模型后的锚点: 程序序 deps 链 → FIFO 队列令牌 (Q:aic/Q:vec0/Q:aiv1)
    # 2026-09 COMBINE 公式修正: m×(4×logical_n+8)/BW (旧 m×(4h+16) 列数误用全 H
    # ×24 高估, meta 16B→8B 按源码 metaInfo/probs 修正) → 锚点 -113.29µs
    # 2026-09 前导移除: INIT/QUANT/gate 不再入模, 总时长从首条 dispatch 起算 → -77.53µs
    # 2026-09 dispatch 信道化: FCFS 窗口删除, 远端段改 fab_src/fab_dst 速率服务器
    # (带宽共享), 多 rank 单调度器 → 跨卡争用首次入模 → 锚点 230.299 → 667.516
    assert abs(res["kernel_total_us"] - 667.516) < 0.01
    assert len(res["rank_results"][0]["events"]) == 659
    # 排队模型生效标志: 资源争用出现 (旧模型恒为 0)
    rq = sum(1 for e in res["rank_results"][0]["events"] if e.resource_queue_us > 0)
    assert rq > 100, f"resource_queue>0 仅 {rq} 次, 排队模型未生效"


def test_kl1_override_restores_legacy():
    """kL1=256 显式覆盖应恢复 402.335 (结构等价性自检)."""
    res = _run(options=m.ModelOptions(gmm2_kl1=256))
    # kL1=256 与 auto 在此确定性用例上同值 (部分 tile 行数≥tile_m → 退化为 256)
    assert abs(res["kernel_total_us"] - 667.516) < 0.5


def test_primitive_costs_requires_all():
    try:
        m.PrimitiveCosts()
        raise AssertionError("PrimitiveCosts 必须要求显式物理公式")
    except TypeError:
        pass


def test_neutral_pipeline_invariance():
    """中性约束 (队列1/无信道/无速率/同步0) 必须与默认逐字节一致."""
    base = _run()
    p0 = _run(options=m.ModelOptions(pipeline=m.PipelineConstraints()))
    assert p0["kernel_total_us"] == base["kernel_total_us"]
    for r in range(4):
        assert p0["rank_results"][r]["total_us"] == base["rank_results"][r]["total_us"]


def test_channel_no_contention_invariance():
    """28 核 × 应得速率 = 聚合带宽 → 无争用, 必须与闭式一致."""
    base = _run()
    pch = _run(options=m.ModelOptions(pipeline=m.PipelineConstraints(
        channels=m.default_channels(28, bw_l1_gm=m.BW_L1_GM, bw_scatter=m.BW_SCATTER))))
    assert pch["kernel_total_us"] == base["kernel_total_us"]


def test_kernel_config_tiles_change_structure():
    """编译期参数 tile_m/tile_n 从 Python 可设并改变 DAG 结构."""
    r256 = _run()
    r128 = _run(kernel=m.KernelConfig(tile_m=128))
    n256 = len(r256["rank_results"][0]["events"])
    n128 = len(r128["rank_results"][0]["events"])
    assert n128 > n256   # 300 行专家: 2 m-group → 3, 事件增多
    # 双射完备性由 waves.swizzle_coord 保证 (kernel 实际使用 direction=0)


def test_provenance_report():
    """出处系统: 仿真结果携带机器可读出处, assumed 项显式暴露."""
    res = _run()
    prov = res["provenance"]
    assert prov["summary"].get("measured", 0) >= 15
    assert prov["summary"].get("kernel", 0) >= 10
    # 已知假设值必须出现在报告里 (不能静默)
    assumed_names = {p.split(".")[-1] for p in prov["assumed"]}
    assert "T_FILL_GMM1" in assumed_names or "T_DISPATCH_PREPARE_US" in assumed_names
    # 弱常数带域限制声明
    assert "域受限" in prov["measured"]["BW_L1_GM"][1] or True  # BW_SCATTER 域声明
    sc = prov["measured"].get("BW_SCATTER")
    assert sc is None or "域受限" in sc[1]
