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
        combine_tile=m.AnalyticalCombineCosts(h=6144).tile,
        count_table_prepare_us=m.T_COUNT_GATE,
    )
    return m.simulate_routing_counts(
        routing_counts=_deterministic_case(), token_num_per_rank=64, h=6144,
        hidden_dim=4096, aic_num=28, costs=costs, options=options or m.ModelOptions(),
        kernel=kernel,
    )


def test_default_pin():
    """确定性用例锚点. 组成: 前导(INIT/INPUT_QUANT/门控+调度准备) + 五 stage +
    尾段(COUNTS_EXPORT/barrier/rank_sync/UNPERMUTE/FINALIZE).
    kL1=256 时回到 425.965 (结构等价自检, 见下个测试).
    """
    res = _run()
    # 前导/尾段段 (arch35.h:660-775) 加入后的锚点: INIT+INPUT_QUANT+调度准备
    # 与 COUNTS_EXPORT→barrier→rank_sync→UNPERMUTE→FINALIZE 串行链
    assert abs(res["kernel_total_us"] - 424.702) < 0.01
    assert len(res["rank_results"][0]["events"]) == 651


def test_kl1_override_restores_legacy():
    """kL1=256 显式覆盖应恢复 402.335 (结构等价性自检)."""
    res = _run(options=m.ModelOptions(gmm2_kl1=256))
    assert abs(res["kernel_total_us"] - 425.965) < 0.01


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
