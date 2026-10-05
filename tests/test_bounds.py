"""下界: 墙钟在物理上不可能低于 算法事实 + 硬件规格 推出的那条线.

这组测试有两个职责:
  1. 钉住下界本身只用算法事实 (换编排不变, 换形状才变);
  2. **把两处已知漏账钉成测试** —— 它们现在确实会穿透带宽下界, 修好之前这两条
     测试断言"能被检出", 修好之后要改成断言"不再穿透"。
"""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import moe_cost_model as m
from moe_cost_model.analysis.bounds import (TOL, BoundViolation, Bounds,
                                            bandwidth_bound_us,
                                            check_wall_clock,
                                            compute_bound_us,
                                            dependency_bound_us,
                                            workload_facts)

SCENARIO = Path("examples/scenario_basic.toml")
PIPE = {"options.pipeline": {"queues": {"mte_aic": 2, "cube": 2}}}


def _shape(h=6144, hidden=4096, rows=(100, 200, 0), half=2):
    return SimpleNamespace(h=h, hidden_dim=hidden, expert_tokens=rows,
                           kernel=SimpleNamespace(activation_n_half=half))


# ------------------------------------------------------- 算法事实

def test_mac_counts_match_the_matmul_definition():
    """GMM1 = A(m x h) x B(h x hidden); GMM2 = A(m x I) x B(I x h). 不含 tile 口径."""
    f = workload_facts(_shape(rows=(100, 200, 0)))
    m_total, h, hidden, inter = 300, 6144, 4096, 2048
    assert f.gmm1_mac == m_total * h * hidden
    assert f.gmm2_mac == m_total * inter * h
    assert f.total_mac == f.gmm1_mac + f.gmm2_mac
    assert f.rows_total == m_total
    assert f.experts_active == 2          # 0 行的专家不算


def test_weight_bytes_counted_once_per_active_expert():
    """权重每专家至少搬一次 —— 这是下界; L2 复用只能让实际更接近它, 不能更少."""
    f = workload_facts(_shape(rows=(100, 200, 0)))
    assert f.gmm1_b_bytes == 2 * 6144 * 4096
    assert f.gmm2_b_bytes == 2 * 2048 * 6144
    # 行数为 0 的专家不搬权重
    assert workload_facts(_shape(rows=(100, 0, 0))).gmm1_b_bytes == 6144 * 4096


def test_facts_do_not_depend_on_orchestration():
    """同一形状, 换 activation_n_half 之外的任何编排都不该改算法事实."""
    a = workload_facts(_shape())
    b = workload_facts(_shape())
    assert a == b


def test_materialised_link_adds_gmm2_a_bytes():
    """ACT 物化到 GM 再被 GMM2 读回是编排选择, 它确实多搬字节 —— 要算进去."""
    on = workload_facts(_shape(), gmm2_a_from_gm=True)
    off = workload_facts(_shape(), gmm2_a_from_gm=False)
    assert on.gmm2_a_bytes > 0 and off.gmm2_a_bytes == 0
    assert on.gm_to_l1_bytes > off.gm_to_l1_bytes


# ------------------------------------------------------- 三个下界

def test_compute_bound_is_mac_over_cores_times_rate():
    f = workload_facts(_shape())
    got = compute_bound_us(f, cube_mac_per_us=1e7, active_cores=28)
    assert got == pytest.approx(f.total_mac / (1e7 * 28))
    # 速率未标定 (<=0) 时给 0, 不参与取最大
    assert compute_bound_us(f, cube_mac_per_us=0, active_cores=28) == 0.0


def test_bandwidth_bound_takes_the_smaller_of_two_hardware_limits():
    f = workload_facts(_shape())
    # 聚合带宽更紧
    us, rate, who = bandwidth_bound_us(f, bw_per_core_bytes_per_us=51900,
                                       active_cores=28,
                                       aggregate_bytes_per_us=1.0e6)
    assert who == "aggregate" and rate == 1.0e6
    assert us == pytest.approx(f.gm_to_l1_bytes / 1.0e6)
    # 每核合计更紧
    us2, rate2, who2 = bandwidth_bound_us(f, bw_per_core_bytes_per_us=10_000,
                                          active_cores=28,
                                          aggregate_bytes_per_us=1.0e9)
    assert who2 == "per_core" and rate2 == 10_000 * 28


def test_dependency_bound_sums_the_chain():
    assert dependency_bound_us({"dispatch": 1.0, "gmm1": 2.0, "activation": 0.5,
                                "gmm2": 3.0, "combine": 1.5}) == pytest.approx(8.0)


def test_binding_names_the_largest_bound():
    f = workload_facts(_shape())
    b = Bounds(compute_us=100.0, bandwidth_us=10.0, dependency_us=1.0, facts=f)
    assert b.binding == "compute" and b.lower_us == 100.0
    b2 = Bounds(compute_us=10.0, bandwidth_us=100.0, dependency_us=1.0, facts=f)
    assert b2.binding == "bandwidth"


def test_check_wall_clock_rejects_below_bound_and_accepts_above():
    f = workload_facts(_shape())
    b = Bounds(compute_us=100.0, bandwidth_us=0.0, dependency_us=0.0, facts=f)
    check_wall_clock(b, 100.0)              # 相等放过
    check_wall_clock(b, 120.0)
    with pytest.raises(BoundViolation, match="低于物理下界"):
        check_wall_clock(b, 80.0)


# ------------------------------- 端到端: 下界进结果, 且两处漏账被检出

@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_bounds_are_attached_to_every_rank():
    r = m.simulate(m.load_scenario(SCENARIO))
    for rr in r["rank_results"].values():
        b = rr["bounds"]
        for k in ("compute_us", "bandwidth_us", "dependency_us", "lower_us",
                  "binding", "total_mac", "gm_to_l1_bytes"):
            assert k in b
        assert b["lower_us"] == max(b["compute_us"], b["bandwidth_us"],
                                    b["dependency_us"])


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_declared_traffic_is_at_least_the_algorithmic_minimum():
    """申报的 GM->L1 字节不得少于算法必搬的字节.

    2026-10-05 修掉的漏账: builders/gmm2.py 只申报 a_gm (激活), 不申报 B 流
    (k2 x cols 的权重); GMM1 一直两股都申报。当时 scenario_basic 上申报 1660.9MB <
    算法必搬 2420.1MB (差 759.2MB ≈ GMM2 权重 805.3MB)。申报量低于算法下界在物理上
    不可能 —— 那是漏账, 不是口径差异。
    申报量**可以高于**下界 (模型按 tile 重复读权重, 多个 m-group 各读一次)。
    """
    r = m.simulate(m.load_scenario(SCENARIO))
    rr = r["rank_results"][0]
    declared = rr["traffic_bytes"]["R0.gm_to_l1"]
    algorithmic = rr["bounds"]["gm_to_l1_bytes"]
    assert declared >= algorithmic, (
        f"申报 {declared / 1e6:.1f}MB 少于算法必搬 {algorithmic / 1e6:.1f}MB")


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_phase_pipelining_respects_the_bandwidth_bound():
    """相位流水不得穿透带宽下界: 载入必须占住本核那条 MTE2 管道.

    2026-10-05 修掉的漏账: .ld 相位 resources=() 不占任何资源, GMM2 的载入又整段裹在
    AIC 事件里 —— 等于每个核有两条载入管道, 聚合载入带宽翻倍。当时墙钟 1221.80us
    低于带宽下界 1665.37us 达 26.6%, 被当成"相位流水省了 30.24%"。
    修法是硬件事实: 一个 AI Core 只有一条 MTE2, 所以 GMM1 与 GMM2 的载入都占
    MTE2:c{core}; 双缓冲 (queues.mte_aic = L1 槽数) 只决定能提前多少发起, 不决定
    能同时搬几笔。修后真实收益是 1751.48 -> 1748.18 (-0.19%), 不是 -30%。
    """
    base = m.load_scenario(SCENARIO)
    r = m.simulate(base.with_overrides(PIPE))
    rr = r["rank_results"][0]
    assert rr["bounds"]["violation"] is None
    assert r["kernel_total_us"] >= rr["bounds"]["lower_us"]


def test_load_phase_holds_the_cores_mte2_pipe():
    """硬件事实: 一个 AI Core 一条 MTE2, 所以同一个核上两笔载入不得重叠.

    不占的话载入并发只受 L1 槽数限制 (28 核 x d 笔同时满带宽), 聚合载入带宽会超过
    核数 x BW_L1_GM 这条硬件规格。
    写成计数信号量 (容量 1) 而不是独占资源, 是为了走 late-bind 的 "c*" 占位重映射 ——
    独占资源会把 .ld 钉在建图时的占位核号上, 与它所在相位组绑定的核冲突。
    """
    base = m.load_scenario(SCENARIO)
    r = m.simulate(base.with_overrides(PIPE))
    lds = [e for e in r["rank_results"][0]["events"] if e.name.endswith(".ld")]
    assert lds, "开了相位流水却没有载入相位事件"
    holding = [e for e in lds
               if any("MTE2:" in t for t, _ in getattr(e, "acquires", ()))]
    assert holding, "载入相位没有占住 MTE2 管道"
    # 同一个核上两笔载入不得重叠 —— 容量 1 的直接推论, 这里直接验时间线。
    # meta["core"] 由引擎改写成**实际绑定**的核号 (晚绑定下建图时的占位号不作数)。
    by_core = {}
    for e in holding:
        by_core.setdefault(e.meta.get("core"), []).append((e.start_us, e.end_us))
    for core, spans in by_core.items():
        spans.sort()
        for (a0, a1), (b0, _) in zip(spans, spans[1:]):
            assert b0 >= a1 - 1e-9, f"核 {core} 上两笔载入重叠: {a1} > {b0}"


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_check_bounds_raises_by_default_and_records_when_switched_off():
    """断言开关本身要有效, 且缺省是**抛**而不是静默.

    用一个带宽低得荒谬的平台把带宽下界顶到墙钟之上来触发 —— 不依赖任何现存漏账,
    所以漏账修好之后这条测试照样有效。
    """
    tiny = m.PlatformSpec(name="tiny-bw", source="test fixture",
                          hbm_bytes_per_us=1.0e3, fabric_bytes_per_us=1.0e3)
    sc = m.load_scenario(SCENARIO)
    with pytest.raises(BoundViolation, match="低于物理下界"):
        m.simulate(sc, platform=tiny)
    r = m.simulate(sc, platform=tiny, check_bounds=False)
    rr = r["rank_results"][0]
    assert rr["bounds"]["violation"], "降级之后也要把诊断记进结果, 不能悄悄丢掉"
    assert rr["bounds"]["bandwidth_limited_by"] == "aggregate"
