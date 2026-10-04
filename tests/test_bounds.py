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
def test_known_gap_declared_traffic_is_below_the_algorithmic_minimum():
    """**已知漏账 1**: GMM2 的权重流进了时长公式却没进字节申报.

    builders/gmm2.py 只申报 a_gm (激活), 不申报 B 流 (k2 x cols 的权重);
    GMM1 两条都申报。于是模型申报的 GM->L1 字节**少于算法必搬的字节** —— 申报量
    低于算法下界在物理上不可能, 所以这是漏账, 不是口径差异。
    修好之后这条测试要翻成 declared >= algorithmic。
    """
    r = m.simulate(m.load_scenario(SCENARIO))
    rr = r["rank_results"][0]
    declared = rr["traffic_bytes"]["R0.gm_to_l1"]
    algorithmic = rr["bounds"]["gm_to_l1_bytes"]
    assert declared < algorithmic, "漏账已修? 把这条测试翻成 declared >= algorithmic"


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_known_gap_phase_pipelining_breaks_the_bandwidth_bound():
    """**已知漏账 2**: 相位流水下载入相位不占任何资源, 于是载入无限并行.

    信道模型 2026-10-03 停用后 channel_bytes 只做申报、不参与准入, 所以拆相位把
    载入从 Cube 的账上挪走却没挪到别的账上。墙钟因此低于带宽下界约 27%。
    这不是"相位流水省了 30%", 是搬运被算成免费。
    修好之后这条测试要翻成"不再穿透"。
    """
    base = m.load_scenario(SCENARIO)
    r = m.simulate(base.with_overrides(PIPE))        # 缺省只记录不抛
    rr = r["rank_results"][0]
    assert rr["bounds"]["violation"], "漏账已修? 把这条测试翻成断言无 violation"
    assert r["kernel_total_us"] < rr["bounds"]["lower_us"]


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_check_bounds_true_raises_on_the_known_violation():
    """断言开关本身要有效: check_bounds=True 时穿透下界必须抛, 不许静默."""
    base = m.load_scenario(SCENARIO)
    sc = base.with_overrides(PIPE)
    with pytest.raises(BoundViolation):
        m.simulate(sc, check_bounds=True)
