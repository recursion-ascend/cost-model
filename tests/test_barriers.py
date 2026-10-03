"""C3: 全核栅栏原语 —— 让"融合 vs 分段式执行"能被算出来."""
import pytest

import moe_cost_model as m
from linkutil import links


def _run(hd=9216, local=3, barriers=(), depth=1):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hd, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(barriers=barriers, links=links(depth)),
        )["rank_results"][0]


def test_default_has_no_barrier_and_one_drain_node():
    """缺省不加栅栏; 排空节点只剩一个 (C3 前是 28 核 x 3 引擎 = 84 个)."""
    evs = _run()["events"]
    assert not [e for e in evs if e.meta.get("stage") == "barrier"]
    drains = [e for e in evs if e.meta.get("stage") == "moe_stage_done"]
    assert len(drains) == 1
    assert drains[0].resources == ()        # 排空语义不需要占核


def test_drain_barrier_waits_for_every_moe_event():
    """扇入必须挂**全部**成员: 建图序不等于时间序, 只挂最后建的那个会漏."""
    rr = _run()
    evs = rr["events"]
    (drain,) = [e for e in evs if e.meta.get("stage") == "moe_stage_done"]
    moe = [e for e in evs if e.meta.get("stage") in
           ("gmm1", "gmm2", "activation", "dispatch", "dispatch_call", "combine")]
    assert drain.end_us >= max(e.end_us for e in moe) - 1e-9


def test_wave_barrier_orders_waves_strictly():
    """波间栅栏: 下一波的任何事件都不得早于上一波的最后一个事件."""
    evs = _run(barriers=("wave",))["events"]
    bars = [e for e in evs if e.meta.get("kind") == "wave"]
    assert bars
    for b in bars:
        w = b.meta["wave"]
        prev_end = max(e.end_us for e in evs if e.meta.get("wave") == w - 1
                       and e.end_us > e.start_us)
        cur_start = min(e.start_us for e in evs if e.meta.get("wave") == w
                        and e.end_us > e.start_us)
        assert cur_start >= prev_end - 1e-9


def _run_mgw_depth(mgw, depth, barriers=("stage",), local=3, hd=9216, late=()):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hd, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(barriers=barriers, m_groups_per_wave=mgw,
                               late_bind_pools=late, links=links(depth)),
        )["rank_results"][0]


def test_stage_barrier_vs_ub_depth_is_conditional_and_per_wave():
    """stage 栅栏与 UB 槽的冲突**范围是一个波内**, 且**有条件**.

    判据: 某个核在同一波里的 GMM1 tile 数 > UB 深度。那时
      第 depth+1 个 GMM1 等 ACT 还槽 -> 那个 ACT 等 barrier.w{w}.activation
      -> 那道栅栏等该波全部 GMM1, 包括第 depth+1 个   => 成环

    实测 (9216/3, 28 核, 每专家每 m-group 18 个 n-tile):
      波宽 1 -> 单核单波最多 1 个 tile -> 深度 1 可行 (230.58)
      波宽 2 -> 最多 2 个             -> 深度 1 死锁, 深度 2 可行 (235.81)
    """
    # 波宽 1: 每核每波最多 1 个 tile, 深度 1 就够
    assert _run_mgw_depth(1, 1)["dag_end_us"] > 0
    # 波宽 2: 最多 2 个, 深度 1 不相容
    with pytest.raises(ValueError, match="不相容"):
        _run_mgw_depth(2, 1)
    # 把深度加到 2 就相容了 —— 不是"stage 栅栏必须深度 0"
    assert _run_mgw_depth(2, 2)["dag_end_us"] > 0
    # 深度 0 (分段式: GMM1 走 L0C->GM) 永远相容
    assert _run_mgw_depth(2, 0)["dag_end_us"] > 0


def test_stage_barrier_check_accounts_for_late_binding():
    """晚绑定下调度器能在池内摊平, 所以下界是 ceil(该波 tile 数 / 核数)."""
    with pytest.raises(ValueError, match="不相容"):
        _run_mgw_depth(2, 1, late=("AIC", "AIV1"))


def test_stage_barrier_error_names_the_three_ways_out():
    with pytest.raises(ValueError) as ei:
        _run_mgw_depth(2, 1)
    msg = str(ei.value)
    assert "depth 设 0" in msg
    assert "m_groups_per_wave" in msg
    assert "UB 深度加到" in msg


def test_barriers_cost_wall_clock():
    """融合 (无栅栏) 必须不慢于分段 —— 这就是"融合值多少"的那个差.

    同样 depth=0 下实测:
        9216/3   融合 226.04  波间 254.82 (+12.7%)  分段 226.38 (+0.2%)
        9216/6   融合 362.57  波间 458.86 (+26.6%)  分段 402.59 (+11.0%)
        18432/6  融合 632.79  波间 678.57 (+7.2%)   分段 716.01 (+13.2%)
    """
    for hd, local in ((9216, 3), (9216, 6)):
        fused = _run(hd, local, (), 0)["dag_end_us"]
        for b in (("wave",), ("stage",), ("wave", "stage")):
            assert _run(hd, local, b, 0)["dag_end_us"] >= fused - 1e-9, (hd, local, b)


def test_unknown_barrier_kind_rejected():
    with pytest.raises(ValueError, match="barriers 只支持"):
        _run(barriers=("expert",))


# ---------------------------------------------------------------------------
# C4: 波宽是决策变量
# ---------------------------------------------------------------------------

def _run_mgw(mgw, local=6, hd=9216):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hd, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(m_groups_per_wave=mgw))["rank_results"][0]


def test_m_groups_per_wave_is_directly_settable():
    """给了就直接当波宽用, 不再只能经 p1/p2 间接表达."""
    for mgw in (1, 2, 3, 4, 6):
        rr = _run_mgw(mgw)
        assert rr["m_groups_per_wave"] == mgw


def test_derived_wave_width_is_not_the_best():
    """推导值 (p1=p2=1 -> 2) 不是最优 —— 所以它必须是能扫的维度.

    实测 9216/6: 波宽 1/2/3/4/6 -> 418.55 / 387.07 / 374.96 / 398.80 / 369.79,
    推导值是 2。
    """
    derived = _run_mgw(0)
    assert derived["m_groups_per_wave"] == 2
    assert _run_mgw(3)["dag_end_us"] < derived["dag_end_us"]
    assert _run_mgw(6)["dag_end_us"] < derived["dag_end_us"]
