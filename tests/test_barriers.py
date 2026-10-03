"""C3: 全核栅栏原语 —— 让"融合 vs 分段式执行"能被算出来."""
import pytest

import moe_cost_model as m


def _run(hd=9216, local=3, barriers=(), depth=1):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hd, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(barriers=barriers),
        policy=m.InstancePolicy(gmm1_activation_depth=depth))["rank_results"][0]


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


def test_stage_barriers_need_ub_depth_zero():
    """物理不可能的组合要给出说得清的报错, 而不是 "capacity deadlock".

    stage 栅栏要求全部 GMM1 先于任何 ACT 完成, 而 UB 深度 1 时核 X 的第二个 GMM1
    等它第一个 ACT 还槽, 那个 ACT 又等全部 GMM1。分段式执行里 GMM1 走 L0C->GM。
    """
    with pytest.raises(ValueError, match="gmm1_activation_depth=0"):
        _run(barriers=("stage",), depth=1)
    assert _run(barriers=("stage",), depth=0)["dag_end_us"] > 0


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
