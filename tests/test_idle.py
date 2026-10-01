"""空闲分解: 守恒恒等式、与模型自算 busy 一致、零时长事件、WC 判定的正反例.

这几条都来自实际踩过的坑 —— 见 analysis/idle.py 的模块说明。
"""
import moe_cost_model as m
from moe_cost_model.analysis import idle_decomposition
from moe_cost_model.scheduler import Event, MultiResourceScheduler


def _run(hidden_dim=9216, local=3, aic=28):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hidden_dim,
        aic_num=aic, costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6)


def test_conservation_identity():
    """busy + forced + avoidable == 核数 x horizon, 逐池成立."""
    rr = _run()["rank_results"][0]
    for role in ("AIC:", "AIV0:", "AIV1:"):
        for rep in idle_decomposition(rr["events"], role).values():
            assert abs(rep.busy_us + rep.idle_us - rep.capacity_us) < 1e-6, role
            assert rep.capacity_us == len(rep.pool) * rep.horizon_us


def test_busy_matches_model_resource_busy():
    """分解算出的 busy 必须与模型自己的 resource_busy_us 逐位一致.

    零时长事件 (dispatch_ready / moe_stage_done) 若被计入占用, 这条就会失败 ——
    实测曾把 AIC busy 从 5455.0 虚报到 6438.7。
    """
    rr = _run()["rank_results"][0]
    for role in ("AIC:", "AIV0:", "AIV1:"):
        tag = role.rstrip(":")
        got = sum(r.busy_us for r in idle_decomposition(rr["events"], role).values())
        ref = sum(v for k, v in rr["resource_busy_us"].items() if f".{tag}:" in k)
        assert abs(got - ref) < 1e-6, (role, got, ref)


def test_horizon_covers_tail():
    """horizon 缺省取全部事件的最大 end (≈dag_end), 不是池内事件的 —— 否则尾段
    (epilogue 在 AIV 上) 期间 AIC 的 forced 空闲会被整段藏掉."""
    rr = _run()["rank_results"][0]
    rep = idle_decomposition(rr["events"], "AIC:")["R0"]
    assert abs(rep.horizon_us - rr["dag_end_us"]) < 1e-6


def test_idle_decomposition_in_rank_results():
    """simulate 的结果里直接带分解, 三个角色池都有."""
    rr = m.simulate(m.load_scenario("examples/scenario_basic.toml"))["rank_results"][0]
    d = rr["idle_decomposition"]
    assert {"R0.AIC", "R0.AIV0", "R0.AIV1"} <= set(d)
    assert d["R0.AIC"].busy_us > 0


def _sched(events, caps):
    return MultiResourceScheduler().schedule(events, capacities=caps)


def test_work_conserving_positive_case():
    """两个核各一个无依赖事件: 没有一刻是"有活却空着" -> avoidable == 0."""
    evs = [Event("a", ("AIC:0",), 10.0, order=0, meta={"stage": "gmm1"}),
           Event("b", ("AIC:1",), 10.0, order=1, meta={"stage": "gmm1"})]
    _, sched = _sched(evs, {})
    rep = idle_decomposition(sched, "AIC:")[""]
    assert rep.avoidable_idle_us == 0.0
    assert rep.work_conserving


def test_work_conserving_negative_case():
    """两个事件都钉在 AIC:0, AIC:1 全程空着 -> 第二个事件就绪却等着, 必须判违规."""
    evs = [Event("a", ("AIC:0",), 10.0, order=0, meta={"stage": "gmm1"}),
           Event("b", ("AIC:0",), 10.0, order=1, meta={"stage": "gmm1"}),
           Event("filler", ("AIC:1",), 1.0, order=2, meta={"stage": "gmm1"})]
    _, sched = _sched(evs, {})
    rep = idle_decomposition(sched, "AIC:")[""]
    assert not rep.work_conserving
    # b 在 [0,10) 就绪却没开始, 而 AIC:1 在 [1,10) 空着 -> 至少 9 核·us
    assert rep.avoidable_idle_us >= 9.0 - 1e-9
    assert rep.segments and rep.segments[0].waiting_ready


def test_forced_idle_is_not_counted_as_violation():
    """链式依赖: b 必须等 a, 期间 AIC:1 空着但没有就绪的活 -> 全是 forced."""
    evs = [Event("a", ("AIC:0",), 10.0, order=0, meta={"stage": "gmm1"}),
           Event("b", ("AIC:0",), 10.0, deps=("a",), order=1, meta={"stage": "gmm1"})]
    _, sched = _sched(evs, {})
    rep = idle_decomposition(sched, "AIC:")[""]
    assert rep.avoidable_idle_us == 0.0, "b 在 a 完成前并未就绪, 不该算违规"
    assert rep.work_conserving
