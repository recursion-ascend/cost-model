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
    """两个事件都固定在 AIC:0, AIC:1 全程空着 -> 第二个事件就绪却等着, 必须判违规."""
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


# ---- work-conservation 护栏 ----

def _guard_costs():
    return m.build_analytical_costs(
        h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency())


def test_the_guard_only_covers_pooled_roles():
    """静态钉核的池不在检查范围: 工作钉死在某个核上, 那个核忙而别处空着时搬不过去,
    那是**那种分核方式的代价** (量出来就是结论), 不是调度器没做到位。
    派发时刻绑定的池里出现 avoidable 才是不变量没守住。
    """
    rep = type("R", (), {"avoidable_idle_us": 5.0, "segments": ()})()
    reports = {"R0.AIC": rep, "R0.AIV1": rep}
    assert m.work_conservation_violations(reports, ()) == {}
    assert m.work_conservation_violations(reports, ("AIC",)) == {"AIC": 5.0}
    assert set(m.work_conservation_violations(reports, ("AIC", "AIV1"))) == {"AIC", "AIV1"}
    # 跑 ACT 的那个角色排除在外, 即使它字面上入了池: ACT 与它的 GMM1 必须同核,
    # 是成对漂移而不能独立落核, 本模块分不开"配对逼出来的空闲"与"真可回收的空闲"
    # (见模块开头"avoidable 仍是上界")。实测 golden 的 mte_topk_prefetch 在派发时刻
    # 绑定下 AIV0 有 1774.4 核·us 而 AIC/AIV1 皆 0 —— 那是共位, 不是调度没做到位。
    assert m.work_conservation_violations({"R0.AIV0": rep}, ("AIC", "AIV0")) == {}
    assert m.work_conservation_violations({"R0.AIV0": rep}, ("AIV1",)) == {}
    # 换了跑 ACT 的角色, 排除的也跟着换 (不写死 AIV0)
    assert m.work_conservation_violations({"R0.AIV1": rep}, ("AIV1",),
                                          act_role="AIV1") == {}


def test_a_non_work_conserving_policy_is_not_checked():
    """按优先级排的策略**主动**让核空着去等高优先级的 stage —— 那是它的语义.

    实测: golden 的 policy_priority_by_stage 形状在派发时刻绑定下 AIC 上有
    41879 核·us 可避免空闲。要求它工作守恒等于要求它不是它, 所以护栏跳过声明
    work_conserving=False 的策略。
    """
    assert m.EarliestStart().work_conserving is True
    assert m.PriorityByStage().work_conserving is False


def test_the_guard_is_in_the_model_layer_and_on_by_default():
    """护栏不能被绕过: 它和下界一样挂在 simulate_multi, 不在 api 层.

    缺省绑定 ("AIC","AIV1") 下模型自己必须守住不变量 —— 这里跑一个真形状确认
    既不抛异常也确实是 0, 否则这条护栏就只是个没触发过的开关。
    """
    import dataclasses

    from moe_cost_model.model import A8W8WaveCostModel

    assert A8W8WaveCostModel(_guard_costs()).check_work_conservation is True
    opts = dataclasses.replace(m.ModelOptions(), late_bind_pools=("AIC", "AIV1"))
    res = m.simulate_routing_counts(
        routing_counts=[[[64] * 2 for _ in range(4)] for _ in range(2)],
        token_num_per_rank=64, h=5120, hidden_dim=9216, aic_num=28, topk=8,
        p1_override=1, p2_override=1, options=opts, costs=_guard_costs())
    for rr in res["rank_results"].values():
        for key, rep in (rr["idle_decomposition"] or {}).items():
            if key.rsplit(".", 1)[-1] in ("AIC", "AIV0", "AIV1"):
                assert rep.avoidable_idle_us <= 1e-6, (key, rep.avoidable_idle_us)
