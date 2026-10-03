"""晚绑定 (ModelOptions.late_bind_pools): AIC 池的 work-conservation 硬不变量."""
import pytest

import moe_cost_model as m


def _rank(hidden_dim, local, late, pacing="per_core"):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    res = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hidden_dim, aic_num=28,
        costs=m.build_analytical_costs(h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(late_bind_pools=late, dispatch_pacing=pacing))
    return res["rank_results"][0]


def _run(hidden_dim, local, late, pacing="per_core"):
    return _rank(hidden_dim, local, late, pacing)["events"]


def _report(rank_result, role):
    """用 simulate 算好的空闲分解 —— 它带了容量表, 能判"这个空闲核的槽有没有余量".

    测试里不要自己调 idle_decomposition(evs, role): 不传 capacities 会退化成上界,
    把等 UB 槽也算成违规。
    """
    (rep,) = [v for k, v in rank_result["idle_decomposition"].items()
              if k.rstrip(":").endswith(role.rstrip(":"))]
    return rep


def _aic(rank_result):
    return _report(rank_result, "AIC")


def test_aic_late_binding_is_work_conserving():
    static, late = _rank(9216, 3, ()), _rank(9216, 3, ("AIC",))
    assert _aic(static).avoidable_idle_us > 1.0          # 基线确有违规
    assert _aic(late).avoidable_idle_us < 1e-6           # 晚绑定后为 0
    # 只换"哪个核做", 不增删工作量
    assert abs(_aic(late).busy_us - _aic(static).busy_us) < 1e-6


def test_act_stays_on_its_gmm1_core():
    evs = _run(9216, 3, ("AIC",))
    by = {e.name: e for e in evs}
    acts = [e for e in evs if e.meta.get("stage") == "activation"]
    assert acts
    for a in acts:
        g = by[a.name.replace(".act.", ".gmm1.")]
        assert a.resources[0].rsplit(":", 1)[1] == g.resources[0].rsplit(":", 1)[1]


def _avoid(rank_result, role):
    return _report(rank_result, role).avoidable_idle_us


def test_aic_and_aiv1_late_binding_work_conserving_all_pacings():
    for pacing in ("per_core", "wave", "none"):
        rr = _rank(9216, 6, ("AIC", "AIV1"), pacing)
        for role in ("AIC:", "AIV0:", "AIV1:"):
            assert _avoid(rr, role) < 1e-6, (pacing, role)


def test_static_aiv1_violates_under_per_core_pacing():
    assert _avoid(_rank(9216, 6, ()), "AIV1:") > 1.0


def test_bad_pacing_rejected():
    with pytest.raises(ValueError):
        _run(9216, 3, (), "bogus")


def test_call_overhead_charged_once_per_wave_core_and_conserving():
    W, PER, local = 5, 64, 3
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    rr = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=9216, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(t_call_oh_us=1.006)),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(late_bind_pools=("AIC", "AIV1")))["rank_results"][0]
    evs = rr["events"]
    charged = [(e.meta["wave"], e.resources[0]) for e in evs if e.meta.get("once_per_core_us")]
    assert charged and len(charged) == len(set(charged))
    # 每个本波做过 dispatch 的核都付过一次
    worked = {(e.meta["wave"], e.resources[0]) for e in evs if e.meta.get("stage") == "dispatch"}
    assert set(charged) == worked
    for role in ("AIC:", "AIV1:"):
        assert _avoid(rr, role) < 1e-6


def _run_pol(hidden_dim, local, late, policy=None, pacing="per_core"):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hidden_dim, aic_num=28,
        costs=m.build_analytical_costs(h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6, scheduling_policy=policy,
        options=m.ModelOptions(late_bind_pools=late, dispatch_pacing=pacing))["rank_results"][0]


def test_critical_path_tiebreak_keeps_zero_idle():
    """零空闲之上按关键链打破平手: 两种选序都不得违规, 工作量也不变.

    **不断言墙钟更好**: 两者都是 work-conserving, 只是就绪事件的先后不同, 所以
    Graham 异常依然可能发生。实测 (晚绑定双池, 违规两者都是 0):
        9216/3   贪心 206.11  关键路径 199.13   CP 更好
        9216/6   贪心 379.77  关键路径 366.94   CP 更好
        14336/6  贪心 549.64  关键路径 542.97   CP 更好
        18432/6  贪心 684.00  关键路径 726.66   **CP 更差 6.2%**
    所以关键路径是"通常更好"的启发式, 不是保证。要选哪个得按形状扫。
    """
    pools = ("AIC", "AIV1")
    for hd, local in ((9216, 3), (18432, 6)):
        greedy = _run_pol(hd, local, pools)
        cp = _run_pol(hd, local, pools, m.WorkConservingCriticalPath())
        for role in ("AIC:", "AIV0:", "AIV1:"):
            assert _avoid(greedy, role) < 1e-6
            assert _avoid(cp, role) < 1e-6
        # 同样的工作量, 只是顺序不同
        assert abs(_aic(cp).busy_us - _aic(greedy).busy_us) < 1e-6


def test_critical_path_usually_wins_but_not_always():
    """记录上面那条结论: 小形状 CP 更好, 18432/6 上 CP 更差."""
    pools = ("AIC", "AIV1")
    better = worse = 0
    for hd, local in ((9216, 3), (9216, 6), (18432, 6)):
        g = _run_pol(hd, local, pools)["dag_end_us"]
        c = _run_pol(hd, local, pools, m.WorkConservingCriticalPath())["dag_end_us"]
        better += c < g
        worse += c > g
    assert better >= 2 and worse >= 1


def test_remaining_path_computed_only_when_policy_asks():
    cp = _run_pol(9216, 3, (), m.WorkConservingCriticalPath())
    assert any(e.meta.get("remaining_path_us", 0) > 0 for e in cp["events"])
    greedy = _run_pol(9216, 3, ())
    assert all("remaining_path_us" not in e.meta for e in greedy["events"])


def test_gmm1_activation_depth_zero_drops_the_ub_constraint():
    """depth=0 = 不要 GMM1->ACT 的 UB 槽约束.

    两段历史: 以前这里是一条"距离依赖"边, 且 depth=0 会取到 history[-0]=history[0]
    (空表直接崩); 现在是计数信号量 (C2), depth=0 则不申报 token。
    """
    W, PER, local = 5, 64, 3
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6

    def run(depth):
        return m.simulate_routing_counts(
            routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=9216, aic_num=28,
            costs=m.build_analytical_costs(
                h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
            p1_override=1, p2_override=1, topk=6,
            policy=m.InstancePolicy(gmm1_activation_depth=depth))["rank_results"][0]

    name = "R0.W0.E1.S1.gmm1.m0.n10"   # C1 后事件名不带核号
    d1 = {e.name: e for e in run(1)["events"]}[name]
    d0 = {e.name: e for e in run(0)["events"]}[name]
    # depth=1: 等 UB 槽 (容量); depth=0: 无此约束, 只被自己的核卡住
    assert d1.critical_reason == "capacity"
    assert d0.critical_reason.startswith("resource:")
    assert d0.start_us < d1.start_us
