"""晚绑定 (ModelOptions.late_bind_pools): AIC 池的 work-conservation 硬不变量."""
import moe_cost_model as m
from moe_cost_model.analysis import idle_decomposition


def _run(hidden_dim, local, late, pacing="per_core"):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    res = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hidden_dim, aic_num=28,
        costs=m.build_analytical_costs(h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(late_bind_pools=late, dispatch_pacing=pacing))
    return res["rank_results"][0]["events"]


def _aic(evs):
    (rep,) = idle_decomposition(evs, "AIC:").values()
    return rep


def test_aic_late_binding_is_work_conserving():
    static, late = _run(9216, 3, ()), _run(9216, 3, ("AIC",))
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


def _avoid(evs, role):
    (rep,) = idle_decomposition(evs, role).values()
    return rep.avoidable_idle_us


def test_aic_and_aiv1_late_binding_work_conserving_all_pacings():
    for pacing in ("per_core", "wave", "none"):
        evs = _run(9216, 6, ("AIC", "AIV1"), pacing)
        for role in ("AIC:", "AIV0:", "AIV1:"):
            assert _avoid(evs, role) < 1e-6, (pacing, role)


def test_static_aiv1_violates_under_per_core_pacing():
    assert _avoid(_run(9216, 6, ()), "AIV1:") > 1.0


def test_bad_pacing_rejected():
    import pytest
    with pytest.raises(ValueError):
        _run(9216, 3, (), "bogus")


def test_call_overhead_charged_once_per_wave_core_and_conserving():
    W, PER, local = 5, 64, 3
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    evs = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=9216, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(t_call_oh_us=1.006)),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(late_bind_pools=("AIC", "AIV1")))["rank_results"][0]["events"]
    charged = [(e.meta["wave"], e.resources[0]) for e in evs if e.meta.get("once_per_core_us")]
    assert charged and len(charged) == len(set(charged))
    # 每个本波做过 dispatch 的核都付过一次
    worked = {(e.meta["wave"], e.resources[0]) for e in evs if e.meta.get("stage") == "dispatch"}
    assert set(charged) == worked
    for role in ("AIC:", "AIV1:"):
        assert _avoid(evs, role) < 1e-6
