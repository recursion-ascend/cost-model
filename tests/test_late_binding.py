"""晚绑定 (ModelOptions.late_bind_pools): AIC 池的 work-conservation 硬不变量."""
import moe_cost_model as m
from moe_cost_model.analysis import idle_decomposition


def _run(hidden_dim, local, late):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    res = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hidden_dim, aic_num=28,
        costs=m.build_analytical_costs(h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(late_bind_pools=late))
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
