"""dispatch 的"谁取哪些行"是调度决策, 不是建图时写死的记账.

  "pooled" (缺省): 只按 dispatch_rows_per_item 把切片切成若干份, 哪个核去取由调度器定
  "precut":        建图时按核分好 (均衡 + 轮转) —— 某实现的分工方式
"""
import pytest

import moe_cost_model as m

DEFAULT_LATE = m.ModelOptions().late_bind_pools


def _run(mode="pooled", rows_per_item=0, local=3, hd=9216, late=DEFAULT_LATE):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hd, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(dispatch_partition=mode, dispatch_rows_per_item=rows_per_item,
                               late_bind_pools=late))["rank_results"][0]


def _rows_per_group(rr):
    """每个 (专家, m-group) 真正被取回来的行数 —— 守恒检查."""
    got = {}
    for e in rr["events"]:
        if e.meta.get("stage") != "dispatch_ready":
            continue
        got[(e.meta["expert"], e.meta["mgroup"])] = (
            e.meta["contributed_rows"], e.meta["required_rows"])
    return got


def test_every_partition_conserves_every_row():
    """换切法不能漏行也不能重复取 —— 建图器本来就有这条守恒校验, 这里显式钉住."""
    for mode, rpi in (("precut", 0), ("pooled", 0), ("pooled", 64), ("pooled", 16)):
        for expert_group, (got, need) in _rows_per_group(_run(mode, rpi)).items():
            assert got == need, (mode, rpi, expert_group, got, need)


def test_granularity_changes_parallelism():
    """切得细 -> 工作项多 -> 用得上更多核."""
    coarse = _run("pooled", 0)
    fine = _run("pooled", 16)

    def n(rr):
        return len([e for e in rr["events"] if e.meta.get("stage") == "dispatch"])

    def cores(rr):
        return len({e.resources[0] for e in rr["events"]
                    if e.meta.get("stage") == "dispatch"})
    assert n(fine) > n(coarse)
    assert cores(fine) > cores(coarse)
    assert fine["dag_end_us"] < coarse["dag_end_us"]


def test_pooled_is_the_default():
    """缺省不预设分工 (最少假设); precut 是 profiles.MEGAMOE_A8W8 的选择."""
    assert m.ModelOptions().dispatch_partition == "pooled"
    assert m.MEGAMOE_A8W8.options.dispatch_partition == "precut"
    assert _run()["dag_end_us"] == _run("pooled")["dag_end_us"]


def test_pooled_mode_works_with_late_binding():
    """pooled 下核号只是轮转占位, AIV1 入池后由调度器决定."""
    rr = _run("pooled", 16, late=("AIC", "AIV1"))
    for role in ("AIC", "AIV0", "AIV1"):
        (rep,) = [v for k, v in rr["idle_decomposition"].items() if k.endswith(role)]
        assert rep.avoidable_idle_us < 1e-6, role


def test_unknown_partition_rejected():
    with pytest.raises(ValueError, match="dispatch_partition"):
        _run("balanced")
