"""C6: dispatch 的"谁取哪些行"是调度决策, 不是建图时写死的 kernel 记账."""
import pytest

import moe_cost_model as m


def _run(mode="kernel", rows_per_item=0, local=3, hd=9216, late=()):
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


def test_rows_mode_conserves_every_row():
    """换切法不能漏行也不能重复取 —— 建图器本来就有这条守恒校验, 这里显式钉住."""
    for mode, rpi in (("kernel", 0), ("rows", 0), ("rows", 64), ("rows", 16)):
        for expert_group, (got, need) in _rows_per_group(_run(mode, rpi)).items():
            assert got == need, (mode, rpi, expert_group, got, need)


def test_granularity_changes_parallelism():
    """切得细 -> 工作项多 -> 用得上更多核. 实测 (9216/3):
        kernel        66 项 / 28 核 / 251.81
        rows  缺省     12 项 /  8 核 / 257.89   <- 每段一项, 只有 12 段
        rows  16 行    48 项 / 28 核 / 249.72   <- 比 kernel 还快
    """
    coarse = _run("rows", 0)
    fine = _run("rows", 16)
    def n(rr):
        return len([e for e in rr["events"] if e.meta.get("stage") == "dispatch"])

    def cores(rr):
        return len({e.resources[0] for e in rr["events"]
                    if e.meta.get("stage") == "dispatch"})
    assert n(fine) > n(coarse)
    assert cores(fine) > cores(coarse)
    assert fine["dag_end_us"] < coarse["dag_end_us"]


def test_kernel_mode_is_the_default():
    assert m.ModelOptions().dispatch_partition == "kernel"
    assert _run()["dag_end_us"] == _run("kernel")["dag_end_us"]


def test_rows_mode_works_with_late_binding():
    """rows 模式下核号只是轮转占位, AIV1 入池后由调度器决定."""
    rr = _run("rows", 16, late=("AIC", "AIV1"))
    for role in ("AIC", "AIV0", "AIV1"):
        (rep,) = [v for k, v in rr["idle_decomposition"].items() if k.endswith(role)]
        assert rep.avoidable_idle_us < 1e-6, role


def test_unknown_partition_rejected():
    with pytest.raises(ValueError, match="dispatch_partition"):
        _run("balanced")
