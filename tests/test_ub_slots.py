"""C2: GMM1->ACT 的 UB 深度是**容量**, 不是程序序边.

物理: GMM1 的结果经 L0C->UB 的 Fixpipe 直给配对 AIV0, UB 里能同时存 depth 块。
所以约束是"同一核上同时在飞的块数 <= depth", 而不是"第 i 个 GMM1 等第 i-depth 个 ACT"。
后者把 kernel 在该核上的发射顺序钉进了图里。
"""
import collections

import pytest

import moe_cost_model as m
from linkutil import links


def _run(depth, pipe=None, local=3, hidden_dim=9216,
         late=m.ModelOptions().late_bind_pools):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hidden_dim, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(pipeline=pipe, links=links(depth),
                               late_bind_pools=late),
        )["rank_results"][0]


def _core_of(e):
    """事件实际落的核号: 取调度结果的资源, 不取 meta["core"].

    meta["core"] 是建图时的静态核号; 晚绑定下真正的核由调度器在派发时刻定,
    只有 resources 里才是对的。
    """
    for r in e.resources:
        if ":" in r:
            return r.rsplit(":", 1)[-1]
    return e.meta.get("core")


def _max_inflight(evs):
    """同一核上同时"GMM1 已开始、配对 ACT 未结束"的块数峰值."""
    by = {e.name: e for e in evs}
    per_core = collections.defaultdict(list)
    for e in evs:
        if e.meta.get("stage") != "gmm1" or e.meta.get("phase") not in (None, "fix"):
            continue
        act = by.get(e.name.replace(".gmm1.", ".act."))
        if act is None:
            continue
        # 拆相位时持有从 grant 相位开始 (lg 取槽), 否则从事件本身开始
        head = by.get(e.name + ".lg") or e
        per_core[_core_of(head if head is not e else e)].append(
            (head.start_us, act.end_us))
    worst = 0
    for ivs in per_core.values():
        for t in {x for p in ivs for x in p}:
            worst = max(worst, sum(1 for s, end in ivs if s <= t < end - 1e-12))
    return worst


@pytest.mark.parametrize("late", [(), ("AIC", "AIV1")])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("mte_aic", [None, 1, 2])
def test_ub_depth_is_respected(depth, mte_aic, late):
    """各组合下容量都不得被突破 —— 含相位拆分 x 静态/晚绑定.

    回归锚: 拆相位时 lg 相位原先丢掉了非引擎 acquire, 只留 release, 计数器变负,
    这条约束悄悄失效 (该形状上墙钟因此虚低 20%+)。
    """
    pipe = None if mte_aic is None else m.PipelineConstraints(
        queues=m.QueueDepths(mte_aic=mte_aic))
    evs = _run(depth, pipe, late=late)["events"]
    assert _max_inflight(evs) <= depth


def test_ub_depth_binds_at_one_and_relaxes():
    """深度 1 必须吃满 (是真瓶颈), 放开后墙钟下降."""
    d1, d2 = _run(1), _run(2)
    assert _max_inflight(d1["events"]) == 1
    assert _max_inflight(d2["events"]) == 2
    assert d2["dag_end_us"] < d1["dag_end_us"]


def test_ub_depth_zero_drops_the_constraint():
    """depth=0 = 假设 UB 不构成瓶颈: 不申报 token, 也就没有上限."""
    assert _run(0)["dag_end_us"] <= _run(2)["dag_end_us"] + 1e-9
