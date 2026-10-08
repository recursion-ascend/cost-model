"""相位流水 + 晚绑定同用 (原缺口 9).

相位事件 (.lg/.ld/.cb/fix) 不持核资源 —— 它们代表同一个核里不同引擎的工作, 在时间上
重叠, 各自独占核资源就等于没拆。它们"属于哪个核"靠按核的计数信号量记着, 而晚绑定下
核号到派发时刻才定。

解决办法是**核组**: 同一个 tile 的几个相位编成一组, 核号由组里最先派发的那个事件选定,
同组其余事件跟随。本文件约束两件事:
  1. 两者同用不再报错, 且不变量 (有就绪的活不空闲) 仍然成立;
  2. L1 缓冲槽 (QUEUE:mte_aic) 的容量**按真正落到的核**计数 —— 这是原先回填不了核号时
     会悄悄失效的那条约束 (失效表现为偏快)。
"""
import collections

import pytest

import moe_cost_model as m

LATE = ("AIC", "AIV1")


def _run(mte_aic=2, late=LATE, local=3, hd=9216):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    pipe = m.PipelineConstraints(queues=m.QueueDepths(mte_aic=mte_aic))
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hd, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(pipeline=pipe, late_bind_pools=late))["rank_results"][0]


def _core(e):
    """事件真正落到的核号 (晚绑定下 meta["core"] 已由引擎改写成绑定核)."""
    for r in e.resources:
        if ":" in r:
            return r.rsplit(":", 1)[-1]
    return str(e.meta.get("core"))


def test_phase_split_and_late_binding_can_be_used_together():
    rr = _run()
    assert rr["total_us"] > 0
    for role in ("AIC", "AIV0", "AIV1"):
        (rep,) = [v for k, v in rr["idle_decomposition"].items() if k.endswith(role)]
        assert rep.avoidable_idle_us < 1e-6, role


def test_every_phase_of_a_tile_lands_on_one_core():
    """核组的核心性质: 同一个 tile 的各相位必须同核 (否则按核的槽扣错)."""
    by_tile = collections.defaultdict(set)
    for e in _run()["events"]:
        if e.meta.get("phase") is None:
            continue
        name = e.name
        for suffix in (".lg", ".ld", ".cb"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        by_tile[name].add(_core(e))
    assert by_tile, "没有相位事件 —— 拆相位没生效"
    bad = {k: v for k, v in by_tile.items() if len(v) != 1}
    assert not bad, f"这些 tile 的相位散在多个核上: {list(bad)[:3]}"


@pytest.mark.parametrize("mte_aic", [2, 3])      # 深度 1 不拆相位, 没有 .lg 可测
def test_l1_buffer_slots_are_counted_on_the_real_core(mte_aic):
    """L1 缓冲槽 = QUEUE:mte_aic: 同一核上在飞的载入数不得超过深度.

    回归锚: 核号回填不了时这条约束会悄悄失效 (工作在 A 核跑、槽从 B 核扣),
    表现为墙钟偏快。
    """
    evs = _run(mte_aic=mte_aic)["events"]
    by = {e.name: e for e in evs}
    per_core = collections.defaultdict(list)
    for e in evs:
        if not e.name.endswith(".lg"):
            continue
        fx = by.get(e.name[: -len(".lg")])          # 归还槽的是 fix 相 (不带后缀)
        if fx is None:
            continue
        per_core[_core(e)].append((e.start_us, fx.end_us))
    assert per_core, "没有 .lg 相位 —— 拆相位没生效"
    for core, ivs in per_core.items():
        for t in {x for p in ivs for x in p}:
            n = sum(1 for s, end in ivs if s <= t < end - 1e-12)
            assert n <= mte_aic, f"核 {core} 在 t={t} 有 {n} 个载入在飞 > {mte_aic}"


def test_deeper_l1_is_not_slower_under_late_binding():
    """槽更多不该更慢 —— 若核号扣错, 深度就不再起作用, 这条也就测不出东西."""
    d1 = _run(mte_aic=1)["total_us"]
    d3 = _run(mte_aic=3)["total_us"]
    assert d3 <= d1 + 1e-9
