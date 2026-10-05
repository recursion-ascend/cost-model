"""stage 边 (StageLink): 消费者等多少 / 中间结果放哪 / 片上存几块.

本文件钉 activation->gmm2 这条边的落点:
  location="gm"     物化: GMM2 的 A 从 GM 读回 (付 m·K2 字节), tile->核 自由
  location="onchip" 不物化: A 留片上 (不付字节), 代价是整个 m-group 共位一个核

要守住的是"不混口径": 物化的 DAG 配不物化的公式 = 把物化算成近乎免费。
就绪粒度 (readiness) 见 test_tiling, UB 深度 (depth) 见 test_ub_slots。
"""
import pytest

import moe_cost_model as m
from linkutil import links

LOCAL, HD = 3, 9216


def _run(**kw):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(LOCAL)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(LOCAL)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=HD, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(**kw))["rank_results"][0]


def test_default_is_the_materialising_orchestration():
    """缺省落 GM = 最少假设 (不声称实现能把中间结果留在片上)."""
    assert m.ModelOptions().link("activation", "gmm2").location == "gm"


def test_unknown_location_is_refused():
    with pytest.raises(ValueError, match="location"):
        m.StageLink("activation", "gmm2", location="l2")


def test_duplicate_edge_is_refused():
    """同一条边两条说法 -> "谁生效"会变成实现细节, 直接拒绝."""
    with pytest.raises(ValueError, match="重复的边"):
        m.ModelOptions(links=(m.StageLink("activation", "gmm2"),
                              m.StageLink("activation", "gmm2", readiness=0)))


def test_materialised_charges_the_a_stream_read():
    """A 流进了 GMM2 的时长: 去掉它只会更快, 不会更慢."""
    gm = _run()["dag_end_us"]
    free = _run(links=links(location="onchip"), late_bind_pools=("AIC",))["dag_end_us"]
    assert gm > 0 and free > 0


def test_a_stream_bytes_are_declared_under_materialisation():
    """每个 m-group 的 A 要被读 (GMM2 tile 数) 次 —— 访存量里看得见."""
    gm = _run()["traffic_bytes"]
    assert gm.get("R0.gm_to_l1", 0.0) > 0.0


def test_onchip_declares_no_act_gm_write():
    """不物化就不写 GM: ACT 的写出申报清零."""
    onchip = _run(links=links(location="onchip"), late_bind_pools=("AIC",))["traffic_bytes"]
    assert onchip.get("R0.hbm_write", 0.0) == 0.0
    assert _run()["traffic_bytes"].get("R0.hbm_write", 0.0) > 0.0


def test_onchip_needs_late_binding_to_express_colocation():
    """静态钉核下共位无从表达 (核号本来各不相同), 直接报错而不是算出一个错数.

    缺省已经是晚绑定, 所以这里显式钉核才触发。
    """
    with pytest.raises(ValueError, match="late_bind_pools"):
        _run(links=links(location="onchip"), late_bind_pools=())


def _cores_per_mgroup(res):
    """每个 m-group 的 GMM1/ACT/GMM2 实际落了几个核号 (按调度结果, 不按 meta)."""
    out = {}
    for e in res["events"]:
        if e.meta.get("stage") not in ("gmm1", "activation", "gmm2"):
            continue
        key = (e.meta.get("wave"), e.meta.get("expert"), e.meta.get("slice"),
               e.meta.get("mgroup"))
        out.setdefault(key, set()).add(e.resources[0].rsplit(":", 1)[-1])
    return out


def test_onchip_colocates_a_whole_mgroup_on_one_core():
    """共位的后果: 一个 m-group 的 GMM1/ACT/GMM2 全在一个核号上.

    并行度上限因此变成 m-group 数而不是核数 —— 这是该编排要被评估的代价,
    不是违反"有就绪的活就不空闲": 多出来的核本来就没有可并行的活。
    """
    onchip = _cores_per_mgroup(_run(links=links(location="onchip"), late_bind_pools=("AIC",)))
    assert max(len(v) for v in onchip.values()) == 1
    gm = _cores_per_mgroup(_run(late_bind_pools=("AIC",)))
    assert max(len(v) for v in gm.values()) > 1       # 物化下 tile->核 自由


def test_onchip_trades_wall_clock_for_traffic():
    """少搬字节、多占墙钟 —— cost model 要能把这笔交换量化出来."""
    gm = _run(late_bind_pools=("AIC",))
    oc = _run(links=links(location="onchip"), late_bind_pools=("AIC",))
    assert oc["traffic_bytes"]["R0.gm_to_l1"] < gm["traffic_bytes"]["R0.gm_to_l1"]
    assert oc["total_us"] > gm["total_us"]


def test_anchor_binds_before_anything_colocated_to_it():
    """共位的锚点总比跟着它的事件先派发 —— 所以"锚点未绑定"这一支取不到.

    为什么必然: 锚点是该 m-group 里 order 最小的那个 GMM1, 而同组 GMM1 的依赖边
    完全相同 (同样的 m 行, 只是列不同), 所以它们同时进就绪集, 就绪堆的并列判据是
    order, 锚点先走。ACT/GMM2 更晚 —— 它们以该 GMM1 为依赖。
    这条断言钉住的是**派发次序**, 不是落核: 落核另有兜底 (派发时锚点已绑定则候选
    收窄为那一个核号), 但如果次序反过来, 起始时刻会按另一个核的空闲时刻算出来,
    而真正绑上的是锚点的核 —— 一个偏乐观的 start。次序在此, 偏差就不存在。
    """
    import moe_cost_model.scheduler as sch

    captured = []
    orig = sch.MultiResourceScheduler.schedule

    def spy(self, events, *a, **kw):
        captured.extend(events)
        return orig(self, events, *a, **kw)

    sch.MultiResourceScheduler.schedule = spy
    try:
        res = _run(links=links(location="onchip"), late_bind_pools=("AIC",))
    finally:
        sch.MultiResourceScheduler.schedule = orig

    anchor_of = {e.name: e.colocate_with for e in captured if e.colocate_with}
    assert anchor_of, "该编排本应产生共位事件"
    pos = {e.name: i for i, e in enumerate(res["events"])}   # events 按派发序
    late = [(n, a) for n, a in anchor_of.items()
            if n in pos and a in pos and pos[a] > pos[n]]
    assert not late, f"共位事件早于锚点派发: {late[:3]}"
