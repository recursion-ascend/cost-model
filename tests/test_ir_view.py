"""事件图的类型化视图: 把字符串约定读成类型, 并把推不出来的东西计数.

这一层针对的是一条具体批评: Event 能表达依赖/资源/信号量/字节, 但**没有明确表达**
MTE2/MTE3 异步发射、硬件 flag、counter、buffer slot、读写内存效果 —— 它们都藏在
字符串约定里 ("MTE2:c7" / "QUEUE:mte_aic:c7" / "gm_to_l1" / deps)。约定不可查询:
调度器靠 rsplit 解核号, 相位拆分靠 (同 stage, 同 core) **猜**程序序边, 访存方向靠
endswith 判断。换一份 kernel 时这些约定不会报错, 只会悄悄对不上。

视图是**只读**的, 所以它对时长与 Event.order 零影响 (judge: golden 39 个指纹不变)。
它要守住三件事:
  1. 类型化读法与字符串约定一致, 且**认不出来就返回 None**, 不编造语义;
  2. 真正表达不了的东西写成明文 (UNREPRESENTABLE), 而不是留白 —— 留白会被当成"已建模";
  3. 推不出来的边标 UNKNOWN 并计数, 不把程序序冒充成数据依赖。
"""
import pytest

import moe_cost_model as m
from moe_cost_model.ir import (UNREPRESENTABLE, DependencyKind, Engine, EventGraphView,
                               MemorySpace, Pipe, TokenKind, TransferDirection,
                               classify_resource, classify_token, classify_transfer)
from moe_cost_model.shape import MegaMoeShape


def _events(urma=False, **shape_kw):
    P = m.MEGAMOE_A8W8
    costs = m.build_analytical_costs(
        h=1024, dispatch_mechanistic=m.DispatchMechanisticLatency(),
        **({"urma_mechanistic": m.UrmaMechanisticLatency()} if urma else {}))
    kw = P.shape_kw()
    kw["kernel"] = m.KernelConfig(topo_urma=urma, combine_meta_bytes_per_row=32)
    kw.update(shape_kw)
    model = m.A8W8WaveCostModel(costs, P.options)
    shape = MegaMoeShape(expert_tokens=(64, 64), token_num=16, h=1024, hidden_dim=1024,
                         aic_num=4, topk=8, p1_override=1, p2_override=1,
                         expert_source_tokens=((32, 32), (32, 32)), **kw)
    return model.build_events(shape)[0]


# --------------------------------------------------------------- 词表与分类

def test_resource_names_classify_including_the_late_binding_placeholder():
    """"AIC:c7" 与 "AIC:*" 都要认: 后者是晚绑定占位, 核号派发时才定."""
    pinned = classify_resource("R0.AIC:c7")
    assert (pinned.engine, pinned.core, pinned.late_bound) == (Engine.AIC, 7, False)
    pooled = classify_resource("R0.AIV1:*")
    assert (pooled.engine, pooled.core, pooled.late_bound) == (Engine.AIV1, None, True)


def test_token_kinds_separate_hardware_facts_from_orchestration_choices():
    """执行单元 (容量恒 1 的硬件事实) 与缓冲槽 (编排选择) 必须分型.

    两者用的是**同一个** acquires/releases 机制, 所以不分型就没法说"这个容量能不能调"
    —— 而这正是 EngineQueueDepths 当初被当成参数的根因 (它其实是空约束)。
    """
    unit = classify_token("R0.MTE2:c3")
    slot = classify_token("R0.QUEUE:mte_aic:c3")
    ub = classify_token("R0.UB:gmm1act:c3")
    queue = classify_token("R0.Q:aic:c3")
    assert unit.kind is TokenKind.EXECUTION_UNIT and unit.is_hardware_fact
    assert (slot.kind, slot.space, slot.pipe) == (TokenKind.BUFFER_SLOT,
                                                 MemorySpace.L1, Pipe.MTE2)
    assert (ub.kind, ub.space) == (TokenKind.BUFFER_SLOT, MemorySpace.UB)
    assert queue.kind is TokenKind.ENGINE_QUEUE and not queue.is_hardware_fact
    assert slot.core == 3 and not slot.is_hardware_fact


def test_transfers_expose_both_ends_and_the_direction():
    """访存的方向与两端内存原先只藏在通路名里, 现在是字段."""
    load = classify_transfer("R0.gm_to_l1", 1024)
    assert (load.src, load.dst, load.direction) == (MemorySpace.GM, MemorySpace.L1,
                                                    TransferDirection.READ)
    write = classify_transfer("R0.hbm_write", 512)
    assert (write.src, write.dst, write.direction) == (MemorySpace.UB, MemorySpace.GM,
                                                       TransferDirection.WRITE)
    read_back = classify_transfer("R0.combine_read", 256)
    assert (read_back.src, read_back.dst) == (MemorySpace.GM, MemorySpace.UB)
    fab = classify_transfer("fab_src:2", 128)
    assert (fab.peer_rank, fab.protocol, fab.dst) == (2, "fabric", MemorySpace.PEER_GM)


def test_unknown_names_return_none_instead_of_a_guess():
    """认不出来就返回 None —— 本层不编造语义.

    DISPATCH_COMM 是个全局资源 (不带核号), 不该被硬塞成某个引擎; 未知通路保留名字
    但两端留 None, 让调用方自己决定怎么办。
    """
    assert classify_resource("DISPATCH_COMM") is None
    assert classify_resource("R0.NOT_AN_ENGINE:c1") is None
    assert classify_token("R0.MYSTERY:c1") is None
    unknown = classify_transfer("some_new_channel", 8)
    assert unknown.src is None and unknown.dst is None and unknown.bytes == 8.0


# --------------------------------------------------------------- 图视图

def test_view_reads_a_real_graph_without_unresolved_names():
    """真实建图的产物里不该有认不出来的资源名或令牌名.

    出现了就说明建图器引入了一个本层不认识的约定 —— 那正是"换 kernel 时悄悄对不上"的
    前兆, 所以这里当作错误而不是容忍。
    """
    view = EventGraphView(_events())
    summary = view.summary()
    assert summary["unresolved_resources"] == []
    assert summary["unresolved_tokens"] == []
    assert summary["tasks"] > 0 and summary["edges"] > 0
    assert summary["has_dependencies"] is True


def test_buffer_lifetimes_are_paired_and_cross_event():
    """缓冲槽的取与还要配对; UB 槽必须是**跨事件**持有 (GMM1 取、配对 ACT 还).

    不配对 = 台账会漂 (计数变负或永不归还), 是真缺陷; pipeline_expand 里记着一个旧 bug:
    拆相位时丢掉了 carried acquire, 只剩 release, 约束就悄悄失效了。
    """
    view = EventGraphView(_events())
    assert view.summary()["unpaired_buffers"] == []
    ub = [b for b in view.buffers() if b.space is MemorySpace.UB
          and b.kind is TokenKind.BUFFER_SLOT]
    assert ub, "这组编排下应当有 UB 槽 (gmm1->activation 的 depth=1)"
    assert all(b.paired for b in ub)
    assert any(b.cross_event for b in ub), "UB 槽的语义就是跨事件持有"


def test_scheduled_events_have_no_edges_and_the_view_says_so():
    """排好的事件不带 deps, 边数 0 —— 必须说出来, 否则会被读成"这张图没有依赖"."""
    rc = [[[32] * 2 for _ in range(4)] for _ in range(2)]
    res = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=32, h=1024, hidden_dim=1024, aic_num=4,
        costs=m.build_analytical_costs(
            h=1024, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        topk=8, p1_override=1, p2_override=1)
    view = EventGraphView(res["rank_results"][0]["events"])
    assert view.summary()["has_dependencies"] is False
    assert view.edges == () and view.summary()["edges"] == 0
    assert view.summary()["tasks"] > 0          # 任务照样读得出来


def test_edge_kinds_are_classified_and_the_mte_path_has_no_unknown():
    """MTE 路径上每条边都能说清种类 —— 说不清的要能被看见."""
    view = EventGraphView(_events())
    kinds = view.summary()["edge_kinds"]
    assert kinds.get("unknown", 0) == 0, f"MTE 路径出现推不出来的边: {kinds}"
    assert kinds["data"] > 0 and kinds["readiness"] > 0


def test_layered_program_order_edges_are_not_passed_off_as_data():
    """Layered 在 AIV1 上串的程序序链分不出来时标 UNKNOWN, 不冒充数据依赖.

    为什么重要: 程序序边可以重排, 数据边不能。把前者当成后者, "这条边能不能去掉"的结论
    就会反过来。要真正分开需要建图器在加边时标 kind (builders/comm/urma.py 的
    aiv1_last 链), 那会动 Event, 是另一步的事。
    """
    view = EventGraphView(_events(urma=True))
    kinds = view.summary()["edge_kinds"]
    assert kinds.get("unknown", 0) > 0, (
        "Layered 的 AIV1 程序序链本应落到 UNKNOWN; 若已为 0, 说明建图器开始标 kind 了, "
        "那就把这条测试改成断言 program_order 的条数")
    by_name = {t.name: t for t in view.tasks}
    unknown_pairs = {(by_name[e.producer].kind, by_name[e.consumer].kind)
                     for e in view.edges if e.kind is DependencyKind.UNKNOWN}
    assert unknown_pairs, unknown_pairs
    # 这些对子全都落在 AIV1 的那几个 stage 之间 (localcopy / maskscan / recv / combine)
    aiv1_stages = {"dispatch_local", "mask_scan", "dispatch_recv", "combine"}
    assert all(p in aiv1_stages and c in aiv1_stages for p, c in unknown_pairs)


def test_transfer_totals_aggregate_by_memory_pair():
    """按 (源, 目的) 聚字节: 原先只能按通路名聚, 方向藏在名字里."""
    view = EventGraphView(_events())
    totals = view.transfer_totals()
    assert totals[(MemorySpace.GM, MemorySpace.L1)] > 0       # GMM 的 A/B 流载入
    assert totals[(MemorySpace.UB, MemorySpace.GM)] > 0       # ACT/combine 的写出
    assert sum(totals.values()) == pytest.approx(
        sum(t.bytes for task in view.tasks for t in task.transfers))


# --------------------------------------------------------------- 缺口要写成明文

def test_unrepresentable_things_are_declared_not_silent():
    """表达不了的东西必须有明文条目 —— 留白会被当成"已经建模了".

    四条都是实打实的: 一个 duration_us 装不下"异步发射 + 完成"; flag 只有延迟没有身份;
    带宽域只有标签没有争用; 跨核等待没有"等谁的哪个 flag"。
    """
    assert set(UNREPRESENTABLE) == {
        "issue_vs_execution", "flag_identity", "bandwidth_contention",
        "cross_core_flag_wait"}
    for key, text in UNREPRESENTABLE.items():
        assert len(text) > 40, f"{key} 的说明太短, 讲不清为什么表达不了"


def test_event_still_has_no_issue_duration_field():
    """约束现状: Event 只有一个时长字段.

    哪天真加了 issue_duration, 这条会红 —— 提醒把 UNREPRESENTABLE 里那条删掉, 并且
    重新生成 golden (拆分持有区间会改排程, 不是纯标注)。
    """
    import dataclasses

    from moe_cost_model.scheduler.events import Event
    names = {f.name for f in dataclasses.fields(Event)}
    assert "duration_us" in names
    assert not {"issue_duration_us", "execution_duration_us"} & names
