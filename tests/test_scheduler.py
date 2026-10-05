"""L0/L1/L2 流水线约束机制测试.

每个机制: 中性值不变 / 激活后方向正确 / 物理语义可验证.
"""
from pathlib import Path

import moe_cost_model as m
from linkutil import links
from moe_cost_model.scheduler import Event, MultiResourceScheduler

P = m.PipelineConstraints

# 测试夹具值, 非标定常数: Cube 速率没有缺省, 测试统一取此值
CUBE_RATE = 2.7e7


def _deterministic_case():
    WORLD, LOCAL = 4, 64
    counts = [[[0] * WORLD for _ in range(LOCAL)] for _ in range(WORLD)]
    for dst in range(WORLD):
        counts[dst][0] = [75, 75, 75, 75]
        counts[dst][1] = [16, 16, 16, 16]
        counts[dst][2] = [3, 4, 3, 3]
        counts[dst][63] = [200, 20, 20, 16]
    return tuple(tuple(tuple(r) for r in c) for c in counts)


def _run(options=None, policy=None, cube_rate=CUBE_RATE, kernel=None):
    costs = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=m.AnalyticalGmmCosts(cube_mac_per_us=cube_rate).gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts(cube_mac_per_us=cube_rate).gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        count_table_prepare_us=m.T_COUNT_GATE,
    )
    return m.simulate_routing_counts(
        routing_counts=_deterministic_case(), token_num_per_rank=64, h=6144,
        hidden_dim=4096, aic_num=28, costs=costs, options=options or m.ModelOptions(),
        p1_override=2, p2_override=1,   # kernel 默认策略 @bs64 (tiling 真值)
        policy=policy, kernel=kernel,
    )


# ---- L0: 同步延迟 (Event/Flag/Barrier) ----

def test_l0_edge_latency_dag_level():
    x = Event("X", (), 5.0)
    y = Event("Y", (), 1.0, deps=("X",), dep_latency_overrides=(("X", 3.0),))
    z = Event("Z", (), 1.0, deps=("X",))
    _, s = MultiResourceScheduler().schedule([x, y, z])
    st = {e.name: e.start_us for e in s}
    assert st["Y"] == 8.0 and st["Z"] == 5.0


def test_l0_sync_latency_slows_model():
    """同步延迟进依赖边: 静态钉核下墙钟准确加上它; 工作守恒调度下只能说"不更快".

    实测本形状: 静态 237.6300 -> 239.6300 (正好 +2.0), 晚绑定 236.0690 -> 236.0565
    (-0.0125)。后者是 Graham 异常那一类 —— 边上多 2us 改变了派发次序, 工作守恒并不
    保证墙钟单调。所以单调性只在静态钉核下断言。
    """
    stat = dict(pipeline=P(), late_bind_pools=STATIC)
    base = _run(options=m.ModelOptions(**stat))
    slow = _run(options=m.ModelOptions(
        pipeline=P(sync=m.SyncLatency(gmm1_act_handshake_us=2.0)),
        late_bind_pools=STATIC))
    assert abs((slow["kernel_total_us"] - base["kernel_total_us"]) - 2.0) < 1e-6

    # 缺省 (晚绑定) 下只记录事实: 不显著更快
    lb_base = _run(options=m.ModelOptions(pipeline=P()))["kernel_total_us"]
    lb_slow = _run(options=m.ModelOptions(pipeline=P(
        sync=m.SyncLatency(gmm1_act_handshake_us=2.0))))["kernel_total_us"]
    assert lb_slow > lb_base - 0.1


# ---- L1: 缓冲槽位 / 队列深度 (MTE/Cube/Vector/Fix) ----

def test_l1_capacity_tokens():
    def chain(depth):
        evs = [
            Event("P0", (), 10.0, acquires=(("BUF", 1),)),
            Event("C0", (), 2.0, deps=("P0",), releases=(("BUF", 1),)),
            Event("P1", (), 10.0, acquires=(("BUF", 1),)),
            Event("C1", (), 2.0, deps=("P1",), releases=(("BUF", 1),)),
        ]
        _, s = MultiResourceScheduler().schedule(evs, capacities={"BUF": depth})
        return {e.name: e for e in s}

    assert chain(1)["P1"].start_us == 12.0   # 等归还
    assert chain(2)["P1"].start_us == 0.0    # 双槽并行


def test_l1_capacity_deadlock_detected():
    try:
        MultiResourceScheduler().schedule(
            [Event("P0", (), 1.0, acquires=(("BUF", 1),)),
             Event("P1", (), 1.0, acquires=(("BUF", 1),))],
            capacities={"BUF": 1})
        raise AssertionError("应检测到容量死锁")
    except ValueError:
        pass


def test_l1_buffer_depth_sweep():
    """UB 握手深度放宽后, 静态绑定下更快; 工作守恒调度下可能更慢 (调度异常).

    实测本形状: 静态 175.28 -> 171.87 (更快), 晚绑定 183.20 -> 191.35 (慢 4.4%)。
    后者是 Graham 异常那一类 —— 放宽一个容量会改变派发次序, 工作守恒并不保证
    墙钟单调。所以这里只在静态绑定下断言单调, 晚绑定下只记录事实。
    """
    stat = dict(pipeline=P(), late_bind_pools=STATIC)
    base = _run(options=m.ModelOptions(**stat))
    deeper = _run(options=m.ModelOptions(**stat, links=links(2)))
    assert deeper["kernel_total_us"] <= base["kernel_total_us"]

    lb = dict(pipeline=P())                      # 缺省晚绑定
    b2 = _run(options=m.ModelOptions(**lb))["kernel_total_us"]
    d2 = _run(options=m.ModelOptions(**lb, links=links(2)))["kernel_total_us"]
    assert b2 > 0 and d2 > 0                     # 不断言方向: 异常已被观察到


def test_from_tiling_wires_real_counts():
    """from_tiling: 槽位真值接线; gmm1 深度不被覆盖 (None 语义)."""
    import struct
    from tempfile import NamedTemporaryFile
    buf = bytearray(232)
    struct.pack_into("<10I", buf, 0, 64, 64, 6144, 4096, 4, 7, 2048, 8, 28, 56)
    struct.pack_into("<4i", buf, 80, 256, 1, 6, 6336)
    struct.pack_into("<4i", buf, 96, 512, 1, 5, 352)
    struct.pack_into("<4i", buf, 112, 512, 1, 4, 352)
    struct.pack_into("<2i", buf, 132, 2, 5)
    struct.pack_into("<I", buf, 204, 4)
    with NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(buf); path = f.name
    t = m.parse_tiling(path)
    Path(path).unlink()
    c = m.PipelineConstraints.from_tiling(t)
    assert c.buffers.dispatch_window == 6
    assert c.buffers.send_mask_with_extra == 5
    assert c.buffers.send_mask_without_extra == 4
    assert c.buffers.unpermute_in == 5
    assert c.buffers.gmm2_combine is None   # wave 路径不覆盖 credit 语义


def test_split_no_deadlock_large_dag():
    """拆相位 + 大 DAG: 距离依赖保序, 无信号量环."""
    WORLD, LOCAL = 4, 64
    counts = [[[8] * WORLD for _ in range(LOCAL)] for _ in range(WORLD)]
    rc = tuple(tuple(tuple(r) for r in c) for c in counts)
    costs = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        count_table_prepare_us=m.T_COUNT_GATE,
    )
    res = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=1024, h=6144,
        hidden_dim=4096, aic_num=28, costs=costs,
        options=m.ModelOptions(pipeline=m.PipelineConstraints(
            queues=m.QueueDepths(mte_aic=2, cube=2, fix=2)),
            late_bind_pools=STATIC),
    )
    assert res["kernel_total_us"] > 0


def test_l1_queue_depth_monotone():
    """L1 缓冲槽越多越不慢: 深度 2 允许预取下一个 tile 的 A 流 (静态绑定下)."""
    base = _run(options=m.ModelOptions(pipeline=P(), late_bind_pools=STATIC))
    deep = _run(options=m.ModelOptions(pipeline=P(
        queues=m.QueueDepths(mte_aic=2)), late_bind_pools=STATIC))
    deeper = _run(options=m.ModelOptions(pipeline=P(
        queues=m.QueueDepths(mte_aic=3)), late_bind_pools=STATIC))
    assert deeper["kernel_total_us"] <= deep["kernel_total_us"] <= base["kernel_total_us"]


# ---- L1: 相位拆分 (生产者/消费者距离, 跨 tile 流水) ----

#: 静态钉核。相位流水与晚绑定**已可同用** (核组, 见 test_phase_late_binding),
#: 这里仍钉核的用例是为了"同一口径下比较"或为了断言在工作守恒调度下不成立的单调性。
STATIC = ()


def _split(**kw):
    return m.ModelOptions(pipeline=P(queues=m.QueueDepths(mte_aic=2), **kw),
                          late_bind_pools=STATIC)


def test_phase_split_self_consistency():
    """拆相位与闭式同口径: tile 内 load 与 cube 并行, stage 忙碌时长不重复计.

    2026-10-05: 原先这里记着"已知缺口: mte_aic > 1 时同一个核可以有多个载入在飞,
    模型不阻止一个核超过自己的 GM→L1 带宽"。那个缺口已修 —— 载入相位占住本核那条
    MTE2 管道 (一个 AI Core 一条, 容量 1 的计数信号量), 所以同核载入不再重叠。
    仍不断言总时长相等: 拆相位允许**跨 tile** 的 load/cube 重叠, 闭式不允许。
    """
    # 速率要按"载入 vs 计算谁大"选, 不能按名义值。实测本夹具载入合计 4835.6 核·us:
    #   2.7e7  -> cube 590.0   (cube/载入 0.12, 载入绑定)
    #   2.0e6  -> cube 7965.0  (cube/载入 1.65, 计算绑定)
    # 6.75e6 (规格 fp16 速率) 的比值只有 0.49 —— 载入修好之后它**不再是计算绑定**,
    # 原先看着像是因为载入免费。
    for rate in (1.0e9, CUBE_RATE, 2.0e6):       # 载入绑定 / 接近交点 / 计算绑定
        # 拆相位只在静态绑定下可表达, 所以闭式那边也钉核, 两边同口径
        base = _run(cube_rate=rate,
                    options=m.ModelOptions(late_bind_pools=STATIC))
        split = _run(cube_rate=rate, options=_split())
        # 1% 容差: 拆相位后提交时序略有差异, 方向上不应系统性变慢
        assert split["kernel_total_us"] <= base["kernel_total_us"] * 1.01
        for stage in ("gmm1", "gmm2"):
            assert abs(split["rank_results"][0]["stage_busy_us"][stage]
                       - base["rank_results"][0]["stage_busy_us"][stage]) < 1e-6
    phases = {str(e.meta.get("phase")) for e in split["rank_results"][0]["events"]}
    assert {"grant", "load", "cube", "fix"} <= phases


def test_phase_split_compute_bound():
    """计算主导 (cube > load) 时必须变慢.

    速率取 2.0e6: 本夹具下 cube 合计 7965.0 核·us 对载入 4835.6, 比值 1.65, 计算确实
    主导。原先取 6.75e6 (规格 fp16 速率) 并要求 +20us —— 那是载入免费时代的标定:
    载入占住 MTE2 之后 6.75e6 的比值只有 0.49, 计算根本不主导, 墙钟只动 3.9us。
    """
    sub = _run(options=_split())
    over = _run(cube_rate=2.0e6, options=_split())    # 两边都钉核 (见 STATIC)
    assert over["kernel_total_us"] > sub["kernel_total_us"] + 20.0


def test_single_l1_buffer_conflicts_with_deep_mte_queue():
    import pytest
    with pytest.raises(ValueError, match="l1_buf_num=1"):
        _run(kernel=m.KernelConfig(l1_buf_num=1), options=_split())


def test_custom_gmm1_callable_cannot_be_split():
    """自定义 gmm1_tile 没有 A流/计算 分解: 拆相位时报错, 不静默."""
    import pytest
    analytical = m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE)
    costs = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=lambda rows, k, cols: 5.0,
        gmm2_tile=analytical.gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        count_table_prepare_us=m.T_COUNT_GATE)
    kw = dict(routing_counts=_deterministic_case(), token_num_per_rank=64, h=6144,
              hidden_dim=4096, aic_num=28, costs=costs, p1_override=2, p2_override=1)
    assert m.simulate_routing_counts(options=m.ModelOptions(pipeline=P()), **kw)
    with pytest.raises(ValueError, match="自定义 callable"):
        m.simulate_routing_counts(options=_split(), **kw)


def test_custom_gmm1_callable_takes_three_args():
    """未开 B 复用时 gmm1_tile 按三参调用, 自定义 callable 不必接 b_load."""
    analytical = m.AnalyticalGmmCosts()
    calls = []

    def custom(rows, k, cols):
        calls.append((rows, k, cols))
        return 5.0

    costs = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=custom, gmm2_tile=analytical.gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        count_table_prepare_us=m.T_COUNT_GATE)
    res = m.simulate_routing_counts(
        routing_counts=_deterministic_case(), token_num_per_rank=64, h=6144,
        hidden_dim=4096, aic_num=28, costs=costs, p1_override=2, p2_override=1)
    assert calls and res["kernel_total_us"] > 0


# ---- L2: 共享带宽信道 (HBM/L1/片间) ----


def test_parse_tiling_real_bin():
    import struct
    from tempfile import NamedTemporaryFile
    # 最小合法 tiling (仅测试字段)
    buf = bytearray(232)
    struct.pack_into("<10I", buf, 0, 64, 64, 6144, 4096, 4, 7, 2048, 8, 28, 56)
    struct.pack_into("<I", buf, 64, 1)
    struct.pack_into("<4i", buf, 80, 256, 1, 6, 6336)   # dispatchBufferConfig
    struct.pack_into("<I", buf, 204, 4)
    with NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(buf)
        path = f.name
    t = m.parse_tiling(path)
    Path(path).unlink()
    assert t["bs"] == 64 and t["h"] == 6144 and t["topk"] == 8
    assert t["dispatchBufferCount"] == 6
    assert t["mGroupsPerWave"] == 4


# ---- 运行时图重构: 空闲核任务转移 ----

def test_restructure_idle_core_stealing():
    """图结构随资源竞争运行时重构: 空闲核转移繁忙核尾部 tile,
    makespan 收敛, busy 守恒, 消费者语义 (同名注入) 保持."""
    from moe_cost_model.scheduler import Event, MultiResourceScheduler
    from moe_cost_model.analysis import idle_core_stealing

    evs = []
    for i in range(5):
        evs.append(Event(f"W0.gmm1.m0.n{i}.c0", ("AIC:0",), 10.0, order=i,
                         meta={"stage": "gmm1"},
                         acquires=(("Q:aic:0", 1),), releases=(("Q:aic:0", 1),)))
    evs.append(Event("W0.gmm1.m0.n5.c1", ("AIC:1",), 10.0, order=5,
                     meta={"stage": "gmm1"},
                     acquires=(("Q:aic:1", 1),), releases=(("Q:aic:1", 1),)))
    caps = {"Q:aic:0": 1, "Q:aic:1": 1}
    total_static, sched_static = MultiResourceScheduler().schedule(evs, capacities=caps)
    total_steal, sched_steal = MultiResourceScheduler().schedule(
        evs, capacities=caps, restructure=idle_core_stealing(min_pending=2))
    assert abs(total_static - 50.0) < 1e-9          # 静态: 5+1 → 50
    assert abs(total_steal - 30.0) < 1e-9           # 任务转移后 3+3 → 30
    busy_s = sum(e.end_us - e.start_us for e in sched_static)
    busy_d = sum(e.end_us - e.start_us for e in sched_steal)
    assert abs(busy_s - busy_d) < 1e-9              # busy 守恒
    # 同名注入: 事件集合不变 (转移的事件以同名换核复活)
    assert {e.name for e in sched_steal} == {e.name for e in evs}
    stolen = [e for e in sched_steal if e.meta.get("stolen_from")]
    assert len(stolen) >= 1 and stolen[0].resources[0] == "AIC:1"
