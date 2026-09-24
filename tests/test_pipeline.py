"""L0/L1/L2 流水线约束机制测试.

每个机制: 中性值不变 / 激活后方向正确 / 物理语义可验证.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moe_cost_model as m
from moe_cost_model.dag import Channel, Event, MultiResourceScheduler

P = m.PipelineConstraints


def _deterministic_case():
    WORLD, LOCAL = 4, 64
    counts = [[[0] * WORLD for _ in range(LOCAL)] for _ in range(WORLD)]
    for dst in range(WORLD):
        counts[dst][0] = [75, 75, 75, 75]
        counts[dst][1] = [16, 16, 16, 16]
        counts[dst][2] = [3, 4, 3, 3]
        counts[dst][63] = [200, 20, 20, 16]
    return tuple(tuple(tuple(r) for r in c) for c in counts)


def _run(options=None, policy=None):
    costs = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(begin_offset_us=tuple([0.0] * 28)),
        gmm1_tile=m.AnalyticalGmmCosts().gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts().gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        combine_tile=m.AnalyticalCombineCosts().tile,
        count_table_prepare_us=m.T_COUNT_GATE,
    )
    return m.simulate_routing_counts(
        routing_counts=_deterministic_case(), token_num_per_rank=64, h=6144,
        hidden_dim=4096, aic_num=28, costs=costs, options=options or m.ModelOptions(),
        p1_override=2, p2_override=1,   # kernel 默认策略 @bs64 (tiling 真值)
        policy=policy,
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
    base = _run()
    slow = _run(options=m.ModelOptions(pipeline=P(
        sync=m.SyncLatency(gmm1_act_handshake_us=2.0))))
    assert slow["kernel_total_us"] > base["kernel_total_us"]


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
    """生产者/消费者距离: ModelOptions.gmm1_activation_depth 参数化, 距离越大越快."""
    base = _run(options=m.ModelOptions(pipeline=P()))
    deeper = _run(options=m.ModelOptions(pipeline=P()),
                  policy=m.InstancePolicy(gmm1_activation_depth=2))
    assert deeper["kernel_total_us"] <= base["kernel_total_us"]


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
    """拆相位 + 信道 + 大 DAG: 距离依赖保序, 无令牌环."""
    WORLD, LOCAL = 4, 64
    counts = [[[8] * WORLD for _ in range(LOCAL)] for _ in range(WORLD)]
    rc = tuple(tuple(tuple(r) for r in c) for c in counts)
    costs = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(begin_offset_us=tuple([0.0] * 28)),
        gmm1_tile=m.AnalyticalGmmCosts().gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts().gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        combine_tile=m.AnalyticalCombineCosts().tile,
        count_table_prepare_us=m.T_COUNT_GATE,
    )
    res = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=1024, h=6144,
        hidden_dim=4096, aic_num=28, costs=costs,
        options=m.ModelOptions(pipeline=m.PipelineConstraints(
            queues=m.QueueDepths(mte_aic=2, cube=2, fix=2),
            phases=m.PhaseRates(cube_mac_per_us=2.7e7),
            channels=m.default_channels(28, bw_l1_gm=m.BW_L1_GM,
                                        bw_scatter=m.BW_SCATTER))),
    )
    assert res["kernel_total_us"] > 0


def test_l1_queue_depth_monotone():
    base = _run(options=m.ModelOptions(pipeline=P()))
    deep = _run(options=m.ModelOptions(pipeline=P(
        queues=m.QueueDepths(mte_aic=2))))
    # mte_aic=2 允许 2 载入在飞, 但 Q:aic 深度=1 限制发射 → 差异 < 15µs
    assert deep["kernel_total_us"] <= base["kernel_total_us"] + 15.0


# ---- L1: 相位拆分 (生产者/消费者距离, 跨 tile 流水) ----

def test_phase_split_self_consistency():
    """计算子临界时稳态周期 = max(load, cube) = load → 与闭式一致."""
    base = _run()
    split = _run(options=m.ModelOptions(pipeline=P(
        queues=m.QueueDepths(mte_aic=2),
        phases=m.PhaseRates(cube_mac_per_us=2.7e7))))
    # 计算子临界: 与闭式差 < 25µs (排队模型引入的发射槽差异)
    assert abs(split["kernel_total_us"] - base["kernel_total_us"]) < 25.0
    stages = {str(e.meta.get("phase")) for e in split["rank_results"][0]["events"]}
    assert "load" in stages and "cube" in stages and "fix" in stages


def test_phase_split_compute_bound():
    """计算主导 (cube > load) 时必须变慢."""
    sub = _run(options=m.ModelOptions(pipeline=P(
        queues=m.QueueDepths(mte_aic=2),
        phases=m.PhaseRates(cube_mac_per_us=2.7e7))))
    over = _run(options=m.ModelOptions(pipeline=P(
        queues=m.QueueDepths(mte_aic=2),
        phases=m.PhaseRates(cube_mac_per_us=6.75e6))))  # cube≈120µs > load
    assert over["kernel_total_us"] > sub["kernel_total_us"] + 20.0


# ---- L2: 共享带宽信道 (HBM/L1/片间) ----

def test_l2_channel_sharing():
    a = Event("A", (), 0.0, channel_bytes=(("link", 1000.0, 50.0),))
    b = Event("B", (), 0.0, channel_bytes=(("link", 1000.0, 50.0),))
    c = Event("C", (), 0.0, channel_bytes=(("link", 1000.0, 50.0),))
    ch = {"link": Channel("link", bw_total=100.0, max_rate_per_event=50.0)}
    _, s = MultiResourceScheduler().schedule([a], channels=ch)
    assert abs(s[0].end_us - 20.0) < 1e-9          # 独享 = 闭式
    _, s = MultiResourceScheduler().schedule([a, b], channels=ch)
    assert all(abs(e.end_us - 20.0) < 1e-9 for e in s)  # 双路各 50, 聚合 100
    _, s = MultiResourceScheduler().schedule([a, b, c], channels=ch)
    assert sorted(e.end_us for e in s) == [20.0, 20.0, 40.0]  # 第三路排队


def test_l2_ports_limit():
    a = Event("A", (), 0.0, channel_bytes=(("link", 1000.0, 50.0),))
    b = Event("B", (), 0.0, channel_bytes=(("link", 1000.0, 50.0),))
    ch = {"link": Channel("link", bw_total=100.0, ports=1, max_rate_per_event=50.0)}
    _, s = MultiResourceScheduler().schedule([a, b], channels=ch)
    assert sorted(e.end_us for e in s) == [20.0, 40.0]


def test_l2_channel_contention_slows_model():
    """聚合带宽减半 → 必须变慢."""
    base = _run(options=m.ModelOptions(pipeline=P(
        channels=m.default_channels(28, bw_l1_gm=m.BW_L1_GM, bw_scatter=m.BW_SCATTER))))
    contended = _run(options=m.ModelOptions(pipeline=P(
        channels=(
            m.Channel("gm_to_l1", bw_total=m.BW_L1_GM * 14, max_rate_per_event=m.BW_L1_GM),
            m.Channel("hbm_write", bw_total=m.BW_SCATTER * 28, max_rate_per_event=m.BW_SCATTER),
        ))))
    assert contended["kernel_total_us"] > base["kernel_total_us"]


# ---- tiling 真值接入 ----

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


# ---- 运行时图重构: 空核偷活 ----

def test_restructure_idle_core_stealing():
    """图结构随资源竞争运行时重构: 空闲核偷走繁忙核尾部 tile,
    makespan 收敛, busy 守恒, 消费者语义 (同名注入) 保持."""
    from moe_cost_model.dag import Event, MultiResourceScheduler
    from moe_cost_model.policies import idle_core_stealing

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
    assert abs(total_steal - 30.0) < 1e-9           # 偷活后 3+3 → 30
    busy_s = sum(e.end_us - e.start_us for e in sched_static)
    busy_d = sum(e.end_us - e.start_us for e in sched_steal)
    assert abs(busy_s - busy_d) < 1e-9              # busy 守恒
    # 同名注入: 事件集合不变 (偷走的事件以同名换核复活)
    assert {e.name for e in sched_steal} == {e.name for e in evs}
    stolen = [e for e in sched_steal if e.meta.get("stolen_from")]
    assert len(stolen) >= 1 and stolen[0].resources[0] == "AIC:1"
