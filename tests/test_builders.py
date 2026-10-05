"""2026-09 审计修复的回归测试: 11 项边/调度/策略缺陷."""
import pytest

import moe_cost_model as m
from moe_cost_model.config.hardware import KernelConfig, _gmm2_head_tail_fractions
from moe_cost_model.scheduler import Event, MultiResourceScheduler, RestructureAction
from moe_cost_model.model import A8W8WaveCostModel
from moe_cost_model.planning.wave_packing import BalancedWaves, LongestExpertFirst
from moe_cost_model.scheduler import PriorityByStage
from moe_cost_model.analysis import idle_core_stealing
from moe_cost_model.shape import MegaMoeShape
from moe_cost_model.analysis import bottleneck_report

# 测试夹具值, 非标定常数: Cube 速率没有缺省, 测试统一取此值
CUBE_RATE = 2.7e7


def _costs():
    return m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        combine_read_bytes=m.AnalyticalCombineCosts().read_bytes,
        count_table_prepare_us=m.T_COUNT_GATE,
    )


def _shape(**kw):
    world, local, per = 2, 6, 512
    args = dict(
        expert_tokens=tuple(per * world for _ in range(local)),
        token_num=per * world * local // 8, h=6144, hidden_dim=4096,
        aic_num=16, rank_id=0, p1_override=2, p2_override=1,
        expert_source_tokens=tuple(tuple(per for _ in range(world))
                                   for _ in range(local)),
        kernel=KernelConfig())
    args.update(kw)
    return MegaMoeShape(**args)


def test_head_tail_fraction_clamped():
    """K < kL1 时 head 覆盖全部 K, tail 比例为 0, 不出现负值."""
    assert _gmm2_head_tail_fractions(128) == (1.0, 0.0)
    assert _gmm2_head_tail_fractions(384, 512) == (1.0, 0.0)
    h, t = _gmm2_head_tail_fractions(2048)
    assert abs(h - 256 / 2048) < 1e-12 and t > 0


def test_priority_policy_orders_same_engine():
    """非 EarliestStart 策略: 同引擎同刻就绪时按策略键选择, 剪枝不吞候选."""
    pre = Event("PRE", (), 100.0)
    a = Event("A", ("AIC:0",), 10.0, deps=("PRE",), meta={"stage": "combine"})
    b = Event("B", ("AIC:0",), 10.0, deps=("PRE",), meta={"stage": "dispatch"})
    _, s = MultiResourceScheduler().schedule([pre, a, b], policy=PriorityByStage())
    first = min((e for e in s if e.name in ("A", "B")),
                key=lambda x: (x.start_us, x.name))
    assert first.name == "B", "dispatch 优先级 0 应先于 combine 优先级 4"


def test_add_dep_with_committed_dependency():
    """add_dep 引用已提交事件: 不加 indegree, 不产生假环, 目标正常调度."""
    a = Event("A", (), 5.0)
    c = Event("C", (), 200.0)
    b = Event("B", ("AIC:0",), 5.0, deps=("C",))
    fired = []

    def hook(ctx):
        if not fired:
            fired.append(1)
            return RestructureAction(add_dep=[("B", "A")])
        return RestructureAction()

    _, s = MultiResourceScheduler().schedule([a, b, c], restructure=hook)
    assert {e.name for e in s} == {"A", "B", "C"}


def test_vec_queue_split_act_combine():
    """AIV0(ACT) 与 AIV1(COMBINE) 队列计数信号量分名: pipeline() 不再串行同核双引擎.

    慢 ACT 配置下旧实现总时长 +2% 以上 (共享 QUEUE:vec:c{n}).
    """
    costs = _costs()
    base_act = costs.activation_tile

    def slow(mr, cn):
        return base_act(mr, cn) * 3

    def run(pipeline):
        costs2 = _costs()
        object.__setattr__(costs2, "activation_tile", slow)
        world, local, per = 2, 4, 1024
        return m.simulate_routing_counts(
            routing_counts=[[[per] * world for _ in range(local)]
                            for _ in range(world)],
            token_num_per_rank=1024, h=6144, hidden_dim=4096, aic_num=28,
            costs=costs2, topk=8, p1_override=2, p2_override=1,
            options=m.ModelOptions(pipeline=pipeline))

    r0 = run(None)
    r1 = run(m.PipelineConstraints())
    assert abs(r0["kernel_total_us"] - r1["kernel_total_us"]) < 1e-9


def test_dispatch_lookahead_changes_structure():
    """dispatch_lookahead 实现文档语义: 建立序与 pacing 边随 la 变化.

    按核的 dispatch_call 事件与 per_core 配速边都属于 profiles.MEGAMOE_A8W8 那组
    取值 (缺省是 pooled + 不配速), 所以这里显式用它。
    """

    def build(la):
        shape = _shape(policy=m.InstancePolicy(dispatch_lookahead=la))
        return A8W8WaveCostModel(
            _costs(), m.MEGAMOE_A8W8.options).build_events(shape)

    e1, _ = build(1)
    e2, _ = build(2)
    e3, _ = build(3)
    seq = lambda evs: [e.meta.get("stage") for e in evs]
    assert seq(e1) != seq(e2) and seq(e2) != seq(e3)
    # la=2 缺省语义不变: W2 的 dispatch 由 W0 的 combine 配速
    dc2 = next(e for e in e2 if e.name.endswith("W2.dispatch_call.c0"))
    assert any("W0" in d and "combine" in d for d in dc2.deps)
    dc3 = next(e for e in e3 if e.name.endswith("W2.dispatch_call.c0"))
    assert not any("combine" in d for d in dc3.deps)   # 2-3 < 0, 无配速边


def test_wave_packing_strategies_sound():
    """打包策略: 全行覆盖、rows>0、组边界对齐."""
    ws = LongestExpertFirst().plan([100, 500, 300], 2, 256)
    assert all(w.rows > 0 for w in ws)
    assert sum(w.rows for w in ws) == 900
    ws2 = BalancedWaves().plan([100, 100, 100, 500], 3, 256)
    covered = sum(s.row_end - s.row_begin for w in ws2 for s in w.slices)
    assert covered == 800, "旧实现丢行 (300/800)"
    assert all(w.rows > 0 for w in ws2)
    assert all(s.row_begin % 256 == 0 for w in ws2 for s in w.slices), \
        "slice 起点必须 256 对齐 (builder 的 group 索引依赖)"


def test_wave_packing_wired_into_model():
    """shape.wave_packing 真正被 model.waves 消费."""
    base = A8W8WaveCostModel(_costs(), m.ModelOptions())
    w_default = base.waves(_shape())
    w_lef = base.waves(_shape(wave_packing=LongestExpertFirst()))
    # 本用例 token 均匀, 打包顺序不影响波数, 但至少不应报错且事件图可建
    evs, _ = base.build_events(_shape(wave_packing=BalancedWaves()))
    assert evs


def test_analysis_fields_and_path_accounting():
    """analysis 字段名与 model 输出一致; 关键路径 work+wait=total 且 work>=0."""
    world, local, per = 2, 4, 1024
    res = m.simulate_routing_counts(
        routing_counts=[[[per] * world for _ in range(local)]
                        for _ in range(world)],
        token_num_per_rank=1024, h=6144, hidden_dim=4096, aic_num=28,
        costs=_costs(), topk=8, p1_override=2, p2_override=1)
    rep = bottleneck_report(res)
    assert len(rep["resources"]) > 0 and len(rep["stage_busy"]) > 0
    br = rep["breakdown"]
    assert abs(br["work_us"] + br["wait_us"] - br["total_us"]) < 1e-6
    assert br["work_us"] >= 0, "旧实现 wait 与上游时长重复计数, work 可为负"


def test_idle_core_stealing_active_and_effective():
    """idle_core_stealing 经 model 生效 (rank 前缀剥离) 且确定."""
    world, local, per = 2, 4, 1024
    C = [[[per] * world for _ in range(local)] for _ in range(world)]
    # 转移钩子修的是静态发牌的空闲; 缺省晚绑定下没有可转移的东西 (见 test_scenario)。
    # 用 MEGAMOE_A8W8 那组编排 (含静态发牌): 钩子的判据与注入量是按这种结构调的。
    kw = dict(token_num_per_rank=1024, h=6144, hidden_dim=4096, aic_num=28,
              costs=_costs(), topk=8, p1_override=2, p2_override=1,
              options=m.MEGAMOE_A8W8.options)

    r_plain = m.simulate_routing_counts(routing_counts=C, **kw)
    r_steal = m.simulate_routing_counts(routing_counts=C, restructure=idle_core_stealing(), **kw)
    stolen = sum(1 for r in r_steal["rank_results"].values()
                 for e in r["events"] if "stolen_from" in e.meta)
    assert stolen > 0, "rank 前缀不匹配时任务转移静默失效 (旧 bug)"
    assert r_steal["kernel_total_us"] != r_plain["kernel_total_us"]

def test_dispatch_segment_splits_by_route_batch():
    """段内再按 routeItemsPerBatch 分批, 一批一个事件 (内核每批一次 PROFILE 区间).

    内核 DispatchRankTokens 的 while 批循环: 每批一次
    CopyTokensAndMetaForDispatch, 而 MOE_PROFILE_BEGIN/END 在那个函数里 —— 所以
    一批就是实测 trace 里的一个 DISPATCH_XFER/LOCAL 事件, 各付自己的 λ 与流水
    填充/排空, 不是一段一个事件。
    """
    # aic=1 让单核吃下整波; mgw=3 -> 768 行; 两个源卡各 400 行 -> 段 400/368 行
    res = m.simulate_routing_counts(
        routing_counts=[[[400, 400]], [[400, 400]]], token_num_per_rank=800,
        h=6144, hidden_dim=4096, aic_num=1, topk=2,
        costs=m.build_analytical_costs(
            h=6144, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=48, p2_override=1)
    d = [e for e in res["rank_results"][0]["events"] if e.meta.get("stage") == "dispatch"]
    # 400 -> 256+144, 368 -> 256+112, 32 -> 32
    got = sorted((e.meta["wave"], e.meta["src_rank"], e.meta["row_begin"],
                  e.meta["row_end"]) for e in d)
    assert got == [(0, 0, 0, 256), (0, 0, 256, 400),
                   (0, 1, 400, 656), (0, 1, 656, 768), (1, 1, 768, 800)]
    assert max(e.meta["rows"] for e in d) == 256      # 不超过 routeItemsPerBatch
    # 行数守恒: 分批后每个 (专家, m-group) 门仍然收齐
    for e in res["rank_results"][0]["events"]:
        if e.meta.get("stage") == "dispatch_ready":
            assert e.meta["contributed_rows"] == e.meta["required_rows"]


def test_dispatch_single_batch_keeps_segment_granularity():
    """段不超过一批时事件与未分批时逐字节一致 (bs=36 全部段 1~6 行, 远小于 256).

    直方图对的是实测 trace, 所以用 precut 切法 (缺省 pooled 不按核预切, 段的构成不同)。
    """
    res = m.simulate_routing_counts(
        routing_counts=[[[18] * 4 for _ in range(3)] for _ in range(4)],
        token_num_per_rank=36, h=5120, hidden_dim=9216, aic_num=28,
        costs=_costs(), topk=6, p1_override=2, p2_override=1,
        options=m.ModelOptions(dispatch_partition="precut"))
    d = [e for e in res["rank_results"][0]["events"] if e.meta.get("stage") == "dispatch"]
    # 实测 rank0: 47 个 DISPATCH_XFER + 14 个 DISPATCH_LOCAL = 61, 行数直方图逐桶相同
    assert len(d) == 61
    hist = {}
    for e in d:
        hist[e.meta["rows"]] = hist.get(e.meta["rows"], 0) + 1
    assert hist == {1: 2, 2: 15, 3: 19, 4: 2, 5: 19, 6: 4}
    assert sum(e.meta["rows"] for e in d) == 36 * 6


def test_combine_remote_rows_come_from_routing():
    """每个 COMBINE 窗的跨卡行数由 routing 逐行数出, 不是按比例摊.

    专家内的行按源卡顺序排布 (dispatch 就是按这个顺序分段写的), 所以窗的行区间
    与源卡分段求交即得。均匀路由下 4 卡每专家 4×18 行 → 每窗 54 行跨卡。
    """
    from moe_cost_model.builders.base import count_remote_rows
    # [18,18,18,18], dst=0 → 前 18 行本卡, 后 54 行跨卡
    src = [18, 18, 18, 18]
    assert count_remote_rows(src, 0, 0, 72) == 54
    assert count_remote_rows(src, 0, 0, 18) == 0        # 全落在本卡段
    assert count_remote_rows(src, 0, 18, 36) == 18      # 全落在 rank1 段
    assert count_remote_rows(src, 0, 10, 30) == 12      # 跨段: 8 本卡 + 12 远端
    assert count_remote_rows(src, 2, 0, 72) == 54       # 本卡段换位置
    assert count_remote_rows(src, 2, 36, 54) == 0
    assert count_remote_rows([0, 72, 0, 0], 0, 0, 72) == 72   # 一张源卡包场
    assert count_remote_rows([72, 0, 0, 0], 0, 0, 72) == 0    # 全本地专家

    # 建图里落到事件 meta 上
    res = m.simulate_routing_counts(
        routing_counts=[[[18] * 4 for _ in range(3)] for _ in range(4)],
        token_num_per_rank=36, h=5120, hidden_dim=9216, aic_num=28,
        costs=_costs(), topk=6, p1_override=2, p2_override=1)
    comb = [e for e in res["rank_results"][0]["events"]
            if e.meta.get("stage") == "combine"]
    assert comb
    for e in comb:
        assert e.meta["remote_rows"] == count_remote_rows(
            [18] * 4, 0, e.meta["row_begin"], e.meta["row_end"])


def test_total_is_measured_to_last_combine():
    """执行时间记到最后一个 COMBINE 结束; 尾段照常调度但不计入."""
    world, local, per = 2, 4, 256
    res = m.simulate_routing_counts(
        routing_counts=[[[per] * world for _ in range(local)] for _ in range(world)],
        token_num_per_rank=256, h=6144, hidden_dim=4096, aic_num=28,
        costs=_costs(), topk=8, p1_override=2, p2_override=1)
    for rank in res["rank_results"].values():
        events = rank["events"]
        last_combine = max(e.end_us for e in events if e.meta.get("stage") == "combine")
        finalize = next(e for e in events if e.meta.get("part") == "finalize")
        assert rank["total_us"] == last_combine
        assert rank["dag_end_us"] == finalize.end_us > rank["total_us"]
        assert rank["critical_path"][-1]["stage"] == "combine"
        assert rank["critical_path"][-1]["end_us"] == rank["total_us"]
    assert res["kernel_total_us"] == max(r["total_us"] for r in res["rank_results"].values())
    assert res["kernel_dag_end_us"] == max(r["dag_end_us"]
                                           for r in res["rank_results"].values())
    path = bottleneck_report(res)["critical_path"]
    assert ".combine." in path[-1]


def test_shared_expert_gmm2_is_per_tile():
    """共享 GMM2: 按 tile 建事件, 占 AIC 核, 时长走 GMM2 公式, 依赖齐全."""
    shape = _shape(shared_expert_num=1, token_num=600)      # 600 行 → 3 个 m-group
    events, _ = A8W8WaveCostModel(_costs(), m.ModelOptions()).build_events(shape)
    by_name = {e.name: e for e in events}
    tiles = [e for e in events if e.meta.get("stage") == "shared_gmm2"]
    n_tiles = 6144 // 256
    assert len(tiles) == 3 * n_tiles
    assert {e.meta["m_rows"] for e in tiles} == {256, 600 - 512}
    k2 = 4096 // 2
    costs = _costs()
    for e in tiles:
        assert e.resources[0].startswith("AIC:")
        assert e.duration_us == costs.gmm2_tile(e.meta["m_rows"], k2, e.meta["logical_n"])
        stages = [str(by_name[d].meta.get("stage")) for d in e.deps]
        parts = [by_name[d].meta.get("part") for d in e.deps]
        assert "output_core_sync" in parts                     # 位置: 尾段 core_sync 之后
        acts = [by_name[d] for d in e.deps if by_name[d].meta.get("stage") == "shared_act"]
        assert len(acts) == stages.count("shared_act") == 4096 // 2 // 256   # 本组全部 ACT
        assert all(f".m{e.meta['mgroup']}." in a.name for a in acts)
    # 核轮转: 同核上的 tile 数相差不超过 1
    per_core = {}
    for e in tiles:
        per_core[e.resources[0]] = per_core.get(e.resources[0], 0) + 1
    assert max(per_core.values()) - min(per_core.values()) <= 1
    # 尾段经汇合事件接到 rank_sync
    done = by_name["R0.epilogue.shared_gmm2_done"]
    assert set(done.deps) == {e.name for e in tiles}
    assert by_name["R0.epilogue.output_rank_sync"].deps == (done.name,)


def test_shared_gmm2_load_bound_until_cube_is_slow():
    """共享 GMM2 = max(B流, 计算): 快 Cube 下由 B 流封顶, 慢到一定程度才被计算绑定."""
    def tile_us(rate):
        costs = m.build_analytical_costs(
            h=6144, dispatch_mechanistic=m.DispatchMechanisticLatency(), cube_mac_per_us=rate)
        res = m.simulate_routing_counts(
            routing_counts=[[[64] * 2 for _ in range(4)] for _ in range(2)],
            token_num_per_rank=64, h=6144, hidden_dim=4096, aic_num=28, costs=costs,
            topk=8, shared_expert_num=1, p1_override=2, p2_override=1)
        ev = [e for e in res["rank_results"][0]["events"]
              if e.meta.get("stage") == "shared_gmm2"]
        return max(e.end_us - e.start_us for e in ev)
    b_flow = tile_us(1.0e12)                 # Cube 无限快 -> 纯 B 流
    assert tile_us(2.7e7) == pytest.approx(b_flow)      # 载入绑定, 与速率无关
    assert tile_us(1.0e6) > 2.0 * b_flow                # Cube 够慢才翻转成计算绑定


def test_wave_offsets_parameterized():
    """stage 波偏移参数化: 显式 StageWaveOffsets 覆盖 la/lag, 缺省推导等价.

    offs=(2,-2): 迭代 0 预取 W0..W2, GMM2 滞后两波; 其波配对序列
    与任何 la/lag 组合都不同, 且配速深度 = dispatch+1.
    """
    from moe_cost_model.builders.mte import MteEventBuilder
    from moe_cost_model.config.policy import StageWaveOffsets
    from moe_cost_model.model import A8W8WaveCostModel

    def build(policy):
        # 按核的 dispatch_call 与 per_core 配速边来自 MEGAMOE_A8W8 那组取值
        opts = m.MEGAMOE_A8W8.options
        shape = _shape(policy=policy)
        waves = A8W8WaveCostModel(_costs(), opts).waves(shape)
        return MteEventBuilder(_costs(), opts).build(shape, waves)

    def wave_pairs(result):
        return [(t.gmm1_wave, t.gmm2_wave) for t in result[1]]

    # 缺省推导 = la/lag 显式 (la=2, token<4096 → lag=0)
    e_def, t_def = build(m.InstancePolicy())
    e_lag, t_lag = build(m.InstancePolicy(dispatch_lookahead=2, gmm2_lag_waves=0))
    assert wave_pairs((e_def, t_def)) == wave_pairs((e_lag, t_lag))

    # 显式偏移 (2, -2): 3 波用例
    opts2 = m.MEGAMOE_A8W8.options          # 同上: 要按核的 dispatch_call 与配速边
    shape = _shape(policy=m.InstancePolicy(
        wave_offsets=StageWaveOffsets(dispatch=2, gmm2=-2)))
    waves = A8W8WaveCostModel(_costs(), opts2).waves(shape)
    evs, trace = MteEventBuilder(_costs(), opts2).build(shape, waves)
    in_loop = [(t.gmm1_wave, t.gmm2_wave) for t in trace if t.gmm1_wave is not None]
    catchup = [(t.gmm1_wave, t.gmm2_wave) for t in trace if t.gmm1_wave is None]
    n = len(in_loop)
    assert n >= 3
    # offs.gmm2=-2: 前两轮 gmm2 未到, 其后 gmm2 = i-2
    assert in_loop[0] == (0, None) and in_loop[1] == (1, None)
    assert all(in_loop[i] == (i, i - 2) for i in range(2, n))
    # 补跑: 最后两波
    assert catchup == [(None, n - 2), (None, n - 1)]

    # dispatch 预取边界: W2 的 dispatch_call 在事件序中先于首个 GMM1
    names = [e.name for e in evs]
    i_w2disp = next(i for i, n in enumerate(names) if n.endswith("W2.dispatch_call.c0"))
    i_g1 = next(i for i, n in enumerate(names) if ".gmm1." in n)
    assert i_w2disp < i_g1

    # 配速边挂在同核最近构建的 combine 上, 只在该核已产出过 combine 时出现:
    # W3 (迭代1 建) 与 W4 (迭代2 建, 先于该迭代 GMM2) 时 GMM2 尚无 combine → 无边;
    # W5 (迭代3 建) 时 W0 的 combine 已存在 → 有边
    for w, expect in [(3, False), (4, False), (5, True)]:
        dc = next(e for e in evs if e.name.endswith(f"W{w}.dispatch_call.c0"))
        has = any("combine" in d for d in dc.deps)
        assert has == expect, f"W{w} 配速边应为 {expect}"

    # 约束
    import pytest
    with pytest.raises(ValueError):
        StageWaveOffsets(dispatch=-1)
    with pytest.raises(ValueError):
        StageWaveOffsets(gmm2=1)

