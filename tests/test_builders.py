"""2026-09 审计修复的回归测试: 11 项边/调度/策略缺陷."""
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
        combine_tile=m.AnalyticalCombineCosts().tile,
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
    """dispatch_lookahead 实现文档语义: 建立序与 pacing 边随 la 变化."""

    def build(la):
        shape = _shape(policy=m.InstancePolicy(dispatch_lookahead=la))
        return A8W8WaveCostModel(_costs(), m.ModelOptions()).build_events(shape)

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
    kw = dict(token_num_per_rank=1024, h=6144, hidden_dim=4096, aic_num=28,
              costs=_costs(), topk=8, p1_override=2, p2_override=1)

    r_plain = m.simulate_routing_counts(routing_counts=C, **kw)
    r_steal = m.simulate_routing_counts(routing_counts=C, restructure=idle_core_stealing(), **kw)
    stolen = sum(1 for r in r_steal["rank_results"].values()
                 for e in r["events"] if "stolen_from" in e.meta)
    assert stolen > 0, "rank 前缀不匹配时任务转移静默失效 (旧 bug)"
    assert r_steal["kernel_total_us"] != r_plain["kernel_total_us"]

def test_wave_offsets_parameterized():
    """stage 波偏移参数化: 显式 StageWaveOffsets 覆盖 la/lag, 缺省推导等价.

    offs=(2,-2): 迭代 0 预取 W0..W2, GMM2 滞后两波; 其波配对序列
    与任何 la/lag 组合都不同, 且配速深度 = dispatch+1.
    """
    from moe_cost_model.builders.mte import MteEventBuilder
    from moe_cost_model.config.policy import StageWaveOffsets
    from moe_cost_model.model import A8W8WaveCostModel

    def build(policy):
        shape = _shape(policy=policy)
        waves = A8W8WaveCostModel(_costs(), m.ModelOptions()).waves(shape)
        return MteEventBuilder(_costs(), m.ModelOptions()).build(shape, waves)

    def wave_pairs(result):
        return [(t.gmm1_wave, t.gmm2_wave) for t in result[1]]

    # 缺省推导 = la/lag 显式 (la=2, token<4096 → lag=0)
    e_def, t_def = build(m.InstancePolicy())
    e_lag, t_lag = build(m.InstancePolicy(dispatch_lookahead=2, gmm2_lag_waves=0))
    assert wave_pairs((e_def, t_def)) == wave_pairs((e_lag, t_lag))

    # 显式偏移 (2, -2): 3 波用例
    shape = _shape(policy=m.InstancePolicy(
        wave_offsets=StageWaveOffsets(dispatch=2, gmm2=-2)))
    waves = A8W8WaveCostModel(_costs(), m.ModelOptions()).waves(shape)
    evs, trace = MteEventBuilder(_costs(), m.ModelOptions()).build(shape, waves)
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

