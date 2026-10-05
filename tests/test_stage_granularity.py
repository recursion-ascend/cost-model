"""事件粒度是五个 stage 共有的维度, 不是 combine 的特性 (缺口 12).

粒度 = 一个事件覆盖多少份该 stage 的自然工作单元。它与 tile 几何是两件事:
tile_m/tile_n 受 L1/L0C 容量约束 (物理), "一个事件覆盖几个 tile" 是同步点密度
与并行度的交换 (编排)。
"""
import collections

import pytest

import moe_cost_model as m
from moe_cost_model.builders.tiling import coalesce_tiles
from moe_cost_model.config.granularity import (DEFAULT_GRANULARITY, STAGES,
                                               GranularityAssignment,
                                               StageGranularity,
                                               resolve_granularity)
from moe_cost_model.planning.tile_grid import Tile
from linkutil import links

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


def _run(granularity=None, depth=2, aic_num=4, core_assignment=None):
    P = m.MEGAMOE_A8W8
    opt = P.with_options(links=links(depth),
                         **({"granularity": granularity} if granularity else {}))
    kw = P.shape_kw(options=opt, kernel=P.kernel, policy=P.policy)
    if core_assignment is not None:
        kw["core_assignment"] = core_assignment
    rc = tuple(tuple(tuple([8] * 2) for _ in range(2)) for _ in range(2))
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=16, h=6144, hidden_dim=4096,
        aic_num=aic_num, costs=_costs(), **kw)


def _stage_counts(res):
    ev = res["rank_results"][0]["events"]
    c = collections.Counter(e.meta.get("stage") for e in ev if e.meta)
    return {k: c[k] for k in ("gmm1", "activation", "gmm2", "combine")}


# ---------------------------------------------------------------- 类型与缺省

def test_every_stage_has_a_granularity():
    """五个 stage 一个不少 —— 这正是 2026-10-04 之前缺的那条对称性."""
    g = GranularityAssignment()
    assert set(STAGES) == {"dispatch", "gmm1", "activation", "gmm2", "combine"}
    for s in STAGES:
        assert isinstance(g.of(s), StageGranularity)


def test_default_is_finest_and_is_a_no_op():
    """缺省 = 最细 = 最少假设: 不预设任何攒批."""
    g = GranularityAssignment()
    assert g.coarser_than_default() == ()
    for s in ("gmm1", "activation", "gmm2", "combine"):
        assert g.items(s) == 1
    # dispatch 的 0 不表示"整片", 表示"沿用 tiling 的 routeItemsPerBatch"
    assert g.items("dispatch") == 0


def test_rejects_unknown_stage_and_negative():
    with pytest.raises(ValueError):
        StageGranularity("swiglu", 1)          # 正名是 activation
    with pytest.raises(ValueError):
        StageGranularity("gmm1", -1)
    with pytest.raises(ValueError):
        GranularityAssignment((StageGranularity("gmm1", 1),
                               StageGranularity("gmm1", 2)))


def test_resolve_accepts_mapping_and_sequence():
    a = resolve_granularity({"gmm2": 3})
    b = resolve_granularity((StageGranularity("gmm2", 3),))
    assert a.items("gmm2") == b.items("gmm2") == 3
    assert resolve_granularity(None).items("gmm1") == 1


# ------------------------------------------------------- 两个历史旋钮是视图

def test_combine_granularity_is_a_view_on_the_unified_field():
    """combine_granularity / dispatch_rows_per_item 折进 granularity, 只有一个真相."""
    assert m.ModelOptions().grain("combine") == 1
    assert m.ModelOptions(combine_granularity="per_expert").grain("combine") == 0
    # 反向也成立: 从统一字段给, 视图跟着走
    o = m.ModelOptions(granularity={"combine": 0})
    assert o.combine_granularity == "per_expert"
    o = m.ModelOptions(dispatch_rows_per_item=64)
    assert o.grain("dispatch") == 64
    o = m.ModelOptions(granularity={"dispatch": 64})
    assert o.dispatch_rows_per_item == 64


def test_two_views_that_contradict_are_rejected():
    with pytest.raises(ValueError, match="矛盾"):
        m.ModelOptions(combine_granularity="per_expert", granularity={"combine": 3})


# ------------------------------------------------------------ 合并规则是物理

def test_coalesce_only_merges_same_rows_and_adjacent_cols():
    """行必须相同、列必须相邻 —— 下游按连续行列区间挑依赖, 别的并集表达不出来."""
    ts = ([Tile(0, 256, c, c + 256) for c in (0, 256, 512)]
          + [Tile(256, 512, 0, 256)])            # 换行 -> 必须截断
    got = coalesce_tiles(ts, 3)
    assert [(t.row_begin, t.col_begin, t.col_end, len(k)) for t, k in got] == [
        (0, 0, 768, 3), (256, 0, 256, 1)]
    # 列有缺口也截断
    gap = [Tile(0, 256, 0, 256), Tile(0, 256, 512, 768)]
    assert [len(k) for _, k in coalesce_tiles(gap, 2)] == [1, 1]


def test_coalesce_at_one_is_byte_identical_to_no_coalescing():
    ts = [Tile(0, 256, c, c + 256) for c in (0, 256, 512)]
    assert [(t, k) for t, k in coalesce_tiles(ts, 1)] == [(t, [t]) for t in ts]


def test_granularity_zero_means_whole_run():
    ts = [Tile(0, 256, c, c + 256) for c in range(0, 5 * 256, 256)]
    merged, members = coalesce_tiles(ts, 0)[0]
    assert len(members) == 5 and merged.col_end == 5 * 256


# ------------------------------------------------------------ 端到端: 真的变了

def test_default_granularity_changes_nothing():
    base = _stage_counts(_run())
    assert _stage_counts(_run({"gmm1": 1, "activation": 1, "gmm2": 1,
                               "combine": 1})) == base


def test_coarser_gmm1_and_gmm2_fold_events():
    base = _stage_counts(_run())
    assert _stage_counts(_run({"gmm1": 2}))["gmm1"] < base["gmm1"]
    assert _stage_counts(_run({"gmm2": 2}))["gmm2"] < base["gmm2"]


def test_coarser_combine_folds_events_and_does_not_need_the_same_core():
    """combine 从 GM 读 GMM2 的输出, 与 GMM2 同核不是物理约束 —— 所以攒批不按核分组.

    按核分组会让这个旋钮失效: 轮转/晚绑定下同一个核拿到的 n-tile 不相邻。
    """
    base = _stage_counts(_run())
    assert _stage_counts(_run({"combine": 2}))["combine"] < base["combine"]
    assert _stage_counts(_run({"combine": 0}))["combine"] < base["combine"]


def test_act_granularity_needs_the_feeding_gmm1_tiles_on_one_core():
    """ACT 的粒度不是自由旋钮: ACT 必须与产它的 GMM1 同核 (L0C->UB 的 Fixpipe).

    于是 g>1 只在"喂它的 GMM1 tile 既同核又 n 相邻"时才生效。轮转分核把相邻
    n-tile 散到不同核, 这时 g>1 是**空操作** —— 这是物理与分核策略的耦合, 不是 bug,
    所以这里把两种分核策略下的差别钉住。
    """
    rr = _stage_counts(_run({"activation": 2}))
    assert rr["activation"] == _stage_counts(_run())["activation"]   # 空操作
    cb = _stage_counts(_run({"activation": 2},
                            core_assignment=m.ContiguousBlock()))
    assert cb["activation"] < rr["activation"]                        # 真的合并了
    res = _run({"activation": 2}, core_assignment=m.ContiguousBlock())
    acts = [e for e in res["rank_results"][0]["events"]
            if e.meta and e.meta.get("stage") == "activation"]
    assert all(e.meta["gmm1_events_in_event"] == 2 for e in acts)


def test_act_granularity_beyond_ub_depth_is_a_deadlock_and_is_refused():
    """g 个 GMM1 各占一个 UB 槽, 要等齐才发 ACT -> depth 必须 >= g (或 0 = 不设限)."""
    with pytest.raises(ValueError, match="死锁"):
        _run({"activation": 2}, depth=1)
    _run({"activation": 2}, depth=0)        # 不设限: 允许


def test_coarse_granularity_costs_parallelism_not_just_saves_sync():
    """粗粒度不是免费的: 一个事件只能落一个核, 项数少于核数就有核闲着.

    这正是这个旋钮要让算子工程师看见的交换 —— 所以模型必须能算出它变差。
    """
    fine = _run(aic_num=28)["kernel_total_us"]
    coarse = _run({"gmm1": 4, "gmm2": 4}, aic_num=28)["kernel_total_us"]
    assert coarse > fine


def test_profile_declares_its_own_granularity():
    """参考实现的粒度是它的选择, 要显式声明, 不能靠"缺省恰好等于它"."""
    g = m.MEGAMOE_A8W8.options.granularity
    assert g.items("gmm1") == 1 and g.items("activation") == 1
    assert g.items("gmm2") == 1 and g.items("combine") == 1


# ------------------------------------------------------ 场景文件也要能给粒度

def test_scenario_file_can_set_granularity_per_stage():
    """日常路径是场景文件 —— 旋钮必须在 toml 里写得出, 否则等于没有."""
    from pathlib import Path

    import moe_cost_model as mm
    base = mm.load_scenario(Path("examples/scenario_basic.toml"))
    fine = mm.simulate(base)["kernel_total_us"]
    coarse = mm.simulate(
        base.with_overrides({"options.granularity": {"gmm2": 2}}))["kernel_total_us"]
    assert coarse != fine
    # 这个形状 tile 数远多于核数, 粗粒度是**收益** —— 与 28 核夹具上全部变慢相反,
    # 正好说明这个旋钮必须扫, 不能照搬取值。
    assert coarse < fine


def test_scenario_rejects_non_integer_and_unknown_stage():
    from pathlib import Path

    import moe_cost_model as mm
    base = mm.load_scenario(Path("examples/scenario_basic.toml"))
    with pytest.raises(ValueError, match="应为整数"):
        base.with_overrides({"options.granularity": {"gmm2": 2.5}})
    with pytest.raises(ValueError, match="未知 stage"):
        base.with_overrides({"options.granularity": {"swiglu": 2}})


def test_scenario_file_can_set_string_tuple_orchestration_knobs():
    """late_bind_pools / barriers 也是编排旋钮, 场景文件里写不出等于没有.

    空数组 [] 表示关掉 —— 这是"静态发牌"与"不加栅栏"的写法。
    """
    from pathlib import Path

    import moe_cost_model as mm
    base = mm.load_scenario(Path("examples/scenario_basic.toml"))
    off = base.with_overrides({"options.late_bind_pools": []})
    assert off.options.late_bind_pools == ()
    on = base.with_overrides({"options.late_bind_pools": ["AIC"]})
    assert on.options.late_bind_pools == ("AIC",)
    assert base.with_overrides({"options.barriers": ["wave"]}).options.barriers == ("wave",)
    with pytest.raises(ValueError, match="应为字符串数组"):
        base.with_overrides({"options.late_bind_pools": "AIC"})
    with pytest.raises(ValueError, match="每一项应为字符串"):
        base.with_overrides({"options.barriers": [2]})
