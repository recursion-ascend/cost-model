"""自定义切分: tile 网格策略 + 建图器注册.

覆盖 "工程师自己定义 megamoe 切分方式, 事件 DAG 自动重建" 这条路径。
"""
import collections
import dataclasses

import pytest

import moe_cost_model as m
from linkutil import links
from golden_cases import CUBE_RATE, SK_TOKENS, run_api, skewed_routing, uniform_routing

CAL = m.Calibration(cube_mac_per_us=CUBE_RATE)


def _scenario(**kw):
    return m.Scenario(
        workload=m.Workload(tokens=SK_TOKENS, routing="explicit", counts=skewed_routing()),
        p1_override=2, p2_override=1, calibration=CAL, **kw)


def _shape_for_grid():
    """给 resolve_grid 用的最小 shape (只读 tile_grid 字段)."""
    from moe_cost_model.shape import MegaMoeShape
    return MegaMoeShape(expert_tokens=(64,), token_num=64, h=6144, hidden_dim=4096,
                        aic_num=28)


def _stage_cores(res, stage, rank=0):
    return collections.Counter(
        int(e.resources[0].split(":")[1]) for e in res["rank_results"][rank]["events"]
        if e.meta.get("stage") == stage and e.resources)


# ---------------------------------------------------------------------------
# 网格: 缺省行主序 (最少假设) 与 swizzle (某实现的遍历序)
# ---------------------------------------------------------------------------

def test_default_grid_is_row_major():
    """缺省网格不声称任何遍历技巧: 行按 tile_m 分组、列按 tile_n 切, m 外 n 内."""
    km = m.KernelConfig()
    tiles = m.RowMajorTileGrid().plan(stage="gmm1", rows=600, cols=2048, kernel=km)
    assert [(t.row_begin, t.col_begin) for t in tiles] == [
        (mg * 256, nt * 256) for mg in range(3) for nt in range(8)]
    m.validate_tiles(tiles, rows=600, cols=2048, tile_m=km.tile_m)
    from moe_cost_model.builders.tiling import resolve_grid
    assert type(resolve_grid(dataclasses.replace(
        _shape_for_grid(), tile_grid=None))).__name__ == "RowMajorTileGrid"


def test_swizzled_grid_matches_kernel_geometry():
    """swizzle 网格 (profiles.MEGAMOE_A8W8 用它): 顺序与 swizzle_coord 一致."""
    from moe_cost_model.planning.waves import swizzle_coord
    km = m.KernelConfig()
    rows, cols = 600, 2048                      # 3 个 m-group x 8 个列块
    tiles = m.SwizzledTileGrid().plan(stage="gmm1", rows=rows, cols=cols, kernel=km)
    assert len(tiles) == 3 * 8
    for idx, t in enumerate(tiles):
        mg, nt = swizzle_coord(idx, 3, 8, km.swizzle_offset, km.swizzle_direction)
        assert (t.row_begin, t.col_begin) == (mg * 256, nt * 256)
        assert t.rows == min(256, rows - mg * 256) and t.cols == 256
    m.validate_tiles(tiles, rows=rows, cols=cols, tile_m=km.tile_m)


def test_explicit_default_grid_is_the_no_op():
    """显式给缺省网格与不给完全一致 (事件级)."""
    base = _scenario()
    explicit = _scenario(tile_grid="row_major")
    from golden_cases import fingerprint
    assert fingerprint(m.simulate(explicit)) == fingerprint(m.simulate(base))


# ---------------------------------------------------------------------------
# 换网格 → DAG 自动重建
# ---------------------------------------------------------------------------

def test_split_rows_fills_more_cores():
    """组内再切行: GMM1 tile 数翻倍, 原本空闲的核被用上, 总行数不变."""
    C = uniform_routing(8, 3, 18)               # 每专家 144 行 < tile_m, 每专家 1 组
    base = run_api(C, 72, topk=6, aic_num=28)
    split = run_api(C, 72, topk=6, aic_num=28,
                    tile_grid=m.SplitRowsTileGrid(parts=2))
    b, s = _stage_cores(base, "gmm1"), _stage_cores(split, "gmm1")
    assert sum(s.values()) == 2 * sum(b.values())
    assert len(b) < 28 and len(s) == 28          # 原先有核闲着, 切细后全用上
    # 行守恒: 每个专家的 GMM1 行数之和不变 (每个列块各覆盖一遍全部行)
    totals = []
    for res in (base, split):
        rows = collections.Counter()
        for e in res["rank_results"][0]["events"]:
            if e.meta.get("stage") == "gmm1":
                rows[e.meta["expert"]] += e.meta["m_rows"]
        assert len(set(rows.values())) == 1      # 三个专家一样
        totals.append(next(iter(rows.values())))
    assert totals[0] == totals[1]


def test_custom_grid_from_scenario_by_name():
    """注册自定义网格后, 场景文件按名字引用即可生效."""
    class ColHalves(m.TileGrid):
        def plan(self, *, stage, rows, cols, kernel):
            km = dataclasses.replace(kernel, tile_n=kernel.tile_n // 2)
            return m.SwizzledTileGrid().plan(stage=stage, rows=rows, cols=cols, kernel=km)

    m.register("tile_grid", "col_halves", ColHalves)
    base, fine = m.simulate(_scenario()), m.simulate(_scenario(tile_grid="col_halves"))
    assert sum(_stage_cores(fine, "gmm1").values()) == \
        2 * sum(_stage_cores(base, "gmm1").values())
    assert fine["kernel_total_us"] != base["kernel_total_us"]


def test_grid_with_constructor_args():
    sc = _scenario(tile_grid={"name": "split_rows", "parts": 3})
    assert sum(_stage_cores(m.simulate(sc), "gmm1").values()) == \
        3 * sum(_stage_cores(m.simulate(_scenario()), "gmm1").values())


def test_gmm2_follows_act_coverage_after_row_split():
    """行切细后 GMM2 仍只等与自己行范围相交、且覆盖整个 K 的 ACT."""
    from moe_cost_model.model import A8W8WaveCostModel
    from moe_cost_model.shape import MegaMoeShape
    from golden_cases import HIDDEN, H, manual_costs

    rows = uniform_routing(8, 3, 18)[0]
    shape = MegaMoeShape(
        expert_tokens=tuple(sum(r) for r in rows), token_num=72, h=H, hidden_dim=HIDDEN,
        aic_num=28, expert_source_tokens=rows, p1_override=2, p2_override=1, topk=6,
        kernel=m.KernelConfig(), tile_grid=m.SplitRowsTileGrid(parts=2))
    events, _ = A8W8WaveCostModel(
        manual_costs(),
        m.ModelOptions(links=links(readiness="first_chunk"))).build_events(shape)
    by_name = {e.name: e for e in events}
    k_gmm2 = HIDDEN // 2
    for e in events:
        if e.meta.get("stage") != "gmm2" or e.meta.get("part") != "tail":
            continue
        head = by_name[next(d for d in e.deps if d.endswith(".h"))]
        acts = [by_name[d] for d in (*e.deps, *head.deps)
                if by_name[d].meta.get("stage") == "activation"]
        assert acts, f"{e.name} 没有 ACT 依赖"
        for a in acts:
            assert a.meta["row_begin"] < e.meta["row_end"]
            assert a.meta["row_end"] > e.meta["row_begin"]
            assert a.meta["mgroup"] == e.meta["mgroup"]
        spans = sorted((a.meta["col_begin"], a.meta["col_end"]) for a in acts)
        cursor = 0
        for cb, ce in spans:
            assert cb <= cursor, f"{e.name} 的 ACT 在 K 的 {cursor} 处有缺口"
            cursor = max(cursor, ce)
        assert cursor == k_gmm2


# ---------------------------------------------------------------------------
# 网格校验
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tiles, fragment", [
    ([m.Tile(0, 100, 0, 128)], "只覆盖到 128"),                       # 列没铺满
    ([m.Tile(0, 50, 0, 256)], "只覆盖到 50"),                         # 行没铺满
    ([m.Tile(0, 100, 0, 256), m.Tile(0, 100, 128, 256)], "重叠"),     # 列重叠
    ([m.Tile(0, 100, 0, 128), m.Tile(0, 100, 200, 256)], "空缺"),     # 列有洞
    ([m.Tile(0, 300, 0, 256)], "跨越了 m-group 边界"),                # 行跨组
    ([], "tile 网格为空"),
])
def test_validate_tiles_rejects(tiles, fragment):
    with pytest.raises(ValueError, match=fragment):
        m.validate_tiles(tiles, rows=100 if tiles and tiles[0].row_end <= 100 else 300,
                         cols=256, tile_m=256, where="")


def test_bad_grid_is_caught_during_build():
    class Gappy(m.TileGrid):
        """行切法照旧, 但列只铺一半."""
        def plan(self, *, stage, rows, cols, kernel):
            return [m.Tile(t.row_begin, t.row_end, t.col_begin, t.col_end)
                    for t in m.SwizzledTileGrid().plan(
                        stage=stage, rows=rows, cols=cols, kernel=kernel)
                    if t.col_begin < cols // 2]

    m.register("tile_grid", "gappy_test", Gappy)
    with pytest.raises(ValueError, match="只覆盖到"):
        m.simulate(_scenario(tile_grid="gappy_test"))


def test_empty_tile_range_rejected():
    with pytest.raises(ValueError, match="行列范围必须非空"):
        m.Tile(0, 0, 0, 256)


# ---------------------------------------------------------------------------
# 建图器注册
# ---------------------------------------------------------------------------

def test_orchestration_by_name_and_path():
    from moe_cost_model.builders.mte import MteEventBuilder
    from golden_cases import fingerprint
    base = fingerprint(m.simulate(_scenario()))
    assert fingerprint(m.simulate(_scenario(orchestration="mte"))) == base
    assert fingerprint(m.simulate(
        _scenario(orchestration="moe_cost_model.builders.mte:MteEventBuilder"))) == base
    assert fingerprint(m.simulate(_scenario(orchestration=MteEventBuilder))) == base


def test_orchestration_layered_matches_kernel_switch():
    from golden_cases import fingerprint
    by_kernel = m.simulate(_scenario(kernel=m.KernelConfig(topo_urma=True)))
    by_name = m.simulate(_scenario(kernel=m.KernelConfig(topo_urma=True),
                                   orchestration="layered"))
    assert fingerprint(by_name) == fingerprint(by_kernel)


def test_custom_orchestration_changes_graph():
    from moe_cost_model.builders.mte import MteEventBuilder

    class NoEpilogue(MteEventBuilder):
        def _add_epilogue(self, *args, **kwargs):
            pass

    m.register("orchestration", "no_epilogue_test", NoEpilogue)
    res = m.simulate(_scenario(orchestration="no_epilogue_test"))
    events = res["rank_results"][0]["events"]
    assert not [e for e in events if e.meta.get("stage") == "epilogue"]
    # 执行时间记到最后一个 COMBINE, 不含尾段 → 去掉尾段不改执行时间
    assert res["kernel_total_us"] == m.simulate(_scenario())["kernel_total_us"]


@pytest.mark.parametrize("spec, fragment", [
    ("mtee", "是否想写 'mte'"),
    ("no.such.module:Thing", "无法从"),
    (123, "应为注册名"),
])
def test_bad_orchestration_is_rejected(spec, fragment):
    with pytest.raises(ValueError, match=fragment):
        m.Scenario(workload=m.Workload(tokens=8, world=2, local_experts=2, topk=2),
                   calibration=CAL, orchestration=spec)


def test_strategy_fields_round_trip():
    sc = _scenario(tile_grid={"name": "split_rows", "parts": 2}, orchestration="mte")
    out = sc.to_dict(defaults=False)
    assert out["tile_grid"] == {"name": "split_rows", "parts": 2}
    assert out["orchestration"] == "mte"


# --------------------------------------------------- GMM2 K 维分段就绪 (编排选择)

def _seg_run(segments):
    """固定形状跑一次, 只变 GMM2 的 K 分段数."""
    # 守恒: 每源 2 卡 x 4 专家 x 128 = 1024 行 = 128 x top-8。原先给 topk=6 配 128 token
    # (要 768 行), 不守恒; 本测试讲的是 GMM2 沿 K 的分段, 与 topk 无关, 故取 8。
    return run_api(uniform_routing(2, 4, 128), 128, topk=8, aic_num=28,
                   options=m.ModelOptions(links=links(readiness=segments)))


def test_readiness_default_is_one_segment():
    """缺省不分段: 一个 GMM2 tile 等齐整个 K 的 ACT 再开工 (最少假设)."""
    assert m.ModelOptions().link("activation", "gmm2").readiness.is_whole
    g2 = [e for e in _seg_run("whole")["rank_results"][0]["events"]
          if e.meta.get("stage") == "gmm2"]
    assert {str(e.meta.get("part")) for e in g2} == {"tail"}
    assert not any(e.name.endswith(".h") for e in g2)


@pytest.mark.parametrize("two_segments", ["first_chunk", 2])
def test_two_segments_keep_the_head_tail_names(two_segments):
    """2 段时必须沿用 ".h"/part=head 与不带后缀的 tail —— combine 与
    gmm2_tail_by_group 按这两个名字挂钩, audit_edges 与 test_api_smoke 也认它们.

    两种写法都是 2 段 (首块+其余 / 均分), 名字规则只看段数, 不看是哪一档。"""
    res = _seg_run(two_segments)
    g2 = [e for e in res["rank_results"][0]["events"] if e.meta.get("stage") == "gmm2"]
    parts = {str(e.meta.get("part")) for e in g2}
    assert parts == {"head", "tail"}
    assert any(e.name.endswith(".h") for e in g2)


def test_readiness_bounds_cover_k_without_gap():
    """分段边界必须无缺口无重叠地覆盖 [0, k); 各段时长占比之和 == 1."""
    from moe_cost_model.config.readiness import parse_readiness, segment_spans
    for k, kl1 in ((4608, 256), (4608, 512), (5120, 256), (256, 256), (300, 256)):
        for seg in ("whole", "per_chunk", "first_chunk", 2, 3, 6, 1000):
            b = segment_spans(parse_readiness(seg), k, kl1)
            assert b[0][0] == 0 and b[-1][1] == k, (k, kl1, seg, b)
            for (a_lo, a_hi), (n_lo, _) in zip(b, b[1:]):
                assert a_hi == n_lo, (k, kl1, seg, b)
            assert abs(sum((hi - lo) / k for lo, hi in b) - 1.0) < 1e-12


def test_gmm2_finer_segments_only_wait_their_own_act_columns():
    """逐块模式下, 第 j 段只依赖列范围与它相交的 ACT —— 这就是"对应 tile ACT 完
    就能进 GMM2"的表达。段数越多, 单段等待的 ACT 数越少。"""
    res2 = _seg_run("first_chunk")
    res0 = _seg_run("per_chunk")

    def n_gmm2(result):
        return len([e for e in result["rank_results"][0]["events"]
                    if e.meta.get("stage") == "gmm2"])

    # 逐块的 gmm2 事件数必须远多于两段 (每个 kL1 块一个事件)
    assert n_gmm2(res0) > n_gmm2(res2) * 2
    # 细粒度不该让墙钟变差 (本形状上应改善)
    assert res0["kernel_total_us"] <= res2["kernel_total_us"] * 1.001


def _interleave_build(interleaved, hidden_dim=9216, local=3):
    """建图(不调度): 返回 (gmm1 事件, activation 事件)."""
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    kc = m.KernelConfig(gmm1_interleaved=interleaved)
    mm = m.A8W8WaveCostModel(costs=m.build_analytical_costs(
        h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(), kernel=kc))
    sh = m.MegaMoeShape(expert_tokens=(256,) * local, token_num=tok, h=5120,
                        hidden_dim=hidden_dim, aic_num=28, p1_override=1, p2_override=1,
                        topk=6, kernel=kc,
                        expert_source_tokens=tuple(tuple(rc[0][e]) for e in range(local)))
    evs = mm.build_events(sh)[0]
    return ([e for e in evs if e.meta.get("stage") == "gmm1"],
            [e for e in evs if e.meta.get("stage") == "activation"])


def test_gmm1_interleaved_doubles_tiles_and_halves_b_stream():
    """交织: n-tile 翻倍, 每 tile 只载一个权重块 —— 整层 B 流总量不变."""
    g_off, _ = _interleave_build(False)
    g_on, _ = _interleave_build(True)
    assert len(g_on) == 2 * len(g_off)
    # B 流总量守恒: 18 tile x 2 块 == 36 tile x 1 块
    gm = m.build_analytical_costs(
        h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(),
        kernel=m.KernelConfig(gmm1_interleaved=False)).gmm1_tile.__self__
    gm_i = m.build_analytical_costs(
        h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(),
        kernel=m.KernelConfig(gmm1_interleaved=True)).gmm1_tile.__self__
    assert gm.wb == 2 and gm_i.wb == 1
    b_off = sum(gm.wb * 5120 * e.meta["logical_n"] for e in g_off)
    b_on = sum(gm_i.wb * 5120 * e.meta["logical_n"] for e in g_on)
    assert b_off == b_on


def test_gmm1_interleaved_keeps_activation_output_columns():
    """交织下 epilogueN = tileN/2: ACT 列区间必须仍然无缝覆盖 hidden_dim/2."""
    for interleaved in (False, True):
        _, acts = _interleave_build(interleaved)
        cov = {}
        for e in acts:
            cov.setdefault((e.meta["expert"], e.meta["mgroup"]), []).append(
                (e.meta["col_begin"], e.meta["col_end"]))
        for key, iv in cov.items():
            iv.sort()
            assert iv[0][0] == 0 and iv[-1][1] == 9216 // 2, (interleaved, key, iv)
            for a, b in zip(iv, iv[1:]):
                assert a[1] == b[0], (interleaved, key, iv)   # 无空洞无重叠
