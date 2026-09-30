"""自定义切分: tile 网格策略 + 建图器注册.

覆盖 "工程师自己定义 megamoe 切分方式, 事件 DAG 自动重建" 这条路径。
"""
import collections
import dataclasses

import pytest

import moe_cost_model as m
from golden_cases import CUBE_RATE, run_api, skewed_routing, uniform_routing

CAL = m.Calibration(cube_mac_per_us=CUBE_RATE)


def _scenario(**kw):
    return m.Scenario(
        workload=m.Workload(tokens=64, routing="explicit", counts=skewed_routing()),
        p1_override=2, p2_override=1, calibration=CAL, **kw)


def _stage_cores(res, stage, rank=0):
    return collections.Counter(
        int(e.resources[0].split(":")[1]) for e in res["rank_results"][rank]["events"]
        if e.meta.get("stage") == stage and e.resources)


# ---------------------------------------------------------------------------
# 默认网格 = kernel 现行为
# ---------------------------------------------------------------------------

def test_swizzled_grid_matches_kernel_geometry():
    """默认网格: 行按 tile_m 分组、列按 tile_n 切, 顺序与 swizzle_coord 一致."""
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


def test_default_grid_is_the_no_op():
    """显式给默认网格与不给完全一致 (事件级)."""
    base = _scenario()
    explicit = _scenario(tile_grid="swizzled")
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
    events, _ = A8W8WaveCostModel(manual_costs(), m.ModelOptions()).build_events(shape)
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
