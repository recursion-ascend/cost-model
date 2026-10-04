"""第 4 层: tile 网格接线 — 取策略、给 tile 起名.

网格策略本身在 planning/tile_grid.py; 这里只负责建图侧的两件小事。
"""
from __future__ import annotations

from ..planning.tile_grid import RowMajorTileGrid, Tile, TileGrid

_DEFAULT = RowMajorTileGrid()


def resolve_grid(shape) -> TileGrid:
    """shape.tile_grid; 未给时用缺省网格 (行主序, 最少假设)."""
    grid = getattr(shape, "tile_grid", None)
    if grid is None:
        return _DEFAULT
    if not isinstance(grid, TileGrid):
        raise ValueError(f"tile_grid 必须是 TileGrid 子类的实例, 得到 {grid!r}")
    return grid


def tile_label(t: Tile, rows: int, cols: int, tile_m: int, tile_n: int) -> str:
    """tile 的名字片段.

    默认网格下就是 m{组号}.n{列块号} —— 与旧事件名一致。自定义网格在组内再切
    行或列时, 追加实际范围, 保证同一组里的名字不重名。
    """
    mg, nt = t.row_begin // tile_m, t.col_begin // tile_n
    label = f"m{mg}.n{nt}"
    full_row = (mg * tile_m, min(mg * tile_m + tile_m, rows))
    full_col = (nt * tile_n, min(nt * tile_n + tile_n, cols))
    if (t.row_begin, t.row_end) != full_row:
        label += f".r{t.row_begin}_{t.row_end}"
    if (t.col_begin, t.col_end) != full_col:
        label += f".k{t.col_begin}_{t.col_end}"
    return label


def coalesce_tiles(tiles, items_per_event: int, *, where: str = ""):
    """把 tile 列表按事件粒度合并成"工作项", 返回 [(合并后的 Tile, 成员 tile 列表)].

    合并规则 (物理决定的, 不是口味): 只合并**行范围相同、列范围相邻**的连续 tile。

      * 行范围必须相同 —— 行范围不同就不是一个 matmul 的输出块, 合并后的
        (row,col) 矩形会盖住没算的格子, 下游按行列范围挑依赖就会错;
      * 列必须相邻 —— ``ctx.activation_ready`` 与 GMM2 的 K 段都按 **连续区间**
        挑依赖 (``col_begin < k_hi and col_end > k_lo``), 不连续的并集表达不出来。

    遇到合不拢的边界 (换行、列不连续) 就**截断**当前项, 所以实际项内个数可能少于
    items_per_event。这是对的: 粒度是上界, 不是强行凑数的配额。

    items_per_event == 1 时原样返回每个 tile 单独一项 —— 与合并前逐字节等价。
    items_per_event == 0 表示"整个切片一个事件", 在这里等价于不设上限 (仍受上面
    两条物理规则截断)。
    """
    if items_per_event < 0:
        raise ValueError(f"{where} 的事件粒度不能为负")
    if items_per_event == 1:
        return [(t, [t]) for t in tiles]
    cap = items_per_event if items_per_event > 0 else len(tiles)
    out = []
    run = []
    for t in tiles:
        if run:
            prev = run[-1]
            mergeable = (t.row_begin == prev.row_begin
                         and t.row_end == prev.row_end
                         and t.col_begin == prev.col_end)
            if not mergeable or len(run) >= cap:
                out.append(_merge_run(run))
                run = []
        run.append(t)
    if run:
        out.append(_merge_run(run))
    return out


def _merge_run(run):
    first, last = run[0], run[-1]
    if len(run) == 1:
        return (first, list(run))
    return (Tile(first.row_begin, first.row_end, first.col_begin, last.col_end),
            list(run))
