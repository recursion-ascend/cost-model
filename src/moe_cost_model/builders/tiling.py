"""第 4 层: tile 网格接线 — 取策略、给 tile 起名.

网格策略本身在 planning/tile_grid.py; 这里只负责建图侧的两件小事。
"""
from __future__ import annotations

from ..planning.tile_grid import SwizzledTileGrid, Tile, TileGrid

_DEFAULT = SwizzledTileGrid()


def resolve_grid(shape) -> TileGrid:
    """shape.tile_grid; 未给时用默认网格 (kernel 现行为)."""
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
