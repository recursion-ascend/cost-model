"""第 3 层: tile 网格策略 — 一个专家切片的输出怎么切成 tile.

GMM1 与 GMM2 的 tile 网格都由这里给出。默认 SwizzledTileGrid 复现 kernel 现行为
(行按 tile_m 分组、列按 tile_n 切、Blaze swizzle 序)；换一个实现就能改切分方式,
事件图随之重建, 不用动建图源码。

坐标都是切片内的相对值, 左闭右开:
  行 row_begin..row_end   该专家切片的第几行 (切片起点为 0)
  列 col_begin..col_end   输出的第几列
     GMM1 的列 = ceil(intermediate / activation_n_half) 个逻辑列
     GMM2 的列 = h

约束 (validate_tiles 强制):
  * tile 必须无重叠地铺满 rows × cols;
  * 行范围不得跨越 m-group 边界 (m-group = tile_m 行一组) —— dispatch 就绪、
    ACT 产出与 GMM2 的依赖都以 m-group 为单位, 跨界会让依赖失去意义。
  组内可以再切: 把一个 m-group 的行或列拆成多个 tile 是允许的, 这正是让更多核
  参与计算的办法。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..config.hardware import ceil_div
from .waves import swizzle_coord

STAGE_GMM1 = "gmm1"
STAGE_GMM2 = "gmm2"


@dataclass(frozen=True)
class Tile:
    """一个输出 tile 的行列范围 (切片内相对坐标, 左闭右开)."""

    row_begin: int
    row_end: int
    col_begin: int
    col_end: int

    def __post_init__(self) -> None:
        if self.row_end <= self.row_begin or self.col_end <= self.col_begin:
            raise ValueError(f"tile 的行列范围必须非空: {self}")

    @property
    def rows(self) -> int:
        return self.row_end - self.row_begin

    @property
    def cols(self) -> int:
        return self.col_end - self.col_begin


class TileGrid:
    """把一个专家切片的 rows × cols 输出切成 tile.

    plan() 返回的列表顺序即建图顺序 (也就是分核策略看到的顺序)。
    """

    def plan(self, *, stage: str, rows: int, cols: int, kernel) -> List[Tile]:
        raise NotImplementedError


class SwizzledTileGrid(TileGrid):
    """默认: 行按 tile_m 分组, 列按 tile_n 切, 按 Blaze swizzle 序遍历.

    与 kernel 现行为逐 tile 一致。
    """

    def plan(self, *, stage, rows, cols, kernel):
        tile_m, tile_n = kernel.tile_m, kernel.tile_n
        m_groups, n_tiles = ceil_div(rows, tile_m), ceil_div(cols, tile_n)
        out: List[Tile] = []
        for idx in range(m_groups * n_tiles):
            mg, nt = swizzle_coord(idx, m_groups, n_tiles,
                                   kernel.swizzle_offset, kernel.swizzle_direction)
            rb, cb = mg * tile_m, nt * tile_n
            out.append(Tile(rb, min(rb + tile_m, rows), cb, min(cb + tile_n, cols)))
        return out


class SplitRowsTileGrid(TileGrid):
    """把每个 m-group 的行再切成 parts 份, 列的切法不变.

    m-group 行数不足 tile_m 时 (小 batch) GMM1 的 tile 数少于核数, 用它可以把
    同一组的行摊到更多核上。列仍按 tile_n 切; tile 仍在 m-group 内, 不跨界。
    """

    def __init__(self, parts: int = 2, stages: Sequence[str] = (STAGE_GMM1, STAGE_GMM2)):
        if parts < 1:
            raise ValueError("parts 必须 >= 1")
        self.parts = int(parts)
        self.stages = tuple(stages)

    def plan(self, *, stage, rows, cols, kernel):
        base = SwizzledTileGrid().plan(stage=stage, rows=rows, cols=cols, kernel=kernel)
        if stage not in self.stages or self.parts == 1:
            return base
        out: List[Tile] = []
        for t in base:
            n = min(self.parts, t.rows)
            step = ceil_div(t.rows, n)
            rb = t.row_begin
            while rb < t.row_end:
                re = min(rb + step, t.row_end)
                out.append(Tile(rb, re, t.col_begin, t.col_end))
                rb = re
        return out


def validate_tiles(tiles: Sequence[Tile], *, rows: int, cols: int, tile_m: int,
                   where: str = "") -> None:
    """铺满校验 + m-group 不跨界校验; 不合法时报出具体缺口."""
    at = f"{where}: " if where else ""
    if not tiles:
        raise ValueError(f"{at}tile 网格为空, 但切片有 {rows} 行 × {cols} 列")
    for t in tiles:
        if t.row_begin < 0 or t.row_end > rows or t.col_end > cols:
            raise ValueError(f"{at}tile {t} 超出切片范围 {rows}×{cols}")
        if t.row_begin // tile_m != (t.row_end - 1) // tile_m:
            raise ValueError(
                f"{at}tile {t} 跨越了 m-group 边界 (每 {tile_m} 行一组); "
                "dispatch 就绪与 ACT 依赖以 m-group 为单位, tile 必须落在组内")
    bands = {}
    for t in tiles:
        bands.setdefault((t.row_begin, t.row_end), []).append((t.col_begin, t.col_end))
    _partition(sorted(bands), rows, at, "行")
    for band, spans in bands.items():
        _partition(sorted(spans), cols, at, f"行 {band[0]}–{band[1]} 的列")


def _partition(spans: Sequence[tuple], total: int, at: str, what: str) -> None:
    """spans 必须无重叠地铺满 [0, total)."""
    cursor = 0
    for begin, end in spans:
        if begin != cursor:
            gap = "重叠" if begin < cursor else "空缺"
            raise ValueError(f"{at}{what}在 {min(begin, cursor)} 处{gap}: "
                             f"{cursor} 之后接的是 {begin}")
        cursor = end
    if cursor != total:
        raise ValueError(f"{at}{what}只覆盖到 {cursor}, 应铺满 {total}")
