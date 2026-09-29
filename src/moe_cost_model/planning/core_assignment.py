""" tile → 核归属策略 — StaticRoundRobin / GreedyLeastBusy / ContiguousBlock."""
from __future__ import annotations

from typing import List, Optional, Sequence

from .waves import ceil_div



class CoreAssignment:
    """tile → core 分配策略.

    assign(n_tiles, n_cores, cursor_start, tile_costs) → List[core_id].
    tile_costs: 每 tile 的预估时长 (可选, 供贪心用; None 时退化为均匀).
    """

    def assign(self, n_tiles: int, n_cores: int, cursor_start: int,
               tile_costs: Optional[Sequence[float]] = None) -> List[int]:
        raise NotImplementedError


class StaticRoundRobin(CoreAssignment):
    """默认: 静态轮转 (start + i) % cores."""

    def assign(self, n_tiles, n_cores, cursor_start, tile_costs=None):
        return [(cursor_start + i) % n_cores for i in range(n_tiles)]


class GreedyLeastBusy(CoreAssignment):
    """贪心最闲核: 按 tile 预估时长逐个分配给当前负载最轻的核.

    逻辑核序从 cursor_start 起轮转, 物理 core =
    (cursor_start + 逻辑位) mod 核数 — 与 StaticRoundRobin 同构,
    逐切片的起始核随游标推进, 不总是核 0.
    """

    def assign(self, n_tiles, n_cores, cursor_start, tile_costs=None):
        loads = [0.0] * n_cores
        assign = []
        for i in range(n_tiles):
            j = min(range(n_cores), key=lambda k: (loads[k], k))
            c = (cursor_start + j) % n_cores
            assign.append(c)
            loads[j] += (tile_costs[i] if tile_costs and i < len(tile_costs) else 1.0)
        return assign


class ContiguousBlock(CoreAssignment):
    """连续块: 每个核领连续一段 tile (减少跨核同步).
       块边界从 cursor_start 起轮转.
    """

    def assign(self, n_tiles, n_cores, cursor_start, tile_costs=None):
        per = ceil_div(n_tiles, n_cores)
        assign = []
        for i in range(n_tiles):
            assign.append((cursor_start + min(i // per, n_cores - 1)) % n_cores)
        return assign