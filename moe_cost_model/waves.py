"""Source-faithful host / wave planning.

PlanNextExpertTokenRangeInWave / AdvanceExpertTokenPositionInWave 的
Python 移植, 与 kernel host tiling 逻辑逐行对应.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .constants import (
    GMM1_MIN_LOGICAL_TILES_PER_CORE,
    GMM2_MIN_LOGICAL_TILES_PER_CORE,
    GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL,
    GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE,
    GMM1_SMALL_BATCH_TOKEN_THRESHOLD,
    GMM1_LARGE_BATCH_TOKEN_THRESHOLD,
    TILE_M, TILE_N, ceil_div,
)


def resolve_gmm1_min_logical_tiles_per_core(token_num: int) -> int:
    """kernel 默认波策略参考 (mega_moe_constants.h:99-103 分层), 仅供复现
    kernel 现行为的工具使用; cost model 不再使用 —— p1/p2 是场景超参,
    由调用方或 tiling 真值给出, 未给时模型取理论下限 (1, 1)."""
    if token_num < GMM1_SMALL_BATCH_TOKEN_THRESHOLD:
        return GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL
    if token_num >= GMM1_LARGE_BATCH_TOKEN_THRESHOLD:
        return GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE
    return GMM1_MIN_LOGICAL_TILES_PER_CORE


def calc_m_groups_per_wave(
    *,
    hidden_dim: int,
    h: int,
    aic_num: int,
    p1: int = GMM1_MIN_LOGICAL_TILES_PER_CORE,
    p2: int = GMM2_MIN_LOGICAL_TILES_PER_CORE,
    tile_n: int = TILE_N,
) -> int:
    """Parameterized form of CalcMGroupsPerWave() used by the validation patch."""
    if hidden_dim <= 0 or h <= 0 or aic_num <= 0:
        return 1
    if p1 <= 0 or p2 <= 0:
        raise ValueError("p1/p2 must be positive effective policy values")
    gmm1_logical_tiles_per_mgroup = ceil_div(hidden_dim, tile_n)
    gmm2_tiles_per_mgroup = ceil_div(h, tile_n)
    gmm1_required = ceil_div(aic_num * p1, gmm1_logical_tiles_per_mgroup)
    gmm2_required = ceil_div(aic_num * p2, gmm2_tiles_per_mgroup)
    return max(gmm1_required, gmm2_required)


@dataclass(frozen=True)
class Position:
    expert: int = 0
    row: int = 0
    global_row: int = 0


@dataclass(frozen=True)
class ExpertSlice:
    expert: int
    row_begin: int
    row_end: int
    global_row_begin: int
    global_row_end: int
    m_groups: int = 0   # 构造期由 plan_waves 按 tile_m 计算

    @property
    def rows(self) -> int:
        return self.row_end - self.row_begin


@dataclass(frozen=True)
class Wave:
    index: int
    begin: Position
    end: Position
    slices: Tuple[ExpertSlice, ...]

    @property
    def rows(self) -> int:
        return self.end.global_row - self.begin.global_row

    @property
    def m_groups(self) -> int:
        return sum(s.m_groups for s in self.slices)


def plan_waves(expert_tokens: Sequence[int], m_groups_per_wave: int,
              tile_m: int = TILE_M) -> List[Wave]:
    """Port PlanNextExpertTokenRangeInWave / AdvanceExpertTokenPositionInWave."""
    if m_groups_per_wave <= 0:
        raise ValueError("m_groups_per_wave must be positive")
    counts = [max(0, int(x)) for x in expert_tokens]
    e = 0
    row = 0
    global_row = 0
    waves: List[Wave] = []

    while e < len(counts):
        while e < len(counts) and (counts[e] == 0 or row >= counts[e]):
            e += 1
            row = 0
        if e >= len(counts):
            break

        wave_begin = Position(e, row, global_row)
        used_groups = 0
        slices: List[ExpertSlice] = []

        while e < len(counts) and used_groups < m_groups_per_wave:
            if counts[e] == 0 or row >= counts[e]:
                e += 1
                row = 0
                continue

            remaining = counts[e] - row
            capacity_rows = (m_groups_per_wave - used_groups) * tile_m
            take = min(remaining, capacity_rows)
            if take <= 0:
                break

            row_begin = row
            global_begin = global_row
            row += take
            global_row += take
            used_groups += ceil_div(take, tile_m)
            slices.append(ExpertSlice(e, row_begin, row, global_begin, global_row,
                                     m_groups=ceil_div(take, tile_m) if take else 0))

            if row >= counts[e]:
                e += 1
                row = 0

        waves.append(
            Wave(
                index=len(waves),
                begin=wave_begin,
                end=Position(e, row, global_row),
                slices=tuple(slices),
            )
        )

    return waves



def swizzle_coord(tile_idx: int, m_groups: int, n_tiles: int, swizzle_offset: int = 3,
                  swizzle_direction: int = 0):
    """Blaze BlockSchedulerSwizzle<3,0> 移植 (block_scheduler_swizzle.h:26-97).

    Direction=0 (kernel 实际使用): loopFirst=M 组, loopSecond=N 列;
    3 个 first 维为块, 块内 first 变化最快 (连续 tile 共享同一 B 列块, L2 友好),
    奇数块 second 方向蛇形反转. Direction=1: N/M 角色互换 (构造函数镜像).
    m_groups==1 且 direction=0 时退化为 (0, tile_idx).
    """
    if swizzle_direction == 0:
        loop_first, loop_second = m_groups, n_tiles
    else:
        loop_first, loop_second = n_tiles, m_groups
    block_span = swizzle_offset * loop_second
    block_idx = tile_idx // block_span
    in_block = tile_idx % block_span
    first_valid = loop_first - block_idx * swizzle_offset
    if first_valid > swizzle_offset:
        first_valid = swizzle_offset
    if first_valid <= 0:
        first_valid = 1
    first_idx = block_idx * swizzle_offset + in_block % first_valid
    second_idx = in_block // first_valid
    if block_idx & 1:
        second_idx = loop_second - second_idx - 1
    if swizzle_direction == 0:
        return first_idx, second_idx      # (mg, nt)
    return second_idx, first_idx          # (mg, nt) = (second, first)
