"""wave planning.

PlanNextExpertTokenRangeInWave / AdvanceExpertTokenPositionInWave 的
Python 移植, 与 kernel host tiling 逻辑逐行对应.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..config.hardware import (
    GMM1_MIN_LOGICAL_TILES_PER_CORE,
    GMM2_MIN_LOGICAL_TILES_PER_CORE,
    GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL,
    GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE,
    GMM1_SMALL_BATCH_TOKEN_THRESHOLD,
    GMM1_LARGE_BATCH_TOKEN_THRESHOLD,
    LAYERED_BALANCED_WAVE_COUNT, LAYERED_FEW_EXPERT_THRESHOLD,
    LAYERED_FIRST_WAVE_ROWS, LAYERED_LATENCY_ROWS_PER_EXPERT,
    LAYERED_LATENCY_WAVE_COUNT, LAYERED_THROUGHPUT_ROWS_PER_EXPERT,
    LAYERED_THROUGHPUT_WAVE_COUNT,
    TILE_M, TILE_N, ceil_div,
)


def resolve_gmm1_min_logical_tiles_per_core(token_num: int) -> int:
    """kernel 默认波策略参考, 仅供复现
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
    """Blaze BlockSchedulerSwizzle 移植 (block_scheduler_swizzle.h:26-97).

    Direction=0 (仓内 kernel 实例化 <3, 0>, 见 KernelConfig.swizzle_direction):
    loopFirst=M 组, loopSecond=N 列; 3 个 first 维为块, 块内 first 变化最快,
    奇数块 second 方向蛇形反转。Direction=1: M/N 角色互换 (构造函数镜像), 此时
    连续 tile 共享同一 A 行块。
    m_groups==1 时两个方向同为 (0, tile_idx)。
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


# =====================================================================
# URMA Layered 宏 Wave 规划
# =====================================================================

def calc_layered_target_wave_count(expert_count: int, token_num: int, topk: int) -> int:
    """Port CalcTargetWaveCount: 计算/macrob Wave 只组织 GMM 阶段的专家范围.

    小负载 1~2 个有效 Wave; 中等负载 ~6; 单专家很大的吞吐场景 ~4.
    """
    if expert_count <= 1:
        return expert_count
    total_rows = token_num * topk
    if total_rows <= int(LAYERED_FIRST_WAVE_ROWS):
        return 1
    est = ceil_div(total_rows, expert_count)
    target = int(LAYERED_BALANCED_WAVE_COUNT)
    if est <= int(LAYERED_LATENCY_ROWS_PER_EXPERT):
        target = int(LAYERED_LATENCY_WAVE_COUNT)
    elif est >= int(LAYERED_THROUGHPUT_ROWS_PER_EXPERT):
        target = int(LAYERED_THROUGHPUT_WAVE_COUNT)
    elif expert_count <= int(LAYERED_FEW_EXPERT_THRESHOLD):
        target = int(LAYERED_LATENCY_WAVE_COUNT)
    return min(target, expert_count)


def calc_layered_first_wave_expert_count(expert_count: int, token_num: int, topk: int,
                                         target_wave_count: int) -> int:
    """Port CalcFirstWaveExpertCount: 首波暖实行预算与 Wave 数预算取小."""
    if expert_count == 0 or target_wave_count == 0:
        return 0
    if target_wave_count == 1:
        return expert_count
    total_rows = token_num * topk
    est = ceil_div(total_rows, expert_count) if total_rows > 0 else 1
    if est == 0:
        est = 1
    by_budget = ceil_div(expert_count, target_wave_count)
    by_warmup = ceil_div(int(LAYERED_FIRST_WAVE_ROWS), est)
    first = min(by_warmup, by_budget)
    if first == 0:
        first = 1
    return min(first, expert_count)


def calc_layered_steady_wave_expert_count(first_wave_experts: int, target_wave_count: int,
                                          expert_count: int) -> int:
    """Port CalcSteadyWaveExpertCount: 剩余专家均摊到剩余 Wave."""
    if first_wave_experts >= expert_count or target_wave_count <= 1:
        return expert_count
    remaining_experts = expert_count - first_wave_experts
    remaining_waves = target_wave_count - 1
    steady = ceil_div(remaining_experts, remaining_waves)
    return steady if steady > 0 else 1


def plan_layered_waves(expert_tokens: Sequence[int], token_num: int, topk: int,
                       tile_m: int = TILE_M) -> List[Wave]:
    """URMA Layered 宏 Wave 规划: Wave = 连续专家范围 (非 256 行 m-group).

    与 MTE Wave (plan_waves) 的区别: 边界只落在专家边界上, 专家不在 Wave 内
    切分; 空专家保留在范围内 (kernel UpdateGroupParams m=0 跳过) 但不产生
    slice/tile. 返回的 Wave 对象直接复用 GMM1/ACT/GMM2 事件构建 (slice =
    完整专家行区间).
    """
    counts = [max(0, int(x)) for x in expert_tokens]
    experts = len(counts)
    target = calc_layered_target_wave_count(experts, token_num, topk)
    first = calc_layered_first_wave_expert_count(experts, token_num, topk, target)
    steady = calc_layered_steady_wave_expert_count(first, target, experts)

    ranges: List[Tuple[int, int]] = []
    begin, end = 0, first
    while begin < experts:
        ranges.append((begin, min(end, experts)))
        begin, end = end, end + steady

    waves: List[Wave] = []
    global_row = 0
    for idx, (b, e) in enumerate(ranges):
        slices: List[ExpertSlice] = []
        wave_global_begin = global_row
        for exp in range(b, e):
            rows = counts[exp]
            if rows <= 0:
                continue
            slices.append(ExpertSlice(exp, 0, rows, global_row, global_row + rows,
                                      m_groups=ceil_div(rows, tile_m)))
            global_row += rows
        waves.append(Wave(
            index=idx,
            begin=Position(b, 0, wave_global_begin),
            end=Position(e, 0, global_row),
            slices=tuple(slices),
        ))
    return waves
