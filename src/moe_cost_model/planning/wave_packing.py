"""第 3 层: wave 打包策略 — SequentialGreedy / LongestExpertFirst / BalancedWaves."""
from __future__ import annotations

from typing import List

from .waves import ExpertSlice, Position, Wave, ceil_div


class WavePacking:
    """专家怎么组成 wave.

    plan(expert_tokens, m_groups_per_wave, tile_m) → List[Wave].
    """

    def plan(self, expert_tokens: Sequence[int], m_groups_per_wave: int,
             tile_m: int) -> List[Wave]:
        raise NotImplementedError


class SequentialGreedy(WavePacking):
    """当前行为: 按专家顺序贪心填满 mGroupsPerWave."""

    def plan(self, expert_tokens, m_groups_per_wave, tile_m):
        # 复用现有 plan_waves 逻辑
        from .waves import plan_waves
        return plan_waves(expert_tokens, m_groups_per_wave, tile_m)


class LongestExpertFirst(WavePacking):
    """大专家优先: 行数最多的专家先打包 (减少尾波浪费)."""

    def plan(self, expert_tokens, m_groups_per_wave, tile_m):
        counts = list(expert_tokens)
        # 按行数降序的专家索引
        sorted_idx = sorted(range(len(counts)), key=lambda e: -counts[e])
        # 用排序后的顺序打包, 但 Wave.slices 保留原专家号
        return self._pack_sorted(counts, sorted_idx, m_groups_per_wave, tile_m)

    def _pack_sorted(self, counts, order, mgw, tile_m):
        """按给定专家序贪心打包; slice 只在组边界切分 (row_begin 对齐 tile_m),
        Wave.end 记录波末 frontier (rows 语义与 plan_waves 一致)."""
        waves: List[Wave] = []
        slices: List[ExpertSlice] = []
        used_groups = 0
        global_row = 0
        wave_begin = Position(0, 0, 0)
        frontier = Position(0, 0, 0)

        for e in order:
            if counts[e] == 0:
                continue
            row = 0
            while row < counts[e]:
                if used_groups >= mgw and slices:
                    waves.append(Wave(len(waves), wave_begin, frontier, tuple(slices)))
                    slices = []
                    used_groups = 0
                    wave_begin = frontier
                take = min(counts[e] - row, (mgw - used_groups) * tile_m)
                mg = ceil_div(take, tile_m)
                slices.append(ExpertSlice(e, row, row + take, global_row,
                                          global_row + take, m_groups=mg))
                global_row += take
                used_groups += mg
                row += take
                frontier = Position(e, row, global_row)
        if slices:
            waves.append(Wave(len(waves), wave_begin, frontier, tuple(slices)))
        return waves


class BalancedWaves(WavePacking):
    """均衡 wave: 每波总行数尽量均匀 (最小化最慢波).

    按 256 行组展开后均匀分配: 组边界切分保证 row_begin 对齐 tile_m,
    全部行都被覆盖 (无丢弃), 每波组数 ≤ mgw, 同专家连续组合并为 slice.
    """

    def plan(self, expert_tokens, m_groups_per_wave, tile_m):
        counts = list(expert_tokens)
        groups: List[Tuple[int, int, int]] = []   # (expert, row_begin, row_end)
        for e, c in enumerate(counts):
            row = 0
            while row < c:
                nxt = min(row + tile_m, c)
                groups.append((e, row, nxt))
                row = nxt
        if not groups:
            return []

        total_rows = sum(g[2] - g[1] for g in groups)
        n_waves = ceil_div(len(groups), m_groups_per_wave)
        target = ceil_div(total_rows, n_waves)

        waves: List[Wave] = []
        slices: List[ExpertSlice] = []
        used_rows = 0
        groups_in_wave = 0
        global_row = 0
        wave_begin = Position(0, 0, 0)
        frontier = Position(0, 0, 0)

        for e, rb, re_ in groups:
            take = re_ - rb
            if slices and (used_rows >= target
                           or groups_in_wave >= m_groups_per_wave):
                waves.append(Wave(len(waves), wave_begin, frontier, tuple(slices)))
                slices = []
                used_rows = 0
                groups_in_wave = 0
                wave_begin = frontier
            if slices and slices[-1].expert == e and slices[-1].row_end == rb:
                last = slices[-1]
                slices[-1] = ExpertSlice(
                    e, last.row_begin, re_, last.global_row_begin,
                    global_row + take, m_groups=last.m_groups + 1)
            else:
                slices.append(ExpertSlice(
                    e, rb, re_, global_row, global_row + take, m_groups=1))
            global_row += take
            used_rows += take
            groups_in_wave += 1
            frontier = Position(e, re_, global_row)
        if slices:
            waves.append(Wave(len(waves), wave_begin, frontier, tuple(slices)))
        return waves
"""策略层: 图结构随资源竞争的运行时重构策略.

每个工厂返回 hook(RestructureContext) -> RestructureAction, 供
MultiResourceScheduler.schedule(restructure=...) 使用. 全部确定性.
契约: cancel 的事件必须同名 reinject (否则消费者死图报错).
"""