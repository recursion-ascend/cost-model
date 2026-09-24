"""可配置调度策略: ready 集选择 / 核归属 / wave 打包.

三个决策点从硬编码提取为策略接口, 默认值 = 当前行为 (source-faithful):
  1. SchedulingPolicy: DAG 调度器从 ready 集选谁先跑
  2. CoreAssignment:   tile 分给哪个核
  3. WavePacking:      专家怎么组成 wave

KernelConfig 中可替换, simulate 结果随之改变 (what-if 分析).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .dag import Event
from .waves import ExpertSlice, Position, Wave, ceil_div


# ---------------------------------------------------------------------------
# 1. Ready 集选择 (dag.py 调度器)
# ---------------------------------------------------------------------------

class SchedulingPolicy:
    """从 ready 集中选择下一个调度事件的排序键.

    返回 tuple, 越小越优先. 调度器取 min.
    """

    def event_key(self, ev: Event, start: float,
                  tbase: Dict[str, float], end_by_name: Dict[str, float]) -> tuple:
        raise NotImplementedError


class EarliestStart(SchedulingPolicy):
    """当前行为: 最早启动 → 建图序 → 字典序 (source-faithful)."""

    def event_key(self, ev, start, tbase, end_by_name):
        return (start, ev.order, ev.name)


class CriticalPathFirst(SchedulingPolicy):
    """关键路径优先: 下游总时差最小的先跑.

    需要预计算 downstream_slack (从 DAG 反向遍历一次).
    若 meta 无 slack 信息则退化为 EarliestStart.
    """

    def event_key(self, ev, start, tbase, end_by_name):
        slack = ev.meta.get("downstream_slack")
        if slack is None:
            return (start, ev.order, ev.name)
        return (slack, start, ev.order, ev.name)


class PriorityByStage(SchedulingPolicy):
    """按阶段优先级: 指定 stage 排序, 同级按最早启动."""

    def __init__(self, stage_order: Sequence[str] = ("dispatch", "gmm1", "act", "gmm2", "combine")):
        self.prio = {s: i for i, s in enumerate(stage_order)}

    def event_key(self, ev, start, tbase, end_by_name):
        p = self.prio.get(str(ev.meta.get("stage", "")), 99)
        return (p, start, ev.order, ev.name)


# ---------------------------------------------------------------------------
# 2. 核归属 (model.py build_events)
# ---------------------------------------------------------------------------

class CoreAssignment:
    """tile → core 分配策略.

    assign(n_tiles, n_cores, cursor_start, tile_costs) → List[core_id].
    tile_costs: 每 tile 的预估时长 (可选, 供贪心用; None 时退化为均匀).
    """

    def assign(self, n_tiles: int, n_cores: int, cursor_start: int,
               tile_costs: Optional[Sequence[float]] = None) -> List[int]:
        raise NotImplementedError


class StaticRoundRobin(CoreAssignment):
    """当前行为: 静态轮转 (start + i) % cores — kernel 源码移植."""

    def assign(self, n_tiles, n_cores, cursor_start, tile_costs=None):
        return [(cursor_start + i) % n_cores for i in range(n_tiles)]


class GreedyLeastBusy(CoreAssignment):
    """贪心最闲核: 按 tile 预估时长逐个分配给当前负载最轻的核."""

    def assign(self, n_tiles, n_cores, cursor_start, tile_costs=None):
        loads = [0.0] * n_cores
        assign = []
        for i in range(n_tiles):
            c = min(range(n_cores), key=lambda k: loads[k])
            assign.append(c)
            loads[c] += (tile_costs[i] if tile_costs and i < len(tile_costs) else 1.0)
        return assign


class ContiguousBlock(CoreAssignment):
    """连续块: 每个 core 领连续一段 tile (减少跨核同步)."""

    def assign(self, n_tiles, n_cores, cursor_start, tile_costs=None):
        per = ceil_div(n_tiles, n_cores)
        assign = []
        for i in range(n_tiles):
            assign.append(min(i // per, n_cores - 1))
        return assign


# ---------------------------------------------------------------------------
# 3. Wave 打包 (waves.py plan_waves)
# ---------------------------------------------------------------------------

class WavePacking:
    """专家怎么组成 wave.

    plan(expert_tokens, m_groups_per_wave, tile_m) → List[Wave].
    """

    def plan(self, expert_tokens: Sequence[int], m_groups_per_wave: int,
             tile_m: int) -> List[Wave]:
        raise NotImplementedError


class SequentialGreedy(WavePacking):
    """当前行为: 按专家顺序贪心填满 mGroupsPerWave (kernel 源码移植)."""

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
        waves: List[Wave] = []
        global_row = 0
        used_groups = 0
        slices: List[ExpertSlice] = []
        wave_begin = Position(0, 0, 0)
        begin_row_in_expert = 0

        for e in order:
            if counts[e] == 0:
                continue
            row = 0
            while row < counts[e]:
                if used_groups >= mgw and slices:
                    waves.append(Wave(len(waves), wave_begin, wave_begin, tuple(slices)))
                    slices = []
                    used_groups = 0
                    wave_begin = Position(e, row, global_row)
                take = min(counts[e] - row, (mgw - used_groups) * tile_m)
                mg = ceil_div(take, tile_m)
                slices.append(ExpertSlice(e, row, row + take, global_row,
                                          global_row + take, m_groups=mg))
                global_row += take
                used_groups += mg
                row += take
        if slices:
            waves.append(Wave(len(waves), wave_begin, wave_begin, tuple(slices)))
        return waves


class BalancedWaves(WavePacking):
    """均衡 wave: 每波总行数尽量均匀 (最小化最慢波)."""

    def plan(self, expert_tokens, m_groups_per_wave, tile_m):
        counts = list(expert_tokens)
        nonzero = [(e, c) for e, c in enumerate(counts) if c > 0]
        if not nonzero:
            return []
        total_rows = sum(c for _, c in nonzero)
        total_mg = sum(ceil_div(c, tile_m) for _, c in nonzero)
        n_waves = ceil_div(total_mg, m_groups_per_wave)
        target = ceil_div(total_rows, n_waves)

        # 贪心: 每波填到 target 行为止
        waves: List[Wave] = []
        slices: List[ExpertSlice] = []
        used_groups = 0
        used_rows = 0
        global_row = 0
        wave_begin = Position(0, 0, 0)

        for e, c in nonzero:
            row = 0
            while row < c:
                remaining_wave = target - used_rows
                if remaining_wave <= 0 and slices:
                    waves.append(Wave(len(waves), wave_begin, wave_begin, tuple(slices)))
                    slices = []
                    used_groups = 0
                    used_rows = 0
                    wave_begin = Position(e, row, global_row)
                    remaining_wave = target
                take = min(c - row, remaining_wave, (m_groups_per_wave - used_groups) * tile_m)
                if take <= 0:
                    break
                mg = ceil_div(take, tile_m)
                slices.append(ExpertSlice(e, row, row + take, global_row,
                                          global_row + take, m_groups=mg))
                global_row += take
                used_groups += mg
                used_rows += take
                row += take
        if slices:
            waves.append(Wave(len(waves), wave_begin, wave_begin, tuple(slices)))
        return waves
"""策略层: 图结构随资源竞争的运行时重构策略 (钩子函数工厂).

每个工厂返回 hook(RestructureContext) -> RestructureAction, 供
MultiResourceScheduler.schedule(restructure=...) 使用. 全部确定性.
契约: cancel 的事件必须同名 reinject (否则消费者死图报错).
"""
from collections import defaultdict

from .dag import Event, RestructureAction


def idle_core_stealing(stage: str = "gmm1", resource_prefix: str = "AIC:",
                       min_pending: int = 2, queue_token: str = "Q:aic"):
    """空核偷活 (反事实策略): 当本实例为静态 cursor 分配时, 量化
    "若改成动态负载均衡" 的收益. 触发: 某核该 stage 的未发射 tile 数
    ≥ min_pending 且存在空闲核 → 取消其尾部一个 tile 事件, 同名注入到空闲核.

    同名注入保证消费者 (如 ACT 的 dep) 语义不变; 资源与队列令牌换到新核.
    """
    def hook(ctx) -> RestructureAction:
        act = RestructureAction()
        # 核集合 = 已见资源 ∪ 未发射事件资源 (空闲核没有 pending 事件, 必须补齐)
        cores = {r for r in ctx.resource_free if r.startswith(resource_prefix)}
        for e in ctx.pending.values():
            for r in e.resources:
                if r.startswith(resource_prefix):
                    cores.add(r)
        if len(cores) < 2:
            return act
        # 从未提交过的核 free 记 0 (自始空闲); min 取最空闲核
        free = {r: ctx.resource_free.get(r, 0.0) for r in cores}
        by_res = defaultdict(list)
        for n, e in ctx.pending.items():
            if str(e.meta.get("stage")) == stage and e.resources:
                by_res[e.resources[0]].append(e)
        if not by_res:
            return act  # 该 stage 无未发射 tile
        idlest = min(free, key=lambda r: (free[r], r))
        pend = by_res.get(idlest, [])
        if len(pend) >= min_pending:
            return act  # 空闲核自己还有活, 不偷
        busiest = max(by_res, key=lambda r: (len(by_res[r]), r))
        if busiest == idlest or len(by_res[busiest]) < min_pending:
            return act
        victim = by_res[busiest][-1]
        new_core = idlest
        old_cid = victim.resources[0].split(":", 1)[1]   # 核标识 (资源名冒号后缀)
        new_cid = new_core.split(":", 1)[1]

        def remap_token(tok: str) -> str:
            # 队列令牌尾缀换核: "Q:aic:c0"/"Q:aic:0" → 对应新核后缀
            return tok[:-len(old_cid)] + new_cid if tok.endswith(old_cid) else tok

        act.cancel.append(victim.name)
        act.inject.append(Event(
            name=victim.name,
            resources=(new_core,),
            duration_us=victim.duration_us,
            deps=victim.deps,
            order=victim.order,
            meta=dict(victim.meta, stolen_from=busiest),
            dep_latency_us=victim.dep_latency_us,
            dep_latency_overrides=victim.dep_latency_overrides,
            acquires=tuple((remap_token(q), k) for q, k in victim.acquires),
            releases=tuple((remap_token(q), k) for q, k in victim.releases),
            channel_bytes=victim.channel_bytes,
        ))
        return act
    return hook

