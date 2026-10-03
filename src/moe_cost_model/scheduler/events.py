"""第 2 层: 调度引擎的数据结构 — 事件 / 信道 / 重构动作 / 计划事件.

三层扩展语义:
  L0 逐边依赖延迟: Event.dep_latency_overrides [(dep_name, us)]
  L1 计数信号量:    Event.acquires/releases + schedule(capacities={...})
                   acquire 于事件 start 计入, release 于配对事件 end 归还;
                   容量不足时事件推迟到下一个归还时刻.
  L2 信道 (速率服务器): **已于 2026-10-03 停用**。Event.channel_bytes 保留为
                   访存量申报 [(name, bytes, entitled_rate)], 供统计用, 但不
                   参与准入、不影响任何时长。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


#: 资源名里的池占位符后缀: "R0.AIC:*" 表示"该池的任意一个成员"。
#: 调度器在**派发时刻**把它解析成最早空闲的成员 (晚绑定), 而不是建图时定死。
POOL_WILDCARD = ":*"


def pool_key(resource: str) -> Optional[str]:
    """占位资源 -> 池名 ("R0.AIC:*" -> "R0.AIC"); 具体资源返回 None."""
    return resource[:-len(POOL_WILDCARD)] if resource.endswith(POOL_WILDCARD) else None


@dataclass
class Event:
    name: str
    resources: Tuple[str, ...]
    duration_us: float
    deps: Tuple[str, ...] = ()
    order: int = 0
    meta: Dict[str, object] = field(default_factory=dict)
    # 依赖边传播延迟(us): flag 握手 RTT, 物理量(实测 WAIT_GMM1_BUFFER median)
    dep_latency_us: float = 0.0
    # L0 逐边覆盖: [(dep_name, latency_us)], 优先于 dep_latency_us
    dep_latency_overrides: Tuple[Tuple[str, float], ...] = ()
    # L1 计数信号量: (资源名, 个数); 本事件 start 占用, 由配对 release 事件 end 归还
    acquires: Tuple[Tuple[str, int], ...] = ()
    releases: Tuple[Tuple[str, int], ...] = ()
    # 访存量申报: (通路名, 字节数, 无争用速率 B/µs)。**只做统计, 不参与准入**
    # —— 速率服务器信道已停用 (2026-10-03)。按通路汇总见
    # rank_results["traffic_bytes"]。保留申报是为了随时能重建争用模型。
    channel_bytes: Tuple[Tuple[str, float, float], ...] = ()
    # L3 晚绑定共位: 本事件必须与 colocate_with 命名的事件落在**同一核号**上。
    # 物理依据: GMM1 的结果经 L0C->UB 的 Fixpipe 硬件通路直给**配对**的 AIV0
    # (builders/activation.py: "ACT 钉在配对 GMM1 同核的 AIV0 上"), 所以 GMM1 晚绑定到
    # 核 X 时, 它的 ACT 必须落 AIV0:X。只在 resources 含池占位符时生效。
    colocate_with: Optional[str] = None
    # L3 晚绑定一次性开销: (键, us)。同一键在同一核号上只计一次 —— 本事件若是该核
    # 上第一个带此键的事件, 时长加 us (例: 每波每核的 dispatch 调用开销, 由该核
    # 在这一波做的第一段 dispatch 承担)。开销落在真正干活的核上, 而不必把事件
    # 钉死在某个核。只在晚绑定 (schedule(pools=...)) 下生效。
    once_per_core: Optional[Tuple[str, float]] = None


@dataclass(frozen=True)
class RestructureContext:
    """重构钩子的只读上下文 (策略函数不得修改返回的容器)."""
    time_us: float
    resource_free: Dict[str, float]      # 每资源当前空闲时刻
    resource_pending: Dict[str, int]     # 每资源未提交事件数
    pending: Dict[str, "Event"]          # 未提交事件视图 (name -> Event)
    committed_tail: Dict[str, str]       # 每资源最后提交的事件名
    # 已提交事件的结束时刻 (name -> end_us). 钩子判"前置是否已完成"的唯一依据:
    # ctx.pending 只说明事件未提交, 不说明它的前置已经跑完 —— 一个未提交事件的
    # 前置可能刚被提交但结束时刻还在未来。搬运只应作用于已就绪的事件, 否则搬过去
    # 的 tile 在新核上照样干等, 等于没搬。默认空 dict 保持旧钩子的向后兼容。
    end_by_name: Dict[str, float] = field(default_factory=dict)

    def ready_at(self, ev: "Event") -> float:
        """ev 的依赖就绪时刻 (逐边延迟计入); 任一前置未提交则返回 inf."""
        if not ev.deps:
            return 0.0
        t = 0.0
        for d in ev.deps:
            if d not in self.end_by_name:
                return float("inf")
            t = max(t, self.end_by_name[d] + edge_latency(ev, d))
        return t

    def is_ready(self, ev: "Event") -> bool:
        """全部前置已完成 (结束时刻不晚于当前时刻)."""
        return self.ready_at(ev) <= self.time_us


@dataclass
class RestructureAction:
    """钩子返回的重构动作: inject/cancel/add_dep.
    契约: cancel 的每个事件必须以同名重新 inject (消费者依赖才能最终满足),
    否则调度以死图错误终止."""
    inject: List["Event"] = field(default_factory=list)
    cancel: List[str] = field(default_factory=list)
    add_dep: List[Tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ScheduledEvent:
    name: str
    resources: Tuple[str, ...]
    start_us: float
    end_us: float
    dependency_ready_us: float
    resource_ready_us: float
    dependency_wait_us: float
    resource_queue_us: float
    critical_parent: Optional[str]
    critical_reason: str
    order: int
    meta: Dict[str, object]
    # L1 归因: 计数信号量等待
    capacity_wait_us: float = 0.0


def edge_latency(ev: "Event", dep: str) -> float:
    """L0: 逐边延迟, override 优先于事件级 dep_latency_us."""
    for name, lat in ev.dep_latency_overrides:
        if name == dep:
            return lat
    return ev.dep_latency_us
