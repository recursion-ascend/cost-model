"""调度引擎的数据结构: Event / ScheduledEvent / 重构钩子的上下文与动作.

Event 是模型的表达能力上界 —— 表达不出来的约束, 模型就不声称。它能表达四种:

    资源独占      resources: 事件跑完之前别人用不了这些资源
    数据依赖      deps + 依赖边上的传播延迟
    容量           acquires / releases: 计数信号量, 占多少还多少
    落核约束      colocate_with / core_group: 必须与谁同核

channel_bytes 是第五种的残留: 带宽争用曾经建模过, 现在只申报字节、不影响时长。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


#: 池占位符后缀。"R0.AIC:*" = "该池的任意一个成员", 由调度器在派发时刻选定 (晚绑定);
#: "R0.AIC:7" = 第 7 号核, 建图时就定死 (静态钉核)。
POOL_WILDCARD = ":*"


def pool_key(resource: str) -> Optional[str]:
    """占位资源 -> 池名 ("R0.AIC:*" -> "R0.AIC"); 具体资源返回 None."""
    return resource[:-len(POOL_WILDCARD)] if resource.endswith(POOL_WILDCARD) else None


@dataclass
class Event:
    """一份工作: 要哪些资源、等谁、占多久。

    name 是工作的身份, **不带核号** —— 落哪个核是调度的产出, 不是工作的属性。
    """

    name: str
    #: 本事件独占的资源; 含 ":*" 的是池占位符 (晚绑定)
    resources: Tuple[str, ...]
    duration_us: float
    #: 前置事件名
    deps: Tuple[str, ...] = ()
    #: 并列时的确定性 tie-break
    order: int = 0
    meta: Dict[str, object] = field(default_factory=dict)
    #: 每条依赖边的传播延迟: flag 握手的 RTT (实测 WAIT_GMM1_BUFFER 中位数)
    dep_latency_us: float = 0.0
    #: 逐边覆盖 [(前置名, 延迟)], 优先于 dep_latency_us
    dep_latency_overrides: Tuple[Tuple[str, float], ...] = ()
    #: 计数信号量 (资源名, 个数): start 时占用, 由配对事件 end 时归还。
    #: 容量不足则推迟到下一个归还时刻。表达的是**容量**, 不是程序序。
    acquires: Tuple[Tuple[str, int], ...] = ()
    releases: Tuple[Tuple[str, int], ...] = ()
    #: 访存量申报 (通路名, 字节数, 无争用速率 B/µs)。**只做统计**: 不参与准入,
    #: 不影响任何时长。汇总见 rank_results["traffic_bytes"]。
    channel_bytes: Tuple[Tuple[str, float, float], ...] = ()
    #: 必须与这个事件落同一核号。物理依据: GMM1 的结果经 Fixpipe (L0C->UB) 直给
    #: **配对**的 AIV0, 这条通路只在绑定对内存在。仅当 resources 含占位符时生效。
    colocate_with: Optional[str] = None
    #: (组名, 角色): 同组事件落同一核号, 核号由组内**最先派发**的事件选定。
    #: 与 colocate_with 的区别是不要求锚点先绑定 —— 相位拆分里先跑的恰是不持核
    #: 资源的那一相 (lg/ld 先于 cb), 它需要在派发时刻拿到核号, 名字带 "c*" 的
    #: 按核信号量才扣得对。角色是池名的后半段 (如 "AIC"), 用来给出候选核表。
    core_group: Optional[Tuple[str, str]] = None
    #: (键, us): 同一键在同一核号上只计一次 —— 本事件若是该核上第一个带此键的,
    #: 时长加 us。用于每核一次的开销 (如每波每核的 dispatch 调用开销), 让开销落在
    #: 真正干活的核上, 而不必把事件钉死在某个核。仅晚绑定下生效。
    once_per_core: Optional[Tuple[str, float]] = None


@dataclass(frozen=True)
class RestructureContext:
    """重构钩子的只读上下文 (钩子不得修改这些容器)."""

    time_us: float
    resource_free: Dict[str, float]      #: 每资源当前空闲时刻
    resource_pending: Dict[str, int]     #: 每资源未提交事件数
    pending: Dict[str, "Event"]          #: 未提交事件 (name -> Event)
    committed_tail: Dict[str, str]       #: 每资源最后提交的事件名
    #: 已提交事件的结束时刻。判"前置跑完了没有"只能看这个: pending 只说明事件
    #: 未提交, 不说明它的前置已经结束。把没就绪的 tile 搬去别的核, 它在新核上
    #: 照样干等。
    end_by_name: Dict[str, float] = field(default_factory=dict)

    def ready_at(self, ev: "Event") -> float:
        """ev 的依赖就绪时刻 (计入逐边延迟); 任一前置未提交则 inf."""
        if not ev.deps:
            return 0.0
        t = 0.0
        for d in ev.deps:
            if d not in self.end_by_name:
                return float("inf")
            t = max(t, self.end_by_name[d] + edge_latency(ev, d))
        return t

    def is_ready(self, ev: "Event") -> bool:
        """全部前置已在当前时刻之前结束."""
        return self.ready_at(ev) <= self.time_us


@dataclass
class RestructureAction:
    """钩子要求的图改动.

    契约: cancel 掉的事件必须以同名重新 inject —— 否则消费者的依赖永远满足不了,
    调度会以死图报错终止。
    """

    inject: List["Event"] = field(default_factory=list)
    cancel: List[str] = field(default_factory=list)
    add_dep: List[Tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ScheduledEvent:
    """排好的事件: 起止时刻 + 这段等待该归因给谁."""

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
    #: 等计数信号量的时长 (从探测起点算)
    capacity_wait_us: float = 0.0
    #: 本事件"真能动"的最早时刻: 依赖已就绪 且 信号量已可准入。
    #: **不含"自己要的资源空出来"** —— 那一关由 analysis/idle.py 判, 它正是
    #: work-conservation 要抓的那种等待 (有活就绪却还有核空着)。
    actionable_us: float = 0.0
    #: 绑定后的信号量 token。analysis 判"某个空闲核能不能接这个活"时, 要把按核的
    #: token 换成那个核的名字再查余量。
    acquires: Tuple[Tuple[str, int], ...] = ()
    releases: Tuple[Tuple[str, int], ...] = ()


def edge_latency(ev: "Event", dep: str) -> float:
    """这条依赖边的延迟: 逐边覆盖优先于事件级 dep_latency_us."""
    for name, lat in ev.dep_latency_overrides:
        if name == dep:
            return lat
    return ev.dep_latency_us
