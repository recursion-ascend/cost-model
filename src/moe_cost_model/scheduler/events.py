"""第 2 层: 调度引擎的数据结构 — 事件 / 信道 / 重构动作 / 计划事件.

三层扩展语义:
  L0 逐边依赖延迟: Event.dep_latency_overrides [(dep_name, us)]
  L1 计数信号量:    Event.acquires/releases + schedule(capacities={...})
                   acquire 于事件 start 计入, release 于配对事件 end 归还;
                   容量不足时事件推迟到下一个归还时刻.
  L2 速率服务器信道: Event.channel_bytes [(name, bytes, entitled_rate)] +
                   schedule(channels={name: Channel}); 事件按应得速率名义时长
                   准入, 窗口内剩余带宽不足则降速或推迟, 无抢占无重定价.
                   信道为占位机制, 默认关闭.
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
    # L2 信道需求: (信道名, 字节数, 应得速率 B/µs)
    channel_bytes: Tuple[Tuple[str, float, float], ...] = ()
    # L3 晚绑定共位: 本事件必须与 colocate_with 命名的事件落在**同一核号**上。
    # 物理依据: GMM1 的结果经 L0C->UB 的 Fixpipe 硬件通路直给**配对**的 AIV0
    # (builders/activation.py: "ACT 钉在配对 GMM1 同核的 AIV0 上"), 所以 GMM1 晚绑定到
    # 核 X 时, 它的 ACT 必须落 AIV0:X。只在 resources 含池占位符时生效。
    colocate_with: Optional[str] = None


@dataclass(frozen=True)
class Channel:
    """共享带宽信道: 聚合带宽 + 并发端口上限 + 单事件速率上限.

    bw_total:          聚合带宽 (B/µs), 所有并发事件速率之和的上界
    ports:             最大并发事件数 (0 = 不限)
    max_rate_per_event: 单事件速率上限 (B/µs, 0 = 不限).
                       物理含义: 单核 MTE/访存流不可能占满聚合带宽
                       (内存控制器多通道并行). 设为 bw_total/N 即 N 路均分.

    语义: 先到先得速率预留, 无抢占无重定价. 无争用时传输时长 = bytes/entitled
    (与闭式公式一致); 争用时按窗口内剩余带宽降速或推迟.
    """
    name: str
    bw_total: float
    ports: int = 0
    max_rate_per_event: float = 0.0

    def __post_init__(self) -> None:
        if self.bw_total <= 0:
            raise ValueError(f"channel {self.name}: bw_total must be positive")
        if self.ports < 0:
            raise ValueError(f"channel {self.name}: ports must be >= 0")
        if self.max_rate_per_event < 0:
            raise ValueError(f"channel {self.name}: max_rate_per_event must be >= 0")
        cap = self.max_rate_per_event or self.bw_total
        if cap > self.bw_total:
            raise ValueError(
                f"channel {self.name}: max_rate_per_event {cap} > bw_total {self.bw_total}"
            )


@dataclass(frozen=True)
class RestructureContext:
    """重构钩子的只读上下文 (策略函数不得修改返回的容器)."""
    time_us: float
    resource_free: Dict[str, float]      # 每资源当前空闲时刻
    resource_pending: Dict[str, int]     # 每资源未提交事件数
    pending: Dict[str, "Event"]          # 未提交事件视图 (name -> Event)
    committed_tail: Dict[str, str]       # 每资源最后提交的事件名
    channel_inflight: Dict[str, float]   # 每信道在飞速率和
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
    # L1/L2 归因: 容量等待/信道等待/实际分得速率
    capacity_wait_us: float = 0.0
    channel_wait_us: float = 0.0
    channel_rate: Dict[str, float] = field(default_factory=dict)


def edge_latency(ev: "Event", dep: str) -> float:
    """L0: 逐边延迟, override 优先于事件级 dep_latency_us."""
    for name, lat in ev.dep_latency_overrides:
        if name == dep:
            return lat
    return ev.dep_latency_us


# 便捷工厂: 按核数与带宽构造 L2 信道对 (gm_to_l1 + hbm_write)
def default_channels(
    aic_num: int,
    *,
    bw_l1_gm: float,
    bw_scatter: float,
    bw_dispatch: float = 0.0,
) -> Tuple[Channel, ...]:
    """L2 默认信道: 每核应得速率 = 闭式公式所用带宽, 聚合 = ×核数.

    gm_to_l1:       GMM1/GMM2 权重+激活 GM→L1 流量
    hbm_write:      COMBINE 散射写
    dispatch_read:  dispatch 从源卡窗口读进 UB
    dispatch_write: dispatch 写本卡 workspace (token + scale + metaInfo)

    每条信道的聚合 = 它自己那组消费者满速率时刚好装下 → **无争用基线**, 事件速率
    = 应得速率, 与闭式公式逐字节一致 (test_channel_no_contention_invariance)。
    要研究争用就显式收紧聚合, 或把两组消费者并到同一条信道上 —— 后者需要该层级
    的真实聚合带宽, 不能拿"每核速率×核数"当整卡值用。
    """
    chans = [
        Channel("gm_to_l1", bw_total=bw_l1_gm * aic_num, ports=0, max_rate_per_event=bw_l1_gm),
        Channel("hbm_write", bw_total=bw_scatter * aic_num, ports=0, max_rate_per_event=bw_scatter),
    ]
    if bw_dispatch > 0:
        chans += [
            Channel("dispatch_read", bw_total=bw_dispatch * aic_num, ports=0,
                    max_rate_per_event=bw_dispatch),
            Channel("dispatch_write", bw_total=bw_dispatch * aic_num, ports=0,
                    max_rate_per_event=bw_dispatch),
        ]
    return tuple(chans)
