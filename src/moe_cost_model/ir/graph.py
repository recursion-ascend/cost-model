"""事件图的类型化视图: 把 Event 列表读成 Task / Transfer / Signal / Buffer / Dependency.

**只读**。它不参与调度, 不改任何字段 —— 所以引入它对时长与 Event.order 零影响。
用途有三个, 都要求"能按类型查询"而不是"按字符串约定猜":

  1. 校验 (步骤 5): 每份实现的 DAG 该满足的结构不变量, 用类型写出来才讲得清
     ("每个 BUFFER_SLOT 令牌的取与还必须配对、且落同一个核号")。
  2. 与实测 trace 比对 (步骤 6): trace 里是 pipe (MTE2/MTE3/FIX) 与字节数, 模型这边
     要能以同样的词表答话。
  3. 暴露缺口: 推不出来的东西标成 UNKNOWN 并计数, 而不是默认成某个值。
     vocabulary.UNREPRESENTABLE 列的是**连字段都没有**的那几项。

边的种类怎么定 —— 这是本模块唯一有判断的地方, 所以规则写在这里, 不散落:
  PHASE       消费者与生产者是**同一个 tile 的两个相位** (名字去掉 .lg/.ld/.cb/fix
              后相同)。相位名是 pipeline_expand 造的, 判据与它一致。
  BARRIER     任一端是栅栏/排空事件 (stage == "barrier" / "moe_stage_done")。
  READINESS   生产者是零时长的就绪标记 (stage 以 "_ready" 结尾)。
  CREDIT      生产者是 combine 而消费者是 gmm2 —— 只有槽位归还会形成这个方向的边
              (gmm2 的数据来自 ACT, 不来自 combine)。
  DATA        生产者与消费者的 stage 在算法链上相邻 (dispatch->gmm1->activation->
              gmm2->combine, 共享专家与尾段同理)。
  PROGRAM_ORDER 同一个角色、同一个核、同一个 stage 的相邻事件。**注意**: 这与
              pipeline_expand._drop_program_order 的猜法相同, 而那个猜法本身是近似 ——
              所以这里标 PROGRAM_ORDER 不代表建图器真的是按程序序加的这条边。
  UNKNOWN     以上都不匹配。计数不为 0 本身就是一条要看的信息。

**已知会落到 UNKNOWN 的一类, 以及为什么不去猜**: Layered 路径在 AIV1 上串了一条程序序链
(builders/comm/urma.py 的 aiv1_last[core]), 把 localcopy / maskscan / recv / combine 按
下发顺序连起来。那些边与数据边在 Event 里**长得一模一样** (都只是 deps 里的一个名字),
而这几个 stage 之间确实也可能有数据关系, 所以从结构上分不出来 —— 分得出来需要建图器在加边
时就标明 kind, 那是改 Event 的事 (会动 Event.order 与事件名, 必须单独一步)。
在那之前, 这里**标 UNKNOWN 并计数**, 不把程序序冒充成数据依赖: 前者可以重排, 后者不能,
弄错方向会让"这条边能不能去掉"的结论反过来。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .vocabulary import (DependencyKind, Engine, MemorySpace, Pipe, ResourceRef,
                         TokenKind, TokenRef, TransferRef, classify_resource,
                         classify_token, classify_transfer)

#: 算法链 (stage -> 它的下游)。出处: analysis/bounds.py 的 CHAIN 与 builders 的实际边。
_CHAIN = {
    # dispatch 的行先汇聚到 dispatch_ready (零时长的就绪标记), 再喂 gmm1 ——
    # 所以 dispatch -> dispatch_ready 是条真边, 不是推不出来的边。
    "dispatch": ("gmm1", "dispatch_ready"),
    "dispatch_call": ("dispatch",),
    "dispatch_ready": ("gmm1",),
    "dispatch_recv": ("mask_scan", "dispatch_local", "gmm1"),
    "dispatch_local": ("gmm1",),
    "mask_scan": ("dispatch_local", "gmm1"),
    "gmm1": ("activation", "gmm2"),
    "activation": ("gmm2",),
    "gmm2": ("combine", "gmm2"),
    "combine": ("epilogue",),
    "shared_gmm1": ("shared_act",),
    "shared_act": ("shared_head_done", "shared_gmm2"),
    "shared_head_done": ("dispatch_call", "dispatch"),
    "shared_gmm2": ("epilogue",),
    "epilogue": ("epilogue",),
}
_PHASE_SUFFIXES = (".lg", ".ld", ".cb", ".fix")
_BARRIER_STAGES = frozenset({"barrier", "moe_stage_done"})


def _phase_base(name: str) -> str:
    for suffix in _PHASE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@dataclass(frozen=True)
class TaskNode:
    """一个事件的类型化读法: 谁在哪条管道上做多久, 读写了什么, 占了哪些令牌."""

    name: str
    kind: str                                  # 原 meta["stage"], IR 里的 Task.kind
    phase: Optional[str]                       # 相位 (lg/ld/cb/fix), 无则 None
    duration_us: float
    resources: Tuple[ResourceRef, ...]
    unresolved_resources: Tuple[str, ...]      # 认不出来的资源名 (不猜)
    acquires: Tuple[TokenRef, ...]
    releases: Tuple[TokenRef, ...]
    unresolved_tokens: Tuple[str, ...]
    transfers: Tuple[TransferRef, ...]
    colocate_with: Optional[str]
    core_group: Optional[Tuple[str, str]]
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def engines(self) -> Tuple[Engine, ...]:
        return tuple(sorted({r.engine for r in self.resources}, key=lambda e: e.value))

    @property
    def is_barrier(self) -> bool:
        return self.kind in _BARRIER_STAGES or (
            self.duration_us == 0.0 and not self.resources and not self.transfers)

    def bytes_on(self, direction) -> float:
        return sum(t.bytes for t in self.transfers if t.direction is direction)


@dataclass(frozen=True)
class DependencyEdge:
    producer: str
    consumer: str
    kind: DependencyKind
    latency_us: float = 0.0


@dataclass(frozen=True)
class BufferLifetime:
    """一个缓冲槽令牌的持有区间: 谁取、谁还、核号、以及是否配对.

    这是 IR 要表达的"buffer acquire/release lifetime"。现有机制里它是**隐式**的:
    台账按令牌名全局结算 (engine.py), 取和还可以在两个不同事件里, 没人检查配对。
    """

    token: str
    kind: TokenKind
    space: Optional[MemorySpace]
    pipe: Optional[Pipe]
    core: Optional[int]
    acquired_by: Tuple[str, ...]
    released_by: Tuple[str, ...]

    @property
    def paired(self) -> bool:
        """取与还的次数相等. 不等 = 台账会漂 (负数或永不归还), 是真缺陷."""
        return len(self.acquired_by) == len(self.released_by)

    @property
    def cross_event(self) -> bool:
        """跨事件持有 (取和还不是同一个事件) —— 这才是真正的缓冲槽语义."""
        return set(self.acquired_by) != set(self.released_by)


class EventGraphView:
    """Event 列表 (或排好的 ScheduledEvent) 的类型化只读视图."""

    def __init__(self, events: Sequence[object]):
        self.tasks: Tuple[TaskNode, ...] = tuple(self._read(e) for e in events)
        self._by_name = {t.name: t for t in self.tasks}
        # 依赖边只在**建图期**的 Event 上 (deps 字段); ScheduledEvent 是排程输出, 不带
        # deps (scheduler/events.py 的两个 dataclass)。所以在排好的事件上建视图时边是空的 ——
        # 这必须说出来, 否则 "edges: 0" 会被读成"这张图没有依赖"。
        self.has_dependencies: bool = any(hasattr(e, "deps") for e in events)
        self.edges: Tuple[DependencyEdge, ...] = (
            tuple(self._edges(events)) if self.has_dependencies else ())

    # ---- 读单个事件 ----
    @staticmethod
    def _read(ev) -> TaskNode:
        meta = dict(getattr(ev, "meta", {}) or {})
        res, bad_res = [], []
        for raw in getattr(ev, "resources", ()) or ():
            got = classify_resource(raw)
            (res if got else bad_res).append(got or raw)
        acq, rel, bad_tok = [], [], []
        for raw, count in getattr(ev, "acquires", ()) or ():
            got = classify_token(raw, count)
            (acq if got else bad_tok).append(got or raw)
        for raw, count in getattr(ev, "releases", ()) or ():
            got = classify_token(raw, count)
            (rel if got else bad_tok).append(got or raw)
        transfers = tuple(classify_transfer(c, b)
                          for c, b, *_ in getattr(ev, "channel_bytes", ()) or ())
        start = getattr(ev, "start_us", None)
        dur = (getattr(ev, "end_us", 0.0) - start) if start is not None \
            else float(getattr(ev, "duration_us", 0.0))
        group = getattr(ev, "core_group", None)
        return TaskNode(
            name=ev.name, kind=str(meta.get("stage", "")),
            phase=(str(meta["phase"]) if meta.get("phase") else None),
            duration_us=float(dur), resources=tuple(res),
            unresolved_resources=tuple(bad_res), acquires=tuple(acq), releases=tuple(rel),
            unresolved_tokens=tuple(bad_tok), transfers=transfers,
            colocate_with=getattr(ev, "colocate_with", None),
            core_group=(tuple(group) if group else None), meta=meta)

    # ---- 边的种类 ----
    def _edges(self, events) -> Iterable[DependencyEdge]:
        for ev in events:
            consumer = self._by_name.get(ev.name)
            for dep in getattr(ev, "deps", ()) or ():
                producer = self._by_name.get(dep)
                yield DependencyEdge(
                    producer=dep, consumer=ev.name,
                    kind=self._edge_kind(producer, consumer),
                    latency_us=self._edge_latency(ev, dep))

    @staticmethod
    def _edge_latency(ev, dep) -> float:
        for name, us in getattr(ev, "dep_latency_overrides", ()) or ():
            if name == dep:
                return float(us)
        return 0.0

    @staticmethod
    def _edge_kind(producer: Optional[TaskNode],
                   consumer: Optional[TaskNode]) -> DependencyKind:
        if producer is None or consumer is None:
            return DependencyKind.UNKNOWN
        if _phase_base(producer.name) == _phase_base(consumer.name) and (
                producer.phase or consumer.phase):
            return DependencyKind.PHASE
        if producer.kind in _BARRIER_STAGES or consumer.kind in _BARRIER_STAGES:
            return DependencyKind.BARRIER
        if producer.kind.endswith("_ready") or consumer.kind.endswith("_ready"):
            # 两侧都算: 汇聚到就绪标记的边与从它出去的边都是"就绪"语义, 不是数据搬运。
            return DependencyKind.READINESS
        if producer.kind == "combine" and consumer.kind == "gmm2":
            return DependencyKind.CREDIT
        if consumer.kind in _CHAIN.get(producer.kind, ()):
            if producer.kind == consumer.kind:
                same_core = (producer.meta.get("core") == consumer.meta.get("core")
                             and producer.meta.get("core") is not None)
                return (DependencyKind.PROGRAM_ORDER if same_core
                        else DependencyKind.DATA)
            return DependencyKind.DATA
        if (producer.kind == consumer.kind
                and producer.meta.get("core") == consumer.meta.get("core")
                and producer.meta.get("core") is not None):
            return DependencyKind.PROGRAM_ORDER
        return DependencyKind.UNKNOWN

    # ---- 查询 ----
    def buffers(self) -> Tuple[BufferLifetime, ...]:
        """按令牌名聚出缓冲槽/执行单元的持有关系."""
        acq: Dict[str, List[str]] = {}
        rel: Dict[str, List[str]] = {}
        info: Dict[str, TokenRef] = {}
        for task in self.tasks:
            for tok in task.acquires:
                acq.setdefault(tok.raw, []).append(task.name)
                info.setdefault(tok.raw, tok)
            for tok in task.releases:
                rel.setdefault(tok.raw, []).append(task.name)
                info.setdefault(tok.raw, tok)
        out = []
        for raw in sorted(set(acq) | set(rel)):
            tok = info[raw]
            out.append(BufferLifetime(
                token=raw, kind=tok.kind, space=tok.space, pipe=tok.pipe, core=tok.core,
                acquired_by=tuple(acq.get(raw, ())), released_by=tuple(rel.get(raw, ()))))
        return tuple(out)

    def transfer_totals(self) -> Dict[Tuple[Optional[MemorySpace], Optional[MemorySpace]], float]:
        """按 (源, 目的) 聚字节 —— 原先只能按通路名聚, 方向藏在名字里."""
        out: Dict[Tuple[Optional[MemorySpace], Optional[MemorySpace]], float] = {}
        for task in self.tasks:
            for tr in task.transfers:
                key = (tr.src, tr.dst)
                out[key] = out.get(key, 0.0) + tr.bytes
        return out

    def summary(self) -> Dict[str, object]:
        """一张可读的小结, 含**推不出来的计数** (不为 0 就该看)."""
        return {
            "tasks": len(self.tasks),
            "edges": len(self.edges),
            # False = 输入是排好的 ScheduledEvent, 本来就不带 deps, 边数 0 不代表无依赖
            "has_dependencies": self.has_dependencies,
            "task_kinds": dict(sorted(Counter(t.kind for t in self.tasks).items())),
            "edge_kinds": dict(sorted(
                Counter(e.kind.value for e in self.edges).items())),
            "token_kinds": dict(sorted(
                Counter(b.kind.value for b in self.buffers()).items())),
            "unresolved_resources": sorted(
                {r for t in self.tasks for r in t.unresolved_resources}),
            "unresolved_tokens": sorted(
                {r for t in self.tasks for r in t.unresolved_tokens}),
            "unpaired_buffers": [b.token for b in self.buffers() if not b.paired],
        }
