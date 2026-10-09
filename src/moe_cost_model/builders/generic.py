"""第 4 层: 词汇表驱动的缺省建图器 —— 声明一条流水, 自动得到事件图.

手写建图器 (builders/mte.py, layered.py 与它们调的 gmm1/activation/gmm2/comm) 一共
两千多行, 里面大部分做的是同一件事: 按单元切工作、按边连依赖、按角色落资源、按粒度
合并、按片上深度挂信号量。这些**全部**已经在 StageVocabulary 与 StageLink 里声明过了,
所以可以生成, 不必每份实现重写一遍。

本模块就是那个生成器。一份实现只需要给三样东西:

  词汇表    StageVocabulary  —— 有哪些 stage、谁连谁、共享轴是什么、落哪个角色
  工作枚举  items(stage, shape) -> [WorkItem]  —— 这个形状下这个 stage 要做哪些份活
  成本      cost(stage, item) -> float         —— 一份活多久

拿不到的那部分 (某份实现特有的尾段、预取、跨卡握手) 由调用方在生成之后自己补事件,
或者整条路径仍写手写建图器 —— 生成器是缺省, 不是唯一路径。

**让 SharedAxis.axis 变成可执行的**: 词汇表里每条边都声明了共享轴的名字 (m / n / expert
...), 原先只用于报错与文档。这里让 WorkItem 按**轴名**携带区间, 于是"消费者等哪些生产者"
就是一条与算子无关的规则: 同一切片内, 共享轴上的区间相交即连边。生成器因此不认识任何
stage 名, 也不认识 m 和 n 的含义 —— 它只做区间相交。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Callable, Dict, Hashable, List, Mapping, Optional,
                    Sequence, Tuple)

from ..config.links import StageLink, resolve_link
from ..config.stages import StageVocabulary
from ..scheduler.events import Event

Interval = Tuple[int, int]


@dataclass(frozen=True)
class WorkItem:
    """一个 stage 在一个切片上的一份自然工作单元.

    slice_id: 边只在同一切片内连 (MegaMoE 里是专家切片; 另一份实现可以是别的分组)。
        不同切片之间要连边, 说明那是另一条 stage 边, 该在词汇表里声明。
    axes: 轴名 -> 左闭右开区间。词汇表里某条边声明的共享轴, 两端都必须有这个键 ——
        缺了就报错, 不猜。区间相交 = 有依赖。
    core: 落哪个核号。生成器不决定分核 (那是 planning/core_assignment 或晚绑定的事),
        调用方枚举工作时给。
    """

    slice_id: Hashable
    axes: Mapping[str, Interval]
    core: int = 0
    label: str = ""
    meta: Mapping[str, object] = field(default_factory=dict)

    def span(self, axis: str) -> Interval:
        try:
            return self.axes[axis]
        except KeyError:
            raise ValueError(
                f"工作项 {self.label or self.slice_id!r} 没有轴 {axis!r} 的区间; "
                f"它有 {sorted(self.axes)} —— 词汇表里声明了这条轴的边, "
                "两端的工作项都必须带它") from None


def _overlap(a: Interval, b: Interval) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _depends(prod: WorkItem, cons: WorkItem, shared: str) -> bool:
    """消费者这一份活要不要等生产者那一份.

    判据是**两者共有的每条轴都相交**, 不只是声明的共享轴。共享轴是那条允许"只覆盖一
    部分"的轴 (分段就绪沿它发生); 其余轴只是位置, 位置不同就是另一块输出 —— 只看共享
    轴会把同一列、不同行的生产者也连进来, 那是一条不存在的依赖。
    """
    common = set(prod.axes) & set(cons.axes)
    if shared not in common:
        raise ValueError(
            f"边的共享轴 {shared!r} 不在两端共有的轴里 (生产者 {sorted(prod.axes)}, "
            f"消费者 {sorted(cons.axes)})")
    return all(_overlap(prod.span(k), cons.span(k)) for k in common)


def _merge(items: Sequence[WorkItem], pack_axis: str, cap: int) -> List[List[WorkItem]]:
    """按粒度把工作项合并成"事件的成员列表".

    合并规则与 builders/tiling.coalesce_tiles 同一条: 只合并**除打包轴外所有轴区间
    相同、且在打包轴上相邻**的连续项。理由也相同 —— 轴区间不同就不是一块连续的输出,
    合并后的区间会盖住没算的部分, 下游按区间相交挑依赖就会错。

    cap 是上界不是配额: 遇到合不拢的边界就截断, 所以组内个数可能少于 cap。
    """
    if cap < 1:
        raise ValueError(f"事件粒度必须 >= 1, 得到 {cap}")
    out: List[List[WorkItem]] = []
    cur: List[WorkItem] = []
    for it in items:
        if cur and len(cur) < cap and _mergeable(cur[-1], it, pack_axis):
            cur.append(it)
            continue
        if cur:
            out.append(cur)
        cur = [it]
    if cur:
        out.append(cur)
    return out


def _mergeable(prev: WorkItem, nxt: WorkItem, pack_axis: str) -> bool:
    """两项能不能并进同一个事件: 同切片、同核、其余轴相同、打包轴首尾相接."""
    if prev.slice_id != nxt.slice_id or prev.core != nxt.core:
        return False
    if set(prev.axes) != set(nxt.axes) or pack_axis not in prev.axes:
        return False
    if any(prev.axes[k] != nxt.axes[k] for k in prev.axes if k != pack_axis):
        return False
    return prev.span(pack_axis)[1] == nxt.span(pack_axis)[0]


def _fuse(group: Sequence[WorkItem], pack_axis: str) -> WorkItem:
    """合并后那一份活的区间: 打包轴取并集, 其余轴照抄 (已校验相同)."""
    first = group[0]
    if len(group) == 1:
        return first
    axes = dict(first.axes)
    axes[pack_axis] = (min(w.span(pack_axis)[0] for w in group),
                       max(w.span(pack_axis)[1] for w in group))
    return WorkItem(slice_id=first.slice_id, axes=axes, core=first.core,
                    label=first.label, meta=dict(first.meta))


@dataclass(frozen=True)
class PipelineSpec:
    """一条流水的完整声明 —— 够生成事件图.

    pack_axis: 每个 stage 的粒度沿哪条轴打包。缺省取该 stage 第一条出边的共享轴
        (那是下游真正在看的那条轴, 沿它合并才不会把一个依赖拆成两半)。
    """

    vocab: StageVocabulary
    items: Callable[[str, object], Sequence[WorkItem]]
    cost: Callable[[str, WorkItem], float]
    pack_axis: Mapping[str, str] = field(default_factory=dict)

    def axis_to_pack(self, stage: str) -> str:
        if stage in self.pack_axis:
            return self.pack_axis[stage]
        for (p, _c), ax in self.vocab.edges.items():
            if p == stage:
                return ax.axis
        for (_p, c), ax in self.vocab.edges.items():
            if c == stage:
                return ax.axis
        raise ValueError(
            f"stage {stage!r} 既没有出边也没有入边, 无法推出打包轴; "
            "在 PipelineSpec.pack_axis 里显式给")


@dataclass
class _Node:
    name: str
    item: WorkItem
    event: Event


def lower_pipeline(spec: PipelineSpec, shape, options, *,
                   links: Sequence[StageLink] = ()) -> List[Event]:
    """按声明生成一张事件图.

    两趟: 先把每个 stage 的工作切好、落核、建出事件 (此时还没有边), 再按词汇表的 edges
    接线。分两趟是因为片上缓冲的信号量**生产者取、消费者还** —— 接线时要回头改生产者,
    一趟做不到。

    每件事都由声明决定:
      切    items() 枚举 + granularity 沿打包轴合并
      落    资源 = options.role_resource(stage, core), 角色来自词汇表 roles
      连    入边 = 词汇表 edges 里以本 stage 为消费者的那些; 同切片内共享轴区间相交
      存    边上 location="onchip" 且 depth>0 -> 生产者 acquires, 消费者 releases
      钉    边上 colocated_by_hardware -> 消费者 colocate_with 那个生产者
    """
    vocab = spec.vocab
    gran = getattr(options, "granularity", None)
    nodes: Dict[str, Dict[Hashable, List[_Node]]] = {}
    events: List[Event] = []
    order = 0

    # ---- 第一趟: 切、落核、建事件 (无边) ----
    for stage in vocab.pipeline:
        pack = spec.axis_to_pack(stage)
        cap = int(gran.items(stage)) if gran is not None else 1
        if cap < 1:
            raise ValueError(
                f"stage {stage!r} 的事件粒度是 {cap}; 生成器只接受 >= 1 —— "
                "整片/沿用 tiling 这类语义由调用方在枚举工作时表达, 不用魔数")
        has_role = stage in vocab.roles
        by_slice: Dict[Hashable, List[WorkItem]] = {}
        for it in spec.items(stage, shape):
            by_slice.setdefault(it.slice_id, []).append(it)
        nodes[stage] = {}
        for sid in by_slice:
            for group in _merge(by_slice[sid], pack, cap):
                item = _fuse(group, pack)
                suffix = f".{item.label}" if item.label else ""
                name = f"{stage}.s{sid}{suffix}"
                ev = Event(
                    name=name,
                    resources=((options.role_resource(stage, item.core),)
                               if has_role else ()),
                    duration_us=float(spec.cost(stage, item)),
                    order=order,
                    meta={"stage": stage, "slice": sid, "core": item.core,
                          "members": len(group), **dict(item.meta)})
                events.append(ev)
                nodes[stage].setdefault(sid, []).append(_Node(name, item, ev))
                order += 1

    # ---- 第二趟: 按 edges 接线 ----
    for (prod, cons), ax in vocab.edges.items():
        link = resolve_link(links, prod, cons) if links else None
        onchip = (link is not None
                  and getattr(link, "location", "gm") == "onchip"
                  and int(getattr(link, "depth", 0) or 0) > 0)
        pinned = link is not None and getattr(link, "colocated_by_hardware", False)
        for sid, cs in nodes.get(cons, {}).items():
            ps = nodes.get(prod, {}).get(sid, ())
            for cn in cs:
                hit = [pn for pn in ps if _depends(pn.item, cn.item, ax.axis)]
                if not hit:
                    continue
                cn.event.deps = tuple(cn.event.deps) + tuple(n.name for n in hit)
                if pinned and cn.event.colocate_with is None:
                    cn.event.colocate_with = hit[0].name
                if onchip:
                    # token 按**生产者**的核命名: 缓冲是生产者写的, 它属于那个核。
                    # 这条边共位时两边核号相同, 不共位时归还仍记在生产者那一份上。
                    for pn in hit:
                        tok = (f"ONCHIP:{prod}_{cons}:c{pn.item.core}", 1)
                        if tok not in pn.event.acquires:
                            pn.event.acquires = tuple(pn.event.acquires) + (tok,)
                        if tok not in cn.event.releases:
                            cn.event.releases = tuple(cn.event.releases) + (tok,)
    return events


def onchip_capacities(spec: PipelineSpec, options, cores: int, *,
                      links: Sequence[StageLink] = ()) -> Dict[str, int]:
    """生成图配套的信号量容量表: 每条片上边每个核一份, 容量 = 那条边声明的 depth."""
    caps: Dict[str, int] = {}
    for (prod, cons) in spec.vocab.edges:
        link = resolve_link(links, prod, cons) if links else None
        depth = int(getattr(link, "depth", 0) or 0) if link is not None else 0
        if link is None or getattr(link, "location", "gm") != "onchip" or depth <= 0:
            continue
        for c in range(cores):
            caps[f"ONCHIP:{prod}_{cons}:c{c}"] = depth
    return caps
