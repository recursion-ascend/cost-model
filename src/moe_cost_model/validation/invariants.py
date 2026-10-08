"""逐实现的 DAG 结构校验: 用 IR 的词表写出"这张图必须满足什么".

与现有校验的分工 —— builders 里已经有不少**建图期**断言 (gmm1 要求每个 (专家, m-group)
有一个 dispatch_ready; gmm2 的 _require_full_k 要求 ACT 覆盖整个 K; comm/peerwrite 查重复生产者
与行数守恒)。它们查的是"这一步算对了吗", 而且散在各个建图器里。

本模块查的是**整张图成形之后的结构**, 并且用的是与 kernel 无关的词表 (ir/vocabulary):

  C1 缓冲槽的取与还必须配对, 且落同一个核号
     —— 不配对 = 台账会漂 (计数变负或永不归还), 约束就悄悄失效了。
        pipeline_expand 里记着这个 bug 的一次真实发生: 拆相位时丢掉 carried acquire。
  C2 执行单元令牌的容量只能是 1
     —— 那是硬件事实 (每核每种管道一条), 不是参数。写成 >1 等于声称硬件有两条 MTE2。
  C3 共位事件与锚点必须同核 (晚绑定下由派发时刻保证)
  C4 事件名唯一, 依赖指向存在的事件, 无自环
  C5 有搬运申报的事件必须说得出两端内存 (unknown 通路 = 新约定没登记)
  C6 零时长事件不得占执行单元 (占了会让 idle 归因失真)

每条都给**反例会怎样**, 因为"为什么要这条"比"这条是什么"更容易丢。
校验器不认识 MegaMoE 的 stage 名: 它只用 TokenKind / MemorySpace 这些类型, 所以换一份
实现照样能用 —— 这正是四层架构要的那种校验。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..ir.graph import EventGraphView
from ..ir.vocabulary import MemorySpace, TokenKind


@dataclass(frozen=True)
class Violation:
    rule: str
    detail: str
    consequence: str


def _core_of_token(raw: str) -> Optional[int]:
    tail = raw.rsplit(":", 1)[-1]
    if tail.startswith("c") and tail[1:].isdigit():
        return int(tail[1:])
    return None


def check_graph(events: Sequence[object],
                capacities: Optional[Dict[str, int]] = None) -> List[Violation]:
    """对一张事件图跑全部结构校验, 返回违规列表 (空 = 通过)."""
    view = EventGraphView(events)
    out: List[Violation] = []

    # C4 名字与边
    names = [t.name for t in view.tasks]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        out.append(Violation("C4 名字唯一", f"重名事件 {dupes[:5]}",
                             "调度器按名字解析依赖, 重名会让边指向错误的事件"))
    known = set(names)
    for edge in view.edges:
        if edge.producer not in known:
            out.append(Violation("C4 依赖存在",
                                 f"{edge.consumer} 依赖不存在的 {edge.producer}",
                                 "调度器会直接报错; 若被静默忽略则是一条丢失的约束"))
        if edge.producer == edge.consumer:
            out.append(Violation("C4 无自环", f"{edge.consumer} 依赖自己",
                                 "自环事件永不就绪, 整图死锁"))

    # C1 / C2 令牌
    for buf in view.buffers():
        if buf.kind is TokenKind.EXECUTION_UNIT:
            cap = (capacities or {}).get(buf.token)
            if cap is not None and cap != 1:
                out.append(Violation(
                    "C2 执行单元容量为 1", f"{buf.token} 容量 {cap}",
                    "每核每种管道只有一条 (硬件事实); 写成 >1 等于声称硬件多了一条通路, "
                    "会凭空给出重叠"))
            continue
        if buf.kind is not TokenKind.BUFFER_SLOT:
            continue
        if not buf.paired:
            out.append(Violation(
                "C1 缓冲槽取还配对",
                f"{buf.token}: 取 {len(buf.acquired_by)} 次, 还 {len(buf.released_by)} 次",
                "台账按令牌名全局结算, 取还不等会让计数漂 (变负或永不归还), "
                "这个约束于是悄悄失效 —— 表现是更快的排程, 不是报错"))
        cores = {_core_of_token(buf.token)}
        for owner in buf.acquired_by + buf.released_by:
            task = next((t for t in view.tasks if t.name == owner), None)
            if task is None:
                continue
            for res in task.resources:
                if res.core is not None:
                    cores.add(res.core)
        cores.discard(None)
        if len(cores) > 1:
            out.append(Violation(
                "C1 缓冲槽同核", f"{buf.token} 牵涉核号 {sorted(cores)}",
                "片上缓冲是按核的; 跨核取还等于声称一个核能还另一个核的槽"))

    # C3 共位
    by_name = {t.name: t for t in view.tasks}
    for task in view.tasks:
        if not task.colocate_with:
            continue
        anchor = by_name.get(task.colocate_with)
        if anchor is None:
            out.append(Violation("C3 共位锚点存在",
                                 f"{task.name} 共位到不存在的 {task.colocate_with}",
                                 "锚点缺失时共位静默失效, 事件会落到任意核"))
            continue
        mine = {r.core for r in task.resources if r.core is not None}
        theirs = {r.core for r in anchor.resources if r.core is not None}
        if mine and theirs and not (mine & theirs):
            out.append(Violation(
                "C3 共位同核",
                f"{task.name} 在核 {sorted(mine)}, 锚点 {anchor.name} 在 {sorted(theirs)}",
                "共位表达的是硬件通路 (L0C->UB Fixpipe 只在绑定对内), 不同核就是物理错误"))

    # C5 搬运两端
    unknown = sorted({t.channel for task in view.tasks for t in task.transfers
                      if t.src is None or t.dst is None})
    if unknown:
        out.append(Violation(
            "C5 搬运两端已知", f"说不出两端内存的通路: {unknown[:5]}",
            "通路名是新约定但没在 ir/vocabulary.CHANNEL_SEMANTICS 登记; "
            "字节会照样累加, 但方向与两端丢失, 下界与访存核算都用不上它"))

    # C6 零时长不占执行单元
    for task in view.tasks:
        if task.duration_us > 0:
            continue
        units = [a.raw for a in task.acquires if a.kind is TokenKind.EXECUTION_UNIT]
        if units:
            out.append(Violation(
                "C6 零时长不占单元", f"{task.name} 时长 0 却占 {units}",
                "零时长事件被 idle 归因排除在占用之外, 占了执行单元会让「这条管道忙了多久」"
                "算不平"))
    return out


def check_adapter(adapter, shape, costs, options,
                  capacities: Optional[Dict[str, int]] = None) -> List[Violation]:
    """给一个适配器建图并校验 (步骤 5 的入口)."""
    from ..implementations import CompileConfig
    compile_cfg = CompileConfig.from_kernel_config(getattr(shape, "kernel", None))
    adapter.accepts(compile_cfg, options)
    plan = adapter.plan(shape, compile_cfg, options)
    events, _ = adapter.lower(shape, plan, costs, options)
    return check_graph(events, capacities)
