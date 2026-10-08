"""建完图之后的一次规范化: 去掉在这个事件代数里**起不了约束**的计数信号量.

判据 (一条定理, 不是口味)
-------------------------
若某个 token 满足两条:

  1. **自取自还**: 每个事件对它的 acquire 份数 == release 份数 (没有"我取别人还");
  2. **共同独占**: 它的全部持有者都独占同一个资源 R;

那么它永远卡不住任何事件。因为 token 的持有区间就是事件的运行区间, 而 R 独占 ⇒
同一时刻最多一个持有者在途 ⇒ 在途份数 <= 单个持有者要的份数 <= 容量 (最后这一步由
engine 的静态校验 ``k > capacities[res]`` 保证)。

两条都满足的 token 就是**同一条约束的第二份抄本** —— R 的独占已经把它表达完了。

为什么不能留着
--------------
它改不了任何事件的起止 (实测: 把容量抬到 10**9, 四个配置的排程逐位相同), 却会改
``ScheduledEvent.actionable_us``: engine 对**带 acquires** 的事件要走
``capacity_feasible``, 于是"等我自己的核空出来"从后门被算进了 actionable_us。而
actionable_us 的契约是**不含**这一关 —— 那一关正是 ``analysis/idle.py`` 要抓的
work-conservation 等待 (有活就绪却还有核空着)。后果是本该算 avoidable 的空闲被算成
forced: 实测 28 核静态钉核, ``R0.AIC`` 的 avoidable_idle_us 由 0.0 变成 1454.67 核·µs,
而排程逐位未变。

一个改不了墙钟、却改得了"给模型打分的那个数"的机制, 比没有这个机制更糟。

判据里第 2 条不能省
-------------------
``.ld`` 相位 (GM->L1 载入) 也是自取自还的 ``MTE2:c7``, 但它 ``resources=()`` ——
那一个是**真约束**: 它是"一个 AI Core 只有一条 MTE2"这条硬件事实在图里的唯一表达,
删掉它载入就能无限并发, 墙钟会低于带宽下界 26.6% (见 analysis/bounds.py
"下界与漏账")。只按"自取自还"删会连它一起删掉。

调用时机
--------
在**晚绑定改写之前**跑 (model.py)。那时资源名还带具体核号, 第 2 条不必跟池占位符
``AIC:*`` 纠缠 —— 占位符不是独占资源 (两个事件可以绑到不同成员), 所以这里显式把它
排除在"共同独占"之外, 本函数在改写之后调用也不会给出错误答案。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from .events import POOL_WILDCARD


def _counts(pairs: Iterable[Tuple[str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tok, k in pairs:
        out[tok] = out.get(tok, 0) + k
    return out


def inert_semaphores(events: Iterable) -> Set[str]:
    """给出满足上面两条的 token 名 (不改图)."""
    holders: Dict[str, List[object]] = {}
    cross_event: Set[str] = set()
    for ev in events:
        acq = _counts(ev.acquires)
        rel = _counts(ev.releases)
        for tok in set(acq) | set(rel):
            if acq.get(tok, 0) != rel.get(tok, 0):
                cross_event.add(tok)           # 第 1 条不满足
            if acq.get(tok, 0):
                holders.setdefault(tok, []).append(ev)
    inert: Set[str] = set()
    for tok, evs in holders.items():
        if tok in cross_event:
            continue
        common: Set[str] = None                # type: ignore[assignment]
        for ev in evs:
            # 池占位符不是独占资源: "AIC:*" 的两个持有者可以落不同成员
            held = {r for r in ev.resources if not r.endswith(POOL_WILDCARD)}
            common = held if common is None else (common & held)
            if not common:
                break
        if common:
            inert.add(tok)
    return inert


def prune_inert_semaphores(events: List) -> Set[str]:
    """就地删掉起不了约束的 token, 返回删掉的 token 名.

    返回值给调用方用来同步容量表 —— 声明了却没人取的容量是另一种死声明。
    """
    dead = inert_semaphores(events)
    if not dead:
        return dead
    for ev in events:
        if any(tok in dead for tok, _ in ev.acquires):
            ev.acquires = tuple((t, k) for t, k in ev.acquires if t not in dead)
        if any(tok in dead for tok, _ in ev.releases):
            ev.releases = tuple((t, k) for t, k in ev.releases if t not in dead)
    return dead
