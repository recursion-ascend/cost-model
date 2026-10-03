"""第 4.5 层: 全核栅栏原语 (C3) — 把"之前的全部"与"之后的全部"隔开.

为什么需要它
------------
图里原本只有**每核**的排空语义 (内核的 WAIT_GMM_DRAIN), 没有全核栅栏 —— kernel 里那两个
`SyncAll` 都在波循环**外面**, 按"前导/尾段不入图"的口径退场了。于是表达不了:

  * **分段式执行**: 像 DeepEP-Ascend 那样把 dispatch / GMM / combine 拆成几个独立 kernel,
    段间全核对齐。这正是"融合算子值多少"这个问题 —— 要回答它, 两边都得算得出来。
  * **波间全核对齐**: 下一波的任何事件都等上一波全部做完 (而不是现在的逐核推进)。

栅栏就是一个**零时长、不占资源**的事件, 特殊之处全在边的形状上:

    前面那一组的全部  ->  [栅栏]  ->  后面那一组的每一个

扇入保证"全做完", 扇出保证"谁都不许提前"。不需要新的调度机制。

注意扇入必须挂**全部**成员而不是"最后一个": 建图序不等于时间序 (同核先后由资源互斥在
调度期决定), 只挂最后建的那个会漏 —— `_add_completion` 的注释里记过同一个坑。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..scheduler.events import Event

#: "stage" 粒度的段序 (一个波内). 不在表里的 stage 不参与分段。
STAGE_ORDER: Tuple[str, ...] = ("dispatch", "gmm1", "activation", "gmm2", "combine")

KINDS: Tuple[str, ...] = ("wave", "stage")


def apply_barriers(events: List[Event], kinds: Sequence[str], *,
                   ub_depth: int = 0, aic_num: int = 0,
                   pooled: bool = False) -> List[Event]:
    """就地给 events 加栅栏, 返回新的事件表 (含栅栏节点).

    kinds:
      "wave"  每两个相邻波之间一道全核栅栏 —— 下一波的任何事件都等上一波全做完
      "stage" **波内**每个 stage 之后一道 (dispatch -> gmm1 -> activation -> gmm2 -> combine)
    两者可同时给。空序列 = 不加 (缺省, 即现有的逐核推进)。

    ub_depth / aic_num / pooled 只用于 "stage" 的可行性判定, 见 _check_stage_feasible。
    """
    kinds = tuple(kinds)
    bad = sorted(set(kinds) - set(KINDS))
    if bad:
        raise ValueError(f"barriers 只支持 {KINDS}, 收到 {bad}")
    if not kinds:
        return events

    by_wave: Dict[int, List[Event]] = {}
    rank = None
    for ev in events:
        if rank is None:
            rank = ev.meta.get("rank")
        w = ev.meta.get("wave")
        if isinstance(w, int) and str(ev.meta.get("stage", "")) in STAGE_ORDER:
            by_wave.setdefault(w, []).append(ev)
    if not by_wave:
        return events
    # 名字与 meta 要和 builders/base.py 的 _event 一致: 名字带 R{rank}. 前缀,
    # meta 带 rank —— 否则 simulate 按 rank 过滤事件时这些栅栏会被整个丢掉。
    pre = f"R{rank}." if rank is not None else ""
    waves = sorted(by_wave)
    new: List[Event] = []
    order = max((ev.order for ev in events), default=0)

    if "wave" in kinds:
        for prev, cur in zip(waves, waves[1:]):
            order += 1
            name = f"{pre}barrier.wave{cur}"
            new.append(Event(name=name, resources=(), duration_us=0.0,
                             deps=tuple(e.name for e in by_wave[prev]), order=order,
                             meta={"stage": "barrier", "kind": "wave", "wave": cur,
                                   "rank": rank}))
            for e in by_wave[cur]:
                e.deps = tuple(e.deps) + (name,)

    if "stage" in kinds:
        _check_stage_feasible(events, by_wave, waves, ub_depth, aic_num, pooled)
        for w in waves:
            present = [st for st in STAGE_ORDER
                       if any(str(e.meta.get("stage")) == st for e in by_wave[w])]
            for prev, cur in zip(present, present[1:]):
                order += 1
                name = f"{pre}barrier.w{w}.{cur}"
                new.append(Event(
                    name=name, resources=(), duration_us=0.0,
                    deps=tuple(e.name for e in by_wave[w]
                               if str(e.meta.get("stage")) == prev),
                    order=order,
                    meta={"stage": "barrier", "kind": "stage", "wave": w,
                          "after": prev, "rank": rank}))
                for e in by_wave[w]:
                    if str(e.meta.get("stage")) == cur:
                        e.deps = tuple(e.deps) + (name,)

    return events + new


def _check_stage_feasible(events, by_wave, waves, ub_depth: int, aic_num: int,
                          pooled: bool) -> None:
    """stage 栅栏与 UB 槽是否相容 —— 不相容时说清原因, 而不是报 capacity deadlock.

    范围是**一个波内**: stage 栅栏要求该波全部 GMM1 先于该波任何 ACT 完成。GMM1 的
    结果走 L0C->UB 的 Fixpipe 直给配对 AIV0, UB 每核只有 ub_depth 块, 而槽由配对的
    ACT 归还。于是一个核在同一波里拿到第 ub_depth+1 个 GMM1 tile 时:

        第 ub_depth+1 个 GMM1 等某个 ACT 还槽
        -> 那个 ACT 等 barrier.w{w}.activation
        -> 那道栅栏等该波**全部** GMM1, 包括第 ub_depth+1 个

    成环。所以判据是"某个核在同一波里的 GMM1 tile 数 > ub_depth":

      * 静态绑定: 按建图时的核号数 (max over (波, 核))
      * 晚绑定:   调度器可以在池内摊平, 所以下界是 ceil(该波 tile 数 / 核数)

    ub_depth = 0 (不要 UB 约束) 时永远相容 —— 分段式执行里 GMM1 本来就该走 L0C->GM
    (kernel 有这条路: Gmm1AicMmadTileToGmGeneric vs ...ToUbGeneric), UB 不是交接缓冲。
    """
    if ub_depth <= 0:
        return
    held = next((tok for e in events if str(e.meta.get("stage")) == "gmm1"
                 for tok, _k in e.acquires if "UB:gmm1act" in tok), None)
    if held is None:
        return
    worst, where = 0, None
    for w in waves:
        g = [e for e in by_wave[w] if str(e.meta.get("stage")) == "gmm1"]
        if not g:
            continue
        if pooled and aic_num > 0:
            need = -(-len(g) // aic_num)          # ceil: 摊平后每核至少这么多
        else:
            per: Dict[int, int] = {}
            for e in g:
                c = e.meta.get("core")
                per[c] = per.get(c, 0) + 1
            need = max(per.values())
        if need > worst:
            worst, where = need, w
    if worst > ub_depth:
        raise ValueError(
            f'barriers 含 "stage" 与 gmm1->activation 的 depth={ub_depth} 不相容: '
            f"波 {where} 里有核要做 {worst} 个 GMM1 tile (> {ub_depth}), 而 stage 栅栏要求"
            f"该波全部 GMM1 先于任何 ACT 完成 —— 第 {ub_depth + 1} 个 GMM1 等 ACT 还 UB 槽 "
            f"({held}), 那个 ACT 又等该波全部 GMM1, 成环。\n"
            "三条出路: (1) 把那条边的 depth 设 0 —— 分段式执行里 GMM1 走 L0C->GM, "
            f"UB 不是交接缓冲; (2) 把波宽调小到每核最多 {ub_depth} 个 tile "
            "(ModelOptions.m_groups_per_wave); (3) 把 UB 深度加到 >= 该值。")
