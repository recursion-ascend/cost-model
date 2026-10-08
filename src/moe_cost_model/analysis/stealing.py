"""第 6 层: 运行时图重构策略 — 空闲核任务转移.

kernel 的静态分核 (startBlockIdx 旋转 + HandleWaveProblemWithoutWork) 只在
**跨专家**维度填充: 某核对本专家无 tile 就跳到下一个专家, 不空等。但它不跨核填充 ——
tile 一旦发给某核, 别的核不能拿。于是 n-tile 数不整除核数时 (如 hidden_dim=9216 下
每专家 18 个 tile 发给 28 核), 拿 1 个 tile 的核做完就闲着, 拿 2 个的还在跑。

本模块量化"若允许跨核搬运"的收益: 每次事件提交后, 把已就绪的 tile 从积压的核搬到
此刻真正空闲的核。搬运用**同名注入** (RestructureAction 契约), 所以依赖边按名字解析,
一条都不动 —— 改的只是事件占用哪个资源。

判定逻辑 (事件状态机):
    waiting   前置事件未完成              -> 不可派发, 搬过去也是干等
    ready     前置已完成, 只差资源        -> 可派发
    running   已派发, 占着资源
    done      已完成 (ctx.end_by_name 里有它)
  搬运条件三者同时成立:
    (a) 目标核此刻真空闲:  resource_free[dst] <= ctx.time_us
    (b) 事件处于 ready:    ctx.is_ready(ev)  (全部前置已完成)
    (c) 事件自己的核此刻忙: resource_free[src] > ctx.time_us
        —— 否则调度器自己就会派发它, 不需要搬。
  注意 (a) **不能**写成"目标核该 stage 无未提交 tile": 核上压着的 tile 可能还在
  waiting, 此刻核其实是空的, 正该去接别处已就绪的活。源核也不设"积压 >= N"的门槛 ——
  判据就是 (b)+(c): 活已就绪而它的核忙着, 同时别处有核空着。

三条物理约束 (都曾被违反过, 记下来免得再犯):
  1. 不跨 rank。simulate_multi 把资源改名为 "R{rank}.AIC:{c}", 核池必须按 rank 前缀
     分组。原实现用 local_name() 剥掉前缀, 把 28xN 个核当成一个池, 实测把 rank4 的
     GMM1 tile 搬到了 rank0 的 AIC 上 —— 跨卡搬 tile 物理不可能。
  2a. 搬运单位是**整个 tile 组**, 不是单个事件。一个 GMM1 tile 对应 {gmm1, act};
     一个 GMM2 tile 对应 {gmm2.h (头), gmm2 (尾), combine}。组内成员按 tile 身份
     (rank/wave/expert/slice/mgroup/ntile/row_begin/col_begin) 聚合, 一起换核。
  2. 不破坏 AIC/AIV0 配对。GMM1 结果经 L0C->UB 硬件通路直给**同核**的 AIV0
     (见 builders/activation.py: "ACT 固定在配对 GMM1 同核的 AIV0 上"), 所以 GMM1 搬核
     必须带着配对的 ACT 一起搬。原实现只搬 GMM1 单个事件, 把 ACT 留在原核。
  3. 只搬已就绪的事件。原实现搬 max(order) 即构建序尾部的 tile, 那个 tile 的前置
     大概率还没完成, 搬到新核上照样干等, 等于没搬 (实测 rank0 上 2515 次调用一次
     没有有效触发)。就绪判定用 RestructureContext.is_ready。
"""
from __future__ import annotations

from collections import defaultdict
import dataclasses
from typing import Dict, List, Optional, Tuple

from ..config.stages import default_vocabulary
from ..scheduler.events import Event, RestructureAction

#: 资源名里 rank 前缀与本地名的分隔符 ("R0.AIC:5" -> ("R0", "AIC:5"))
_RANK_SEP = "."


def _split_rank(resource: str) -> Tuple[str, str]:
    """("R0.AIC:5") -> ("R0", "AIC:5"); 无前缀时 rank 为空串 (单 rank 仿真)."""
    if _RANK_SEP in resource:
        head, tail = resource.split(_RANK_SEP, 1)
        return head, tail
    return "", resource


def _core_id(local: str) -> str:
    """("AIC:5") -> "5"."""
    return local.split(":", 1)[1] if ":" in local else local


def _retarget(resource: str, new_core: str) -> str:
    """把资源名的核号换成 new_core, 保留 rank 前缀与角色 ("R0.AIV0:3" -> "R0.AIV0:7")."""
    rank, local = _split_rank(resource)
    role = local.split(":", 1)[0] if ":" in local else local
    moved = f"{role}:{new_core}"
    return f"{rank}{_RANK_SEP}{moved}" if rank else moved


def _remap_token(token: str, old_core: str, new_core: str) -> str:
    """队列计数信号量尾缀换核: "Q:aic:c0" -> "Q:aic:c7" (只换结尾的核号)."""
    if token.endswith(old_core):
        return token[: -len(old_core)] + new_core
    return token


def _moved(ev: Event, new_core: str, old_core: str, meta_extra: Dict) -> Event:
    """同名、同依赖、同时长, 只把资源与队列 token 换到新核.

    **必须逐字段搬全**。2026-10-06 之前这里漏了三个字段 —— colocate_with / core_group /
    once_per_core —— 于是被转移的事件静默丢掉约束:

      colocate_with  共位是硬件通路 (GMM1 的 L0C -> 配对 AIV0 的 UB, Fixpipe 只在绑定对内)。
                     丢了它, 调度器就不再强制 ACT 与它的 GMM1 同核。本模块是成组搬的
                     (gmm1 连同它的 activation 一起), 所以**今天**两者仍然同核 —— 但那是
                     搬运逻辑恰好保证的, 不是约束还在。改一下分组或加一个 stage, 它就不成立,
                     而且不会报错。
      core_group     不持核资源的相位事件 (lg/ld/fix) 靠它拿候选核 (engine 的 _group_members)。
                     丢了它, 这类事件在新核上没有候选 -> _pick_core 返回 None -> 排不上。
      once_per_core  每核一次的开销 (dispatch 的调用开销走这条)。丢了它, 开销凭空消失。

    用 dataclasses.replace 而不是逐字段重建: 以后 Event 加字段, 这里自动带上, 不会再漏。
    只有真要改的四项显式覆盖。
    """
    return dataclasses.replace(
        ev,
        resources=tuple(_retarget(r, new_core) for r in ev.resources),
        meta=dict(ev.meta, **meta_extra),
        acquires=tuple((_remap_token(q, old_core, new_core), k) for q, k in ev.acquires),
        releases=tuple((_remap_token(q, old_core, new_core), k) for q, k in ev.releases),
    )


#: 搬运时必须随动的配对 (驱动 stage, 随动 stage) 由**实现声明的词汇表**给
#: (config/stages.py): gmm1 -> act 是 AIC/AIV0 同核 (L0C->UB 硬件通路),
#: gmm2 -> combine 是 AIC/AIV1 同核 (gmmToEpilogueFlag 按核索引) —— 这是同核关系,
#: 不是策略。缺省开哪几组、为什么只开一部分, 也写在那份声明里。
DEFAULT_STEAL_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    default_vocabulary().default_steal_groups)

#: 全部搬运组都开, 供显式实验用 (声明处写了两组同开的 thrashing 实测)。
ALL_STEAL_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    default_vocabulary().steal_groups)

#: 旧名字, 仓内调用沿用 (= ALL_STEAL_GROUPS)
GMM1_AND_GMM2_GROUPS = ALL_STEAL_GROUPS


def idle_core_stealing(stage: Optional[str] = None, resource_prefix: str = "AIC:",
                       queue_token: str = "Q:aic",
                       paired_stages: Optional[Tuple[str, ...]] = None,
                       max_moves_per_commit: int = 0,
                       min_pending: int = 1,
                       groups: Optional[Tuple[Tuple[str, Tuple[str, ...]], ...]] = None,
                       balance: bool = True):
    """空闲核任务转移 (work-conserving: 就绪的活 + 空闲的核 = 立刻开工).

    groups:           ((驱动 stage, 配对 stage 元组), ...); 缺省 DEFAULT_STEAL_GROUPS
    stage/paired_stages: 旧式单组入参, 给出 stage 时覆盖 groups (向后兼容)
    resource_prefix:  核池的资源前缀 (默认 AIC:)
    max_moves_per_commit: 单次钩子调用最多搬几组; 0 = 不限 (搬到没有空闲核为止)
    min_pending:      源核至少有几个未提交 tile 才允许搬 (默认 1 = 不设门槛)
    balance:          严格 work-conserving 无机会时是否做主动配平 (默认 True);
                      False = 只做严格判据, 可用来量化两者的差别
    """
    if min_pending < 1:
        raise ValueError("min_pending must be >= 1")
    if stage is not None:
        groups = ((stage, tuple(paired_stages) if paired_stages is not None
                   else ("activation",)),)
    elif groups is None:
        groups = DEFAULT_STEAL_GROUPS

    def hook(ctx) -> RestructureAction:
        act = RestructureAction()

        # ---- 核池: 按 rank 分组, 绝不跨 rank ----
        pools: Dict[str, set] = defaultdict(set)
        for r in ctx.resource_free:
            rank, local = _split_rank(r)
            if local.startswith(resource_prefix):
                pools[rank].add(r)
        for e in ctx.pending.values():
            for r in e.resources:
                rank, local = _split_rank(r)
                if local.startswith(resource_prefix):
                    pools[rank].add(r)
        if not pools:
            return act

        def tile_key(e: Event):
            """tile 身份. 不含 part/stage —— 一个 GMM2 tile 的头尾两个事件和它的
            combine 共享同一个身份, 必须整组搬。rank 必须在里面: simulate_multi 用
            **一个**调度器跑全部 rank, 不带 rank 的话 R0 与 R4 的同坐标 tile 在索引里
            互相覆盖 (实测: 一次调用搬 40 个 gmm1 只带了 8 个 ACT)。"""
            m = e.meta
            k = (m.get("rank"), m.get("wave"), m.get("expert"), m.get("slice"),
                 m.get("mgroup"), m.get("ntile"),
                 m.get("row_begin"), m.get("col_begin"))
            if not any(x is not None for x in k):
                # meta 不带 tile 坐标 (手写事件、微基准) —— 退化成"每个事件自成一组",
                # 否则全 None 的 key 会把所有事件当成同一个 tile, 一次调用把它们全搬到
                # 同一个核上 (实测微基准里 5 个事件被一起搬走)。
                return ("__self__", e.name)
            return k

        taken: set = set()           # 本次调用已占用的目标核
        moved_names: set = set()     # 本次调用已搬过的事件名
        moves = 0

        for drive_stage, pairs in groups:
            member_stages = (drive_stage,) + tuple(pairs)
            # 驱动 stage 的未提交事件按核归集
            by_res: Dict[str, List[Event]] = defaultdict(list)
            # tile 身份 -> 同组全部成员 (驱动 + 配对)
            group_of: Dict[tuple, List[Event]] = defaultdict(list)
            for e in ctx.pending.values():
                st = str(e.meta.get("stage"))
                if st not in member_stages or not e.resources:
                    continue
                group_of[tile_key(e)].append(e)
                if st == drive_stage:
                    by_res[e.resources[0]].append(e)
            if not by_res:
                continue

            for rank, pool in sorted(pools.items()):
                while True:
                    if max_moves_per_commit and moves >= max_moves_per_commit:
                        return act
                    # (a) 目标核: 此刻真空闲。不看它压着多少 tile —— 那些 tile 可能
                    #     还在 waiting, 核此刻是空的。
                    idle = sorted(
                        r for r in pool
                        if r not in taken
                        and ctx.resource_free.get(r, 0.0) <= ctx.time_us
                    )
                    if not idle:
                        break
                    # (b)+(c) 可搬: 已就绪 (前置全完成) 且它自己的核此刻忙。
                    #     自己的核空着的话调度器会直接派发, 不需要搬。
                    movable: List[Tuple[Event, str]] = []
                    for src_res, evs in by_res.items():
                        if _split_rank(src_res)[0] != rank:
                            continue
                        if ctx.resource_free.get(src_res, 0.0) <= ctx.time_us:
                            continue
                        if len(evs) < min_pending:
                            continue
                        for e in evs:
                            if e.name in moved_names:
                                continue
                            if ctx.is_ready(e):
                                movable.append((e, src_res))
                    if not movable and balance:
                        # 退化判据: 严格 work-conserving 无机会时做**主动配平**。
                        # 两个核同时空闲时 (a)+(c) 都不触发 —— 调度器会各派一个 —— 但
                        # 这正是提前搬走尾部 tile、让各核事件数对等的时机。不做这一步,
                        # 未来会出现"一核还剩 2 个、另一核已空"的不可挽回的倾斜。
                        # 实测微基准 (5 个 10us 事件在 c0 + 1 个在 c1):
                        #   静态 50us / 只做严格 work-conserving 40us / 加主动配平 30us。
                        for src_res, evs in by_res.items():
                            if _split_rank(src_res)[0] != rank:
                                continue
                            # 配平的门槛是它自己的: 源核至少有 2 个未提交 tile (分出去
                            # 一个还留一个), 且目标核比源核少。不复用 min_pending ——
                            # 那是严格判据的"懒得为小积压动手"阈值, 两件事。
                            if len(evs) < 2 or len(by_res.get(idle[0], ())) >= len(evs) - 1:
                                continue
                            for e in evs:
                                if e.name not in moved_names and ctx.is_ready(e):
                                    movable.append((e, src_res))
                    if not movable:
                        break
                    # 从积压最多的核搬; 同核内取构建序最大的 (尾部) —— 队首继续按序跑,
                    # 搬走最不急的那个。name 兜底保证与哈希种子无关的确定性。
                    movable.sort(key=lambda t: (-len(by_res[t[1]]), -t[0].order, t[0].name))
                    victim, src = movable[0]

                    dst = idle[0]
                    new_core = _core_id(_split_rank(dst)[1])
                    group = [e for e in group_of[tile_key(victim)]
                             if e.name not in moved_names]
                    if not group:
                        break
                    for e in group:
                        e_old = _core_id(_split_rank(e.resources[0])[1])
                        act.cancel.append(e.name)
                        act.inject.append(_moved(
                            e, new_core, e_old,
                            {"stolen_from": e.resources[0],
                             "core": int(new_core) if new_core.isdigit() else new_core}))
                        moved_names.add(e.name)
                    by_res[src] = [e for e in by_res[src] if e.name not in moved_names]
                    taken.add(dst)
                    moves += 1

        return act

    return hook
