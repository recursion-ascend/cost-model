"""第 6 层: 运行时图重构策略 — 空闲核任务转移."""
from __future__ import annotations

from collections import defaultdict

from ..scheduler.events import Event, RestructureAction


def idle_core_stealing(stage: str = "gmm1", resource_prefix: str = "AIC:",
                       min_pending: int = 2, queue_token: str = "Q:aic"):
    """空闲核任务转移: 当本实例为静态 cursor 分配时, 量化
    "若改成动态负载均衡" 的收益. 触发: 某核该 stage 的未提交 tile 数
    ≥ min_pending 且存在空闲核 → 取消其尾部一个 tile 事件, 同名注入到空闲核.

    同名注入保证消费者 (如 ACT 的 dep) 语义不变; 资源与队列计数信号量换到新核.
    """
    def hook(ctx) -> RestructureAction:
        act = RestructureAction()

        def local_name(r: str) -> str:
            # 剥离 rank 前缀: simulate_multi 会把资源改名为 "R0.AIC:5"
            return r.split(".", 1)[1] if "." in r else r

        # 核集合 = 已见资源 ∪ 未提交事件资源 (空闲核没有 pending 事件, 必须补齐)
        cores = {r for r in ctx.resource_free
                 if local_name(r).startswith(resource_prefix)}
        for e in ctx.pending.values():
            for r in e.resources:
                if local_name(r).startswith(resource_prefix):
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
            return act  # 该 stage 无未提交 tile
        idlest = min(free, key=lambda r: (free[r], r))
        pend = by_res.get(idlest, [])
        if len(pend) >= min_pending:
            return act  # 空闲核自己还有未提交任务, 不转移
        busiest = max(by_res, key=lambda r: (len(by_res[r]), r))
        if busiest == idlest or len(by_res[busiest]) < min_pending:
            return act
        # 确定性: 转移构建序最大的 tile (尾部), 不受 pending 集合遍历序影响
        victim = max(by_res[busiest], key=lambda e: (e.order, e.name))
        new_core = idlest
        old_cid = victim.resources[0].split(":", 1)[1]   # 核标识 (资源名冒号后缀)
        new_cid = new_core.split(":", 1)[1]

        def remap_token(tok: str) -> str:
            # 队列计数信号量尾缀换核: "Q:aic:c0"/"Q:aic:0" → 对应新核后缀
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