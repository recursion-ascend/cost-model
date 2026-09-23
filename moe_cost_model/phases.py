"""L0/L1/L2 流水线约束施加器.

输入 build_events 产出的事件表, 输出 (事件表, 容量表, 信道表) 供调度器.

  L0 同步延迟: 在 stage 间握手边上挂 dep_latency_overrides
  L1 队列令牌: stage 事件挂 QUEUE:* 令牌; gmm1→act 深度依赖转为 BUF 令牌
     (跨 tile 流水需要时拆 load/cube/fix 相位, 核资源移到 cube 相位)
  L2 信道需求: 承载闭式时长的相位声明 channel_bytes = 时长×应得速率
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .constants import BW_L1_GM, BW_SCATTER, TILE_N, KernelConfig
from .dag import Channel, Event
from .pipeline import PipelineConstraints

CH_GM_TO_L1 = "gm_to_l1"
CH_HBM_WRITE = "hbm_write"

_STAGE_GMM1 = "gmm1"
_STAGE_ACT = "activation"
_STAGE_GMM2 = "gmm2"
_STAGE_COMBINE = "combine"


def apply_pipeline(
    events: List[Event],
    cons: PipelineConstraints,
    *,
    aic_num: int,
    h: int,
    gmm1_act_depth: int = 1,
    kernel=None,
) -> Tuple[List[Event], Dict[str, int], Dict[str, Channel]]:
    """施加约束, 返回 (事件表, 容量表, 信道表).

    aic_num/h: 形状参数 (队列资源命名与 MAC 计算用)
    gmm1_act_depth: ModelOptions.gmm1_activation_depth, BUF 槽位默认值
    """
    km = kernel if kernel is not None else KernelConfig()
    by_name = {ev.name: ev for ev in events}
    channels = {ch.name: ch for ch in cons.channels}

    # ---- L0: 同步延迟挂到握手边 ----
    for ev in events:
        stage = str(ev.meta.get("stage", ""))
        if stage == _STAGE_ACT and cons.sync.gmm1_act_handshake_us:
            ev.dep_latency_overrides = ev.dep_latency_overrides + tuple(
                (d, cons.sync.gmm1_act_handshake_us)
                for d in ev.deps
                if str(by_name[d].meta.get("stage", "")) == _STAGE_GMM1
            )
        elif stage == _STAGE_GMM2 and cons.sync.act_gmm2_ready_us:
            ev.dep_latency_overrides = ev.dep_latency_overrides + tuple(
                (d, cons.sync.act_gmm2_ready_us)
                for d in ev.deps
                if str(by_name[d].meta.get("stage", "")) == _STAGE_ACT
            )
        elif stage == _STAGE_COMBINE and cons.sync.gmm2_combine_ack_us:
            ev.dep_latency_overrides = ev.dep_latency_overrides + tuple(
                (d, cons.sync.gmm2_combine_ack_us)
                for d in ev.deps
                if str(by_name[d].meta.get("stage", "")) == _STAGE_GMM2
            )

    # ---- L1: gmm1→act 生产者/消费者距离 ----
    # 保留 build_events 的距离依赖 (gmm1[i] deps act[i-depth]), 拆相位时自动
    # 落到 load 相位. 不用可互换令牌: 令牌配对在乱序调度下会与 act 的程序序
    # 依赖成环 (load_A 等 token ← act_B 程序序等 act_A ← fix_A ← load_A).
    # 距离依赖天然保序无环, 深度由 ModelOptions.gmm1_activation_depth 参数化.

    # ---- L1/L2: 队列令牌 + 信道需求 (gmm1 需要时拆相位) ----
    split_gmm1 = cons.queues.mte_aic > 1 or cons.phases.cube_mac_per_us is not None
    new_events: List[Event] = []
    for ev in events:
        stage = str(ev.meta.get("stage", ""))
        if stage == _STAGE_GMM1:
            new_events.extend(_expand_gmm1(ev, cons, split_gmm1, channels, by_name, h, km))
        elif stage == _STAGE_GMM2:
            new_events.extend(_annotate(ev, channels, queue="QUEUE:mte_aic"))
        elif stage == _STAGE_ACT:
            new_events.extend(_expand_aiv(
                ev, cons, channels, by_name, vec=True, km=km,
                load_bw=cons.phases.act_load_bw_bytes_per_us))
        elif stage == _STAGE_COMBINE:
            new_events.extend(_expand_aiv(
                ev, cons, channels, by_name, vec=False, km=km,
                load_bw=cons.phases.combine_load_bw_bytes_per_us))
        else:
            new_events.append(ev)

    # ---- 容量表 (只声明实际被引用的资源) ----
    capacities: Dict[str, int] = {}
    used = set()
    for ev in new_events:
        for res, _ in ev.acquires:
            used.add(res)
        for res, _ in ev.releases:
            used.add(res)
    for res in used:
        if res.startswith("QUEUE:mte_aic:"):
            capacities[res] = cons.queues.mte_aic
        elif res.startswith("QUEUE:cube:"):
            capacities[res] = cons.queues.cube
        elif res.startswith("QUEUE:fix:"):
            capacities[res] = cons.queues.fix
        elif res.startswith("QUEUE:vec:"):
            capacities[res] = cons.queues.vec
        elif res.startswith("QUEUE:mte_aiv:"):
            capacities[res] = cons.queues.mte_aiv
        else:
            raise ValueError(f"unknown capacity resource {res}")
    return new_events, capacities, channels


def _annotate(
    ev: Event,
    channels: Dict[str, Channel],
    *,
    queue: str,
) -> List[Event]:
    """GMM2 head/tail: 保留原结构, 挂 MTE 队列令牌 + 信道需求."""
    core = ev.meta.get("core")
    ch = ()
    if CH_GM_TO_L1 in channels and ev.duration_us > 0:
        ch = ((CH_GM_TO_L1, ev.duration_us * BW_L1_GM, BW_L1_GM),)
    q = (f"{queue}:c{core}", 1)
    return [Event(
        name=ev.name, resources=ev.resources, duration_us=ev.duration_us,
        deps=ev.deps, order=ev.order, meta=dict(ev.meta),
        dep_latency_us=ev.dep_latency_us,
        dep_latency_overrides=ev.dep_latency_overrides,
        acquires=ev.acquires + (q,), releases=ev.releases + (q,),
        channel_bytes=ch,
    )]


def _drop_program_order(ev: Event, stage: str, by_name: Dict[str, Event]) -> Tuple[str, ...]:
    """剔除同 stage 同 core 的程序序依赖 (交给队列令牌/核资源)."""
    return tuple(
        d for d in ev.deps
        if not (str(by_name[d].meta.get("stage", "")) == stage
                and by_name[d].meta.get("core") == ev.meta.get("core"))
    )


def _expand_gmm1(
    ev: Event,
    cons: PipelineConstraints,
    split: bool,
    channels: Dict[str, Channel],
    by_name: Dict[str, Event],
    h: int,
    km=None,
) -> List[Event]:
    """GMM1 tile: 默认整体标注; 深度>1 或给了 cube 速率时拆 load/cube/fix.

    拆分结构: load(流量, MTE 队列+信道, 无核资源) → cube(计算, 核资源+Cube
    队列) → fix(写回, Fix 队列, 保留原名承接下游依赖). 跨 tile 的 load 与
    cube 重叠由 MTE 队列深度控制 — 稳态周期 = max(load, cube) 与闭式
    max(载入/BW, 计算/R) 一致.
    """
    km = km if km is not None else KernelConfig()
    stage = _STAGE_GMM1
    core = ev.meta.get("core")
    m_rows = int(ev.meta.get("m_rows", 0))
    base_dur = ev.duration_us
    ch = ((CH_GM_TO_L1, base_dur * BW_L1_GM, BW_L1_GM),) if CH_GM_TO_L1 in channels else ()

    if not split:
        q = (f"QUEUE:mte_aic:c{core}", 1)
        return [Event(
            name=ev.name, resources=ev.resources, duration_us=base_dur,
            deps=ev.deps, order=ev.order, meta=dict(ev.meta),
            dep_latency_us=ev.dep_latency_us,
            dep_latency_overrides=ev.dep_latency_overrides,
            acquires=ev.acquires + (q,), releases=ev.releases + (q,),
            channel_bytes=ch,
        )]

    cube_rate = cons.phases.cube_mac_per_us
    macs = 2.0 * m_rows * km.tile_n * h   # SwiGLU 双投影
    cube_dur = (macs / cube_rate) if cube_rate else 0.0
    fix_bw = cons.phases.fix_bw_bytes_per_us
    fix_dur = (m_rows * km.tile_n * 2 / fix_bw) if fix_bw else 0.0
    load_dur = max(0.0, base_dur - cube_dur - fix_dur)

    ld = Event(
        name=ev.name + ".ld", resources=(), duration_us=load_dur,
        deps=_drop_program_order(ev, stage, by_name), order=ev.order,
        meta=dict(ev.meta, phase="load"),
        dep_latency_us=ev.dep_latency_us,
        dep_latency_overrides=ev.dep_latency_overrides,
        acquires=ev.acquires + ((f"QUEUE:mte_aic:c{core}", 1),),
        releases=((f"QUEUE:mte_aic:c{core}", 1),),
        channel_bytes=ch,
    )
    cb = Event(
        name=ev.name + ".cb", resources=ev.resources, duration_us=cube_dur,
        deps=(ld.name,), order=ev.order, meta=dict(ev.meta, phase="cube"),
        acquires=((f"QUEUE:cube:c{core}", 1),),
        releases=((f"QUEUE:cube:c{core}", 1),),
    )
    fx = Event(
        name=ev.name, resources=(), duration_us=fix_dur,
        deps=(cb.name,), order=ev.order, meta=dict(ev.meta, phase="fix"),
        acquires=((f"QUEUE:fix:c{core}", 1),),
        releases=((f"QUEUE:fix:c{core}", 1),),
    )
    return [ld, cb, fx]


def _expand_aiv(
    ev: Event,
    cons: PipelineConstraints,
    channels: Dict[str, Channel],
    by_name: Dict[str, Event],
    *,
    vec: bool,
    load_bw: Optional[float],
    km=None,
) -> List[Event]:
    """ACT/COMBINE tile (AIV).

    ACT:     vec 相位承载闭式时长 (UB 流量, 核内私有无信道)
    COMBINE: scatter 相位承载闭式时长 + hbm_write 信道
    给了 load 带宽则前置 GM 读相位 (MTE_AIV 队列), 否则整体标注.
    """
    km = km if km is not None else KernelConfig()
    stage = str(ev.meta.get("stage", ""))
    core = ev.meta.get("core")
    base_dur = ev.duration_us
    vec_queue = (f"QUEUE:vec:c{core}", 1)
    if stage == _STAGE_COMBINE:
        ch = ((CH_HBM_WRITE, base_dur * BW_SCATTER, BW_SCATTER),) \
            if CH_HBM_WRITE in channels and base_dur > 0 else ()
    else:
        ch = ()

    need_split = load_bw is not None and base_dur > 0
    if not need_split:
        return [Event(
            name=ev.name, resources=ev.resources, duration_us=base_dur,
            deps=ev.deps, order=ev.order, meta=dict(ev.meta),
            dep_latency_us=ev.dep_latency_us,
            dep_latency_overrides=ev.dep_latency_overrides,
            acquires=ev.acquires + (vec_queue,),
            releases=ev.releases + (vec_queue,),
            channel_bytes=ch,
        )]

    m_rows = int(ev.meta.get("m_rows", 0))
    load_bytes = m_rows * km.tile_n * 2   # BF16 GM 读
    load_dur = load_bytes / load_bw
    ld = Event(
        name=ev.name + ".ld", resources=(), duration_us=load_dur,
        deps=_drop_program_order(ev, stage, by_name), order=ev.order,
        meta=dict(ev.meta, phase="load"),
        dep_latency_us=ev.dep_latency_us,
        dep_latency_overrides=ev.dep_latency_overrides,
        acquires=ev.acquires + ((f"QUEUE:mte_aiv:c{core}", 1),),
        releases=((f"QUEUE:mte_aiv:c{core}", 1),),
    )
    main = Event(
        name=ev.name, resources=ev.resources, duration_us=base_dur,
        deps=(ld.name,), order=ev.order,
        meta=dict(ev.meta, phase="vec" if vec else "scatter"),
        releases=ev.releases + (vec_queue,),
        acquires=(vec_queue,),
        channel_bytes=ch,
    )
    return [ld, main]
