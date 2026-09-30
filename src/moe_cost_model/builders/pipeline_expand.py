"""L0/L1/L2 流水线约束施加器.

输入 build_events 产出的事件表, 输出 (事件表, 容量表, 信道表) 供调度器.

  L0 同步延迟: 在 stage 间握手边上挂 dep_latency_overrides
  L1 队列计数信号量: stage 事件挂 QUEUE:* 信号量; gmm1→act 深度依赖走距离依赖
     (跨 tile 流水需要时拆 load/cube/fix 相位, 核资源移到 cube 相位)
  L2 信道需求: 有 GM 流量的相位声明 channel_bytes = 流量时长×应得速率

GMM 口径 (与 costs.AnalyticalGmmCosts 一致). 占用与计时是两回事:
  L1 缓冲 (MTE 队列) 是容量约束. A 与 B 都经 L1 进 L0, 所以 GMM1 与 GMM2 的
        tile 都占一个 L1 缓冲槽 — B 流不计时, 但权重仍在 L1 里占着位置.
  gm_to_l1 信道字节按各 stage 的载入相位时长折算 (GMM1 = A流+B流, GMM2 = B流).
  GMM1  载入走 GM→L1: 占 L1 缓冲槽 + gm_to_l1 信道; 计算占 Cube 队列.
        load / cube 相位时长取事件 meta 的 load_us / compute_us (公式分解).
  GMM2  B 权重流走 GM→L1: 占 L1 缓冲槽 + gm_to_l1 信道 + Cube 队列
        (A 已在片上, 不计 GM 流量).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config.hardware import BW_L1_GM, BW_SCATTER, KernelConfig
from ..scheduler.events import Channel, Event
from ..config.pipeline import PipelineConstraints

CH_GM_TO_L1 = "gm_to_l1"
CH_HBM_WRITE = "hbm_write"
# dispatch 的访存: 读源卡窗口 → UB, 写本卡 workspace。
# 单独两条而不是并到 gm_to_l1: default_channels 的契约是"每核速率 x 核数 = 刚好
# 不争用", 往 gm_to_l1 上再塞一个消费者就把中性基线破坏了, 多出来的时长是构造
# 出来的不是物理的 (见 test_channel_no_contention_invariance)。
# 要研究 dispatch 与 GMM1 抢访存 (即 T_GMM1_OVERLAP 那 0.9us 的机制), 得先有
# **整卡访存带宽**的实测: 把两者并到一条聚合为整卡值的信道上, 争用才是算出来的。
CH_DISPATCH_READ = "dispatch_read"
CH_DISPATCH_WRITE = "dispatch_write"

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

    aic_num/h: 形状参数 (保留以兼容调用方; 相位时长取自事件 meta)
    gmm1_act_depth: ModelOptions.gmm1_activation_depth, BUF 槽位默认值
    """
    km = kernel if kernel is not None else KernelConfig()
    if km.l1_buf_num == 1 and cons.queues.mte_aic > 1:
        raise ValueError(
            f"KernelConfig.l1_buf_num=1 (单 L1 缓冲) 与 queues.mte_aic="
            f"{cons.queues.mte_aic} (多个 L1 缓冲槽) 矛盾: 二者描述同一个硬件资源")
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
    # 落到 load 相位. 不用可互换信号量: 信号量配对在乱序调度下会与 act 的程序序
    # 依赖成环 (load_A 等 token, act_B 程序序等 act_A, fix_A, load_A).
    # 距离依赖天然保序无环, 深度由 ModelOptions.gmm1_activation_depth 参数化.

    # ---- L1/L2: 队列计数信号量 + 信道需求 (gmm1 需要时拆相位) ----
    # 深度 1 = 闭式时长整体标注; 深度 >1 才有跨 tile 的 load/cube 重叠可建模
    split_gmm1 = cons.queues.mte_aic > 1
    new_events: List[Event] = []
    for ev in events:
        stage = str(ev.meta.get("stage", ""))
        if stage == _STAGE_GMM1:
            new_events.extend(_expand_gmm1(ev, cons, split_gmm1, channels, by_name, h, km))
        elif stage == _STAGE_GMM2:
            new_events.extend(_annotate(ev, channels,
                                        queues=("QUEUE:mte_aic", "QUEUE:cube")))
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
        elif res.startswith("Q:aic:"):
            pass  # model.py 的引擎队列, 容量由 simulate() 设置
        elif res.startswith("Q:vec0:"):
            pass
        elif res.startswith("Q:aiv1:"):
            pass
        elif res.startswith("BUF:gmm1act:"):
            pass  # 已移除信号量机制, 距离依赖替代
        else:
            pass  # 未识别的引擎队列 — 容量由调用方 (simulate) 设置
    return new_events, capacities, channels


def _annotate(
    ev: Event,
    channels: Dict[str, Channel],
    *,
    queues: Tuple[str, ...],
) -> List[Event]:
    """GMM2 head/tail: 保留原结构, 挂队列计数信号量 (L1 缓冲槽 + Cube) 与 B 流信道.

    head/tail 是同一个 tile 的两段, 各按自己的时长占比分摊该 tile 的 B 流字节。
    """
    core = ev.meta.get("core")
    qs = tuple((f"{queue}:c{core}", 1) for queue in queues)
    ch = ()
    load_us = ev.meta.get("load_us")      # 该事件自己的载入份额, 不是整段时长
    if CH_GM_TO_L1 in channels and load_us:
        ch = ((CH_GM_TO_L1, load_us * BW_L1_GM, BW_L1_GM),)
    return [Event(
        name=ev.name, resources=ev.resources, duration_us=ev.duration_us,
        deps=ev.deps, order=ev.order, meta=dict(ev.meta),
        dep_latency_us=ev.dep_latency_us,
        dep_latency_overrides=ev.dep_latency_overrides,
        acquires=ev.acquires + qs, releases=ev.releases + qs,
        channel_bytes=ch,
    )]


def _drop_program_order(ev: Event, stage: str, by_name: Dict[str, Event]) -> Tuple[str, ...]:
    """剔除同 stage 同 core 的程序序依赖 (交给队列计数信号量/核资源)."""
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
    """GMM1 tile: 默认整体标注; MTE 队列深度 >1 时拆相位.

    拆分结构 (闭式 max(A流, 计算) 的含义是 tile 内载入与计算重叠, 慢侧绑定):

        grant ─┬─ load (A 流, gm_to_l1 信道, 无核资源) ─┬─ fix (写回, 保留原名)
               └─ cube (计算, 核资源 + Cube 队列)      ─┘

      grant  取一个 L1 缓冲槽 (MTE 队列), fix 结束时归还. 时长 = 闭式时长里
             公式以外的部分 (流水填充、每核首 tile 启动开销).
      load 与 cube 并行, fix 等两者都完成 → 单 tile 时长 = max(load, cube),
             与闭式一致; A 流被信道切速时 load 拉长, tile 随之拉长.
      跨 tile: cube 独占核, 后一个 tile 的 load 可在前一个 tile 计算时预取,
             预取个数受 MTE 队列深度限制.

    load / cube 时长取 meta 的 load_us / compute_us (GMM 公式的分解).
    stage 忙碌时长只计 load 与 cube 中较长的一个 (较短者标 overlapped),
    否则重叠的时间会被算两遍.
    """
    km = km if km is not None else KernelConfig()
    stage = _STAGE_GMM1
    core = ev.meta.get("core")
    m_rows = int(ev.meta.get("m_rows", 0))
    base_dur = ev.duration_us
    load_us = ev.meta.get("load_us")
    compute_us = ev.meta.get("compute_us")
    if load_us is None and (split or CH_GM_TO_L1 in channels):
        raise ValueError(
            f"{ev.name}: 缺 A 流/计算分解 — gmm1_tile 是自定义 callable. "
            "相位拆分 (queues.mte_aic > 1) 与 gm_to_l1 信道需要 AnalyticalGmmCosts 的公式")
    # 字节 = 载入相位时长 × 应得速率 (无争用服务时长 = load_us), 信道因此中性。
    #
    # 已声明未建模: 2026-09-30 起 GMM1 的载入是 max(A流, B流) 而不是相加 (两点 m 扫
    # 实测: tile 时长与 m 无关), 于是这里折算出的字节只有 max(A,B) = B 流那一份,
    # **少算了 A 流真实搬运的 m*k 字节**。
    # 这不是笔误, 是当前单信道抽象表达不了的东西: A 流的字节是真的要过去, 但它的
    # 延迟被跨 tile 的 L1 双缓冲藏住了 (tile i+1 的载入与 tile i 的计算重叠), 所以
    # 它占带宽却不占本 tile 载入相位的时长。若在这里改成申报 A+B 字节, 调度器会按
    # max(名义, 字节/速率) 把事件拉长回相加口径 —— 反而与实测矛盾。
    # 要同时表达"占带宽、不占本 tile 时长", 需要把预取建成跨 tile 的独立事件
    # (载入 tile i+1 的 A 流挂在 tile i 的计算旁), 那是编排层的改动。
    ch = ((CH_GM_TO_L1, load_us * BW_L1_GM, BW_L1_GM),) \
        if CH_GM_TO_L1 in channels and load_us else ()

    if not split:
        # 整体标注: 事件时长 = 闭式时长; A 流被信道切速到超过它时随之拉长
        q = (f"QUEUE:mte_aic:c{core}", 1)
        return [Event(
            name=ev.name, resources=ev.resources, duration_us=base_dur,
            deps=ev.deps, order=ev.order, meta=dict(ev.meta),
            dep_latency_us=ev.dep_latency_us,
            dep_latency_overrides=ev.dep_latency_overrides,
            acquires=ev.acquires + (q,), releases=ev.releases + (q,),
            channel_bytes=ch,
        )]

    # 尾 N-tile 精确: 写回按实际列数, 不按整 tileN (缺省 meta 时退回整 tile)
    logical_n = int(ev.meta.get("logical_n", km.tile_n))
    fix_bw = cons.phases.fix_bw_bytes_per_us
    fix_dur = (m_rows * logical_n * 2 / fix_bw) if fix_bw else 0.0
    closed_form = max(load_us, compute_us)      # 走到这里必为双缓冲 (入口已校验)
    overhead = max(0.0, base_dur - closed_form)
    load_longer = load_us > compute_us

    # 缓冲占用语义: mte 计数信号量 = L1 缓冲槽.
    # grant 开始时取, fix 结束时还 (载入与计算都完成, 缓冲才空出来) —
    # QueueDepths(mte_aic=d) 因此精确等于 d 个 L1 缓冲. 相位事件不继承引擎
    # 信号量 (Q:aic), 否则容量 1 的引擎信号量会卡死 mte 深度 (审计已确认的旧 bug).
    mte = (f"QUEUE:mte_aic:c{core}", 1)
    lg = Event(
        name=ev.name + ".lg", resources=(), duration_us=overhead,
        deps=_drop_program_order(ev, stage, by_name), order=ev.order,
        meta=dict(ev.meta, phase="grant"),
        dep_latency_us=ev.dep_latency_us,
        dep_latency_overrides=ev.dep_latency_overrides,
        acquires=(mte,),
    )
    ld = Event(
        name=ev.name + ".ld", resources=(), duration_us=load_us,
        deps=(lg.name,), order=ev.order,
        meta=dict(ev.meta, phase="load", overlapped=not load_longer),
        channel_bytes=ch,
    )
    cb = Event(
        name=ev.name + ".cb", resources=ev.resources, duration_us=compute_us,
        deps=(lg.name,), order=ev.order,
        meta=dict(ev.meta, phase="cube", overlapped=load_longer),
        acquires=((f"QUEUE:cube:c{core}", 1),),
        releases=((f"QUEUE:cube:c{core}", 1),),
    )
    fx = Event(
        name=ev.name, resources=(), duration_us=fix_dur,
        deps=(ld.name, cb.name), order=ev.order, meta=dict(ev.meta, phase="fix"),
        acquires=((f"QUEUE:fix:c{core}", 1),),
        releases=((f"QUEUE:fix:c{core}", 1), mte),
    )
    return [lg, ld, cb, fx]


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
    # 队列计数信号量按引擎分名: AIV0(ACT) 与 AIV1(COMBINE) 是两个引擎, 不共享.
    # 共享旧名 QUEUE:vec:c{core} 会把同核 ACT/COMBINE 错误串行.
    eng = "aiv0" if vec else "aiv1"
    vec_queue = (f"QUEUE:vec:{eng}:c{core}", 1)
    # 保留建图器自己申报的字节 (ACT 的 GM 写出、COMBINE 的片间写), 只在这里**追加**
    # COMBINE 的散射写。原先这里是直接覆盖 —— ACT 的写信道字节与 COMBINE 的
    # fab_* 字节都会在启用相位流水时被悄悄丢掉。
    ch = ev.channel_bytes
    if stage == _STAGE_COMBINE and CH_HBM_WRITE in channels and base_dur > 0:
        ch = ch + ((CH_HBM_WRITE, base_dur * BW_SCATTER, BW_SCATTER),)

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
    # 尾 N-tile 精确: GM 读字节按实际列数 (缺省 meta 时退回整 tile)
    logical_n = int(ev.meta.get("logical_n", km.tile_n))
    load_bytes = m_rows * logical_n * 2   # BF16 GM 读
    load_dur = load_bytes / load_bw
    ld = Event(
        name=ev.name + ".ld", resources=(), duration_us=load_dur,
        deps=_drop_program_order(ev, stage, by_name), order=ev.order,
        meta=dict(ev.meta, phase="load"),
        dep_latency_us=ev.dep_latency_us,
        dep_latency_overrides=ev.dep_latency_overrides,
        acquires=ev.acquires + ((f"QUEUE:mte_aiv:{eng}:c{core}", 1),),
        releases=((f"QUEUE:mte_aiv:{eng}:c{core}", 1),),
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
