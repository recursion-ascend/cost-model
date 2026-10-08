"""第 4.5 层: 相位展开 — 把一个事件拆成 load / cube / fix 相位.

**字节申报的口径 (2026-10-05 立为不变量)**: 访存量由**建图器**按算法逐项算出并申报;
本模块只负责把它**重新分配**到各相位上, **绝不从时长倒推字节**。

为什么这是条硬规矩 —— 倒推 (时长 x 名义带宽) 在三种情况下直接给错数:
  1. load_overlap="max" 口径下 load_us = max(A流, B流), 倒推只拿到较大那一股;
  2. serial 口径下 load_us 还含 chunk_restart, 那不是字节;
  3. weight_nz 时 B 流用的是 bw_b 而不是 BW_L1_GM, 两个常数不同。
更要命的是: 倒推只发生在拆相位这条路径上, 于是**开不开相位流水会改变"搬了多少
字节"** —— 换一个编排参数不该改变算法必搬的量。
COMBINE 那条 (base_dur x BW_SCATTER) 已于 2026-10-05 删除, GMM1/GMM2 这两条同日改成
原样透传。tests/test_bounds.py::test_declared_bytes_do_not_depend_on_phase_pipelining
把这条不变量约束。
L0/L1/L2 流水线约束施加器.

输入 build_events 产出的事件表, 输出 (事件表, 容量表, 信道表) 供调度器.

  L0 同步延迟: 在 stage 间握手边上挂 dep_latency_overrides
  L1 队列计数信号量: stage 事件挂 QUEUE:* 信号量; gmm1→act 深度依赖走距离依赖
     (跨 tile 流水需要时拆 load/cube/fix 相位, 核资源移到 cube 相位)
  L2 信道需求: 有 GM 流量的相位声明 channel_bytes = 流量时长×应得速率

GMM 口径 (与 costs.AnalyticalGmmCosts 一致). 占用与计时是两回事:
  L1 缓冲 (MTE 队列) 是容量约束. A 与 B 都经 L1 进 L0, 所以 GMM1 与 GMM2 的
        tile 都占一个 L1 缓冲槽 — B 流不计时, 但权重仍在 L1 里占着位置.
  gm_to_l1 访存量按各 stage 的载入相位时长折算 —— 注意 max(A流,B流) 口径下这只
        折得出较大那一股的字节 (非相位路径在 builders/gmm1.py、gmm2.py 里按字节直接申报).
  GMM1  载入走 GM→L1: 占 L1 缓冲槽 + gm_to_l1 信道; 计算占 Cube 队列.
        load / cube 相位时长取事件 meta 的 load_us / compute_us (公式分解).
  GMM2  载入走 GM→L1: 占 L1 缓冲槽 + gm_to_l1 信道 + Cube 队列. 载入 = max(A流, B流),
        A 流只在 activation->gmm2 落 GM (物化) 时存在.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config.hardware import BW_L1_GM, KernelConfig
from ..scheduler.events import Event
from ..config.pipeline import PipelineConstraints

CH_GM_TO_L1 = "gm_to_l1"
CH_HBM_WRITE = "hbm_write"
#: COMBINE 把 GMM2 tile + 路由元数据从 GM 读回 UB。单独一条通路名: 它既不是
#: gm_to_l1 (目的地是 UB 不是 L1, 且 bounds 的算法字节只数 GMM 的 A/B 流), 也不是
#: hbm_write (那是写)。混进任何一条都会让别处的断言失去意义。
CH_COMBINE_READ = "combine_read"
# dispatch 的访存: 读源卡窗口 → UB, 写本卡 workspace。
# 这些名字现在只是**访存量的分类标签** (信道模型已停用), 按通路汇总在
# rank_results["traffic_bytes"] 里。要重建争用模型, 得先有**整卡访存带宽**的实测。
#: prefetch 路径 (TopkWeightsPrefetch=true) 的 ACT 把 GMM1 输出 + topk 权重从 GM
#: 读回 UB。单独一条通路名, 理由同 CH_COMBINE_READ: 目的地是 UB 不是 L1, 也不是写。
CH_ACT_READBACK = "act_readback"
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
) -> Tuple[List[Event], Dict[str, int]]:
    """施加约束, 返回 (事件表, 容量表).

    aic_num/h: 形状参数 (保留以兼容调用方; 相位时长取自事件 meta)
    gmm1_act_depth: gmm1->activation 那条边的 depth, BUF 槽位默认值
    """
    km = kernel if kernel is not None else KernelConfig()
    if km.l1_buf_num == 1 and cons.queues.mte_aic > 1:
        raise ValueError(
            f"KernelConfig.l1_buf_num=1 (单 L1 缓冲) 与 queues.mte_aic="
            f"{cons.queues.mte_aic} (多个 L1 缓冲槽) 矛盾: 二者描述同一个硬件资源")
    by_name = {ev.name: ev for ev in events}

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
    # 距离依赖天然保序无环, 深度由 gmm1->activation 那条边的 depth 参数化.

    # ---- L1/L2: 队列计数信号量 + 信道需求 (gmm1 需要时拆相位) ----
    # 深度 1 = 闭式时长整体标注; 深度 >1 才有跨 tile 的 load/cube 重叠可建模
    split_gmm1 = cons.queues.mte_aic > 1
    new_events: List[Event] = []
    for ev in events:
        stage = str(ev.meta.get("stage", ""))
        if stage == _STAGE_GMM1:
            new_events.extend(_expand_gmm1(ev, cons, split_gmm1, by_name, h, km))
        elif stage == _STAGE_GMM2:
            # split 与 GMM1 同一个开关: 深度 1 时整段闭式时长记在 AIC 上 (载入含在
            # 其中, 本来就被 AIC 独占串起来, 不会多出并发), 所以**不拆** ——
            # PipelineConstraints() 这种中性约束必须与不开相位流水逐字节一致
            # (tests/test_api_smoke.py::test_neutral_pipeline_invariance)。
            new_events.extend(_annotate(ev, split=split_gmm1,
                                        queues=("QUEUE:mte_aic", "QUEUE:cube")))
        elif stage == _STAGE_ACT:
            new_events.extend(_expand_aiv(
                ev, cons, by_name, vec=True, km=km,
                load_bw=cons.phases.act_load_bw_bytes_per_us))
        elif stage == _STAGE_COMBINE:
            new_events.extend(_expand_aiv(
                ev, cons, by_name, vec=False, km=km,
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
        if res.startswith(("MTE2:", "FIXPIPE:", "MTE_AIV:")):
            # **执行单元**, 容量恒为 1 —— 这是硬件事实, 不是参数:
            #   MTE2     一个 AIC 一条 (GM→L1 搬运)
            #   FIXPIPE  一个 AIC 一条 (L0C→UB/GM 搬出)
            #   MTE_AIV  一个 AIV 一条 (GM↔UB 搬运); 名字带 aiv0/aiv1 是因为
            #            AIC:c7 配的两个 AIV 是**两个**物理核, 各有自己的 MTE。
            # 与 QueueDepths 的区别见 config/pipeline.py 的 QueueDepths 文档:
            # 队列深度 = 在飞上限/缓冲槽数 (能攒多少笔), 执行单元 = 同时能跑几笔。
            # 两者在深度 1 时重合, 所以缺省下这几条是空操作; 深度 >1 才分开。
            #
            # 用计数信号量而不是独占资源, 是为了走 late-bind 的 "c*" 占位重映射
            # (model._rewrite_for_late_binding 只改写 acquires/releases 的核后缀) ——
            # 写成独占资源会把相位固定在建图时的占位核号上, 与它所在相位组绑定的核冲突。
            capacities[res] = 1
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
    return new_events, capacities


def _annotate(
    ev: Event,
    *,
    queues: Tuple[str, ...],
    split: bool = False,
) -> List[Event]:
    """GMM2 head/tail: 保留原结构, 挂队列计数信号量 (L1 缓冲槽 + Cube) 与 B 流信道.

    head/tail 是同一个 tile 的两段, 各按自己的时长占比分摊该 tile 的 B 流字节。

    **载入也要占本核的 MTE2 管道**: GMM2 的 GM→L1 与 GMM1 的走同一条 MTE2
    (一个 AI Core 只有一条)。2026-10-05 之前 GMM2 的载入整段裹在 AIC 事件里, 等于
    给每个核**第二条载入管道** —— 于是 GMM1 的载入 (占 MTE2) 与 GMM2 的载入 (占 AIC)
    可以在同一个核上同时满带宽跑, 聚合载入带宽翻倍, 墙钟低于带宽下界。
    这里按 meta 里的载入份额拆出一个前置 .ld 事件占住 MTE2 (与 ACT/COMBINE 的
    _expand_aiv 同一套做法), 主事件只留计算份额。
    """
    core = ev.meta.get("core")
    qs = tuple((f"{queue}:c{core}", 1) for queue in queues)
    # 字节**原样取建图器申报的那一份**, 不从时长倒推 (见本模块顶部的口径说明)。
    ch = ev.channel_bytes
    load_us = ev.meta.get("load_us")      # 该事件自己的载入份额, 不是整段时长
    pre: List[Event] = []
    deps, dur = ev.deps, ev.duration_us
    compute_us = ev.meta.get("compute_us")
    if split and load_us and compute_us is not None and core is not None:
        mte2 = (f"MTE2:c{core}", 1)
        ld = Event(
            name=ev.name + ".ld", resources=(),
            duration_us=float(load_us), deps=ev.deps, order=ev.order,
            meta=dict(ev.meta, phase="load"),
            dep_latency_overrides=ev.dep_latency_overrides,
            acquires=(mte2,), releases=(mte2,),
            channel_bytes=ch,
        )
        pre = [ld]
        # 主事件时长 = 原时长 - 载入份额, **不是** meta 里的 compute_us:
        # ev.duration_us 里除了 load+compute 还可能有别的项 (首次占核的
        # gmm2_problem_startup_us、serial 口径下的 chunk_restart)。拿 compute_us 当
        # 主事件时长会把这些项悄悄丢掉 —— 实测丢了 7.96us, 被
        # tests/test_scheduler.py::test_phase_split_self_consistency 抓住
        # (它断言拆相位前后 stage 忙碌时长逐位一致)。
        deps, dur, ch = (ld.name,), max(0.0, ev.duration_us - float(load_us)), ()
    return pre + [Event(
        name=ev.name, resources=ev.resources, duration_us=dur,
        deps=deps, order=ev.order, meta=dict(ev.meta),
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
    base_dur = ev.duration_us
    load_us = ev.meta.get("load_us")
    compute_us = ev.meta.get("compute_us")
    if load_us is None and split:
        raise ValueError(
            f"{ev.name}: 缺 A 流/计算分解 — gmm1_tile 是自定义 callable. "
            "相位拆分 (queues.mte_aic > 1) 需要 AnalyticalGmmCosts 的公式")
    # 字节**原样取建图器申报的那一份** (builders/gmm1.py 按 A 流 m·K + B 流 wb·K·cols
    # 逐项算出), 不从时长倒推 —— 见本模块顶部的口径说明。
    #
    # 已声明未建模: 本批 run 是 MXFP8 (config.json5 的 dtype=fp8_e5m2 ->
    # PROFILE_QUANT=E5M2_QUANT), 载入还有 MX scale 两条流 (A-scale m*k/32,
    # B-scale 2*k*cols/32, 合计 +3.1%), gmm1_phases 与建图器的字节申报都没算。
    # 方向上模型已经偏高 (1 个 m-group 的两个形状 +3.5%/+1.3%), 补上 scale 会更高 ——
    # 缺的不是这几个字节, 是 BW_L1_GM 本身 (其出处标签已写"旧口径下标定, 待重标")。
    ch = ev.channel_bytes

    if not split:
        # 整体标注: 事件时长 = 闭式时长; A 流被信道切速到超过它时随之拉长
        q = (f"QUEUE:mte_aic:c{core}", 1)
        return [Event(
            name=ev.name, resources=ev.resources, duration_us=base_dur,
            deps=ev.deps, order=ev.order, meta=dict(ev.meta),
            dep_latency_overrides=ev.dep_latency_overrides,
            acquires=ev.acquires + (q,), releases=ev.releases + (q,),
            channel_bytes=ch,
        )]

    # 数据释放事件 (结果 L0C -> GM/UB) 忽略不计 —— 与闭式公式同口径
    # (gmm1_phases/gmm2_phases 都只计搬入, 不计写出)。fix 相位保留成 0 时长的节点:
    # 它仍然承载 QUEUE:fix 与归还 mte 缓冲槽的语义, 只是不再贡献时长。
    # 因此 PipelineConstraints.phases.fix_bw_bytes_per_us 不再影响任何时长。
    fix_dur = 0.0
    closed_form = max(load_us, compute_us)      # 走到这里必为双缓冲 (入口已校验)
    overhead = max(0.0, base_dur - closed_form)
    load_longer = load_us > compute_us

    # 缓冲占用语义: mte 计数信号量 = L1 缓冲槽.
    # grant 开始时取, fix 结束时还 (载入与计算都完成, 缓冲才空出来) —
    # QueueDepths(mte_aic=d) 因此精确等于 d 个 L1 缓冲. 相位事件不继承**引擎**
    # 信号量 (Q:aic), 否则容量 1 的引擎信号量会卡死 mte 深度 (审计已确认的旧 bug).
    #
    # 但引擎队列之外的 acquire 必须跟到 lg 上 (C2 的 UB:gmm1act 槽就是这种): 它由
    # 另一个事件 (配对 ACT) 归还, 丢掉 acquire 只剩 release 会让计数器变负 —— 等于
    # 这条约束悄悄失效。拆相位前后持有区间不变: 原事件 start == lg.start。
    mte = (f"QUEUE:mte_aic:c{core}", 1)
    carried = tuple(a for a in ev.acquires if not a[0].startswith("Q:"))
    lg = Event(
        name=ev.name + ".lg", resources=(), duration_us=overhead,
        deps=_drop_program_order(ev, stage, by_name), order=ev.order,
        meta=dict(ev.meta, phase="grant"),
        dep_latency_overrides=ev.dep_latency_overrides,
        acquires=(mte,) + carried,
    )
    # 载入相位独占**本核的 MTE2 管道** (GM→L1 搬运单元)。这是硬件事实:
    # 一个 AI Core 只有一条 MTE2, 所以同一个核上同时只能有一笔 GM→L1 在飞。
    # 双缓冲 (l1_buf_num / queues.mte_aic = L1 缓冲槽数) 决定能提前多少发起下一笔,
    # **不是**能同时搬几笔 —— 两件事以前被当成了一件:
    #   2026-10-05 之前 .ld 的 resources=() (不占任何资源), 于是载入的并发只受
    #   QUEUE:mte_aic 的深度 d 限制 = 28 核 x d 笔同时按满带宽搬。d=2 时聚合载入带宽
    #   达到 2.9e6 B/us, 是每核规格 (28 x 51900 = 1.45e6) 的两倍, 墙钟因此低于带宽
    #   下界 26.6% (见 analysis/bounds.py 与 docs/design_space_gaps.md "下界与漏账")。
    # 占住 MTE2 之后载入并发上限 = 核数, 聚合载入带宽自动不超过 核数 x BW_L1_GM,
    # 带宽下界由构造满足, 不需要速率服务器。
    mte2 = (f"MTE2:c{core}", 1)
    ld = Event(
        name=ev.name + ".ld", resources=(), duration_us=load_us,
        deps=(lg.name,), order=ev.order,
        meta=dict(ev.meta, phase="load", overlapped=not load_longer),
        acquires=(mte2,), releases=(mte2,),
        channel_bytes=ch,
    )
    cb = Event(
        name=ev.name + ".cb", resources=ev.resources, duration_us=compute_us,
        deps=(lg.name,), order=ev.order,
        meta=dict(ev.meta, phase="cube", overlapped=load_longer),
        acquires=((f"QUEUE:cube:c{core}", 1),),
        releases=((f"QUEUE:cube:c{core}", 1),),
    )
    # fix 相位独占本核的 **FixPipe** (L0C→UB/GM 的搬出单元): 一个 AIC 一条。
    # 与 MTE2 同一个道理 —— QUEUE:fix 的深度是"在飞上限/缓冲槽数", 决定能攒多少笔
    # 待搬出; FIXPIPE 容量 1 才是"同时能搬几笔"。深度恒为 1 时两者重合, 深度 >1 时
    # 不补这一条就等于给每个核多出几条 FixPipe。
    fixpipe = (f"FIXPIPE:c{core}", 1)
    fx = Event(
        name=ev.name, resources=(), duration_us=fix_dur,
        deps=(ld.name, cb.name), order=ev.order, meta=dict(ev.meta, phase="fix"),
        acquires=((f"QUEUE:fix:c{core}", 1), fixpipe),
        releases=((f"QUEUE:fix:c{core}", 1), fixpipe, mte),
    )
    return [lg, ld, cb, fx]


def _expand_aiv(
    ev: Event,
    cons: PipelineConstraints,
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
    # 建图器自己申报的字节原样保留 (ACT 的 GM 写出、COMBINE 的片间写与本卡读写)。
    # 2026-10-05 之前这里还会**追加** (CH_HBM_WRITE, base_dur x BW_SCATTER) ——
    # 从时长倒推字节, 方向是反的; 用的 BW_SCATTER 自己标着"旧口径, 已不用";
    # 而且它只在开了相位流水时出现 —— 换一个编排参数不该改变搬了多少字节。
    # COMBINE 的本卡读回与本卡行写现在由 builders/comm/peerwrite.py 按字节直接申报。
    ch = ev.channel_bytes

    need_split = load_bw is not None and base_dur > 0
    if not need_split:
        return [Event(
            name=ev.name, resources=ev.resources, duration_us=base_dur,
            deps=ev.deps, order=ev.order, meta=dict(ev.meta),
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
        dep_latency_overrides=ev.dep_latency_overrides,
        acquires=ev.acquires + ((f"QUEUE:mte_aiv:{eng}:c{core}", 1),
                                (f"MTE_AIV:{eng}:c{core}", 1)),
        releases=((f"QUEUE:mte_aiv:{eng}:c{core}", 1),
                  (f"MTE_AIV:{eng}:c{core}", 1)),
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
