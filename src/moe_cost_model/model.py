"""Wave cost model 编排层 — 建图 + 调度 + 后处理.

"""
from __future__ import annotations

import dataclasses
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import registry
from .builders.mte import MteEventBuilder
from .config.hardware import KernelConfig, ceil_div
from .scheduler.events import POOL_WILDCARD, Event, ScheduledEvent
from .scheduler.engine import MultiResourceScheduler
from .scheduler.policies import (CriticalPathFirst, EarliestStart, PriorityByStage,
                                 WorkConservingCriticalPath)
from .builders.barriers import apply_barriers
from .builders.pipeline_expand import CH_HBM_WRITE, apply_pipeline
from .costs import PrimitiveCosts
from .analysis.bounds import attach_bounds
from .implementations import CompileConfig, RuntimeConfig, WavePlan
from .implementations.megamoe import (_BuilderShim, adapter_for,
                                      resolve as resolve_adapter)
from .guardrails import check_shape_conservation
from .shape import (
    CursorTrace, MegaMoeShape, ModelOptions,
)
from .planning.waves import (
    Wave, calc_m_groups_per_wave, plan_waves, plan_layered_waves,
)


#: 可晚绑定的角色池。moe_stage_done 之类"代表某个核"的零时长栅栏不参与。
POOLABLE_ROLES = ("AIC", "AIV0", "AIV1")
_CORE_SUFFIX = re.compile(r"c\d+$")


def _rewrite_for_late_binding(events: List[Event], roles: Sequence[str], aic_num: int,
                             pre: str, pools: Dict[str, Tuple[str, ...]],
                             act_role: str = "AIV0") -> None:
    """把指定角色的核资源从"建图时定死"改成"派发时绑定最早空闲的核"。

    做法: 事件声明 "AIC:*" 而不是 "AIC:7", 调度器在准入那一刻挑成员 (engine 的
    _candidates/_bind)。每核的自取自还队列信号量 (Q:aic:c7) 在独占核资源下是空约束,
    直接去掉 (否则落到核 3 的 tile 会去扣核 7 的队列深度)。

    **不删不改任何已有依赖边**: 事件名、deps、dep_latency 原样, 变的只有"哪个核来做"。

    物理约束用 colocate_with 保住:
      GMM1 -> ACT 同核 (L0C->UB Fixpipe 直给配对 AIV0), 所以 AIC 入池即隐含 AIV0 入池,
      且每个 ACT 必须与它的 GMM1 落同一核号。
      dispatch 的每波调用开销改成"该核本波第一段 dispatch 的一次性开销"
      (Event.once_per_core), dispatch_call 事件本身归 0 —— 不钉核, 不加边。
      GMM2 -> combine 不是物理共位 (GMM2 写 GM, combine 读 GM), 故不加约束。
    """
    pooled = set(roles)
    if "AIC" in pooled:
        # ACT 必须与它的 GMM1 同核 (L0C->UB 的 Fixpipe 只在绑定对内), 所以 GMM1 入池
        # 就隐含"跑 ACT 的那个向量角色"一起入池 —— 整对漂移。哪个角色跑 ACT 由
        # ModelOptions.roles 决定, 不写死 AIV0。
        pooled.add(act_role)
    bad = sorted(pooled - set(POOLABLE_ROLES))
    if bad:
        raise ValueError(f"late_bind_pools 只支持 {POOLABLE_ROLES}, 收到 {bad}")
    for role in sorted(pooled):
        pools[f"{pre}{role}"] = tuple(f"{pre}{role}:{c}" for c in range(aic_num))

    stage_of = {e.name: str(e.meta.get("stage", "")) for e in events}
    dur_of = {e.name: e.duration_us for e in events}
    for ev in events:
        if str(ev.meta.get("stage", "")) == "moe_stage_done":
            continue  # 每核排空栅栏: 它代表的就是那个核, 不能漂移
        new_res, touched = [], False
        for r in ev.resources:
            role, _, core_s = r.partition(":")
            if core_s and role in pooled:
                new_res.append(role + POOL_WILDCARD)
                touched = True
            else:
                new_res.append(r)
        if not touched:
            continue
        ev.resources = tuple(new_res)
        core = ev.meta.get("core")
        if isinstance(core, int):
            # 按核的计数信号量分两类:
            #
            # 1) 自取自还 (Q:aic:c7 之类的引擎队列): 事件同时独占该核资源, 同核在途数
            #    恒 <= 1, 容量 1 就已经不起约束 (见下面声明容量处的说明)。
            #    **去掉**, 语义不变, 也省掉一次核号解析。
            # 2) 跨事件持有 (UB:gmm1act:c7 —— GMM1 取、配对 ACT 还): 真约束, 核号换成
            #    占位 c*, 由引擎在派发时回填 (_tok / _remap_tokens)。取与还必须落同一个
            #    核号, 这由共位保证 (ACT 的 colocate_with 指向它的 GMM1)。
            self_paired = set(ev.acquires) & set(ev.releases)

            def _norm(tok):
                t, k = tok
                if not _CORE_SUFFIX.search(t):
                    return tok
                return (_CORE_SUFFIX.sub("c*", t), k)

            ev.acquires = tuple(_norm(a) for a in ev.acquires if a not in self_paired
                                or not _CORE_SUFFIX.search(a[0]))
            ev.releases = tuple(_norm(r) for r in ev.releases if r not in self_paired
                                or not _CORE_SUFFIX.search(r[0]))
        if str(ev.meta.get("stage", "")) == "activation":
            anchor = next((d for d in ev.deps if stage_of.get(d) == "gmm1"), None)
            if anchor is not None:
                ev.colocate_with = anchor
        if str(ev.meta.get("stage", "")) == "dispatch" and ".dispatch." in ev.name:
            # 调用开销 = 该核这一波第一段 dispatch 的一次性开销 (once_per_core),
            # 不再靠单独钉核的 dispatch_call 事件承担: 段可落任意空闲核, 开销
            # 落在真正搬数据的那个核上。无新增依赖边。
            call = f"{ev.name.split('.dispatch.', 1)[0]}.dispatch_call.c{core}"
            if stage_of.get(call) == "dispatch_call" and dur_of[call] > 0:
                ev.once_per_core = (call.rsplit(".c", 1)[0], dur_of[call])

    # dispatch_call 的开销已转给各段 dispatch 的 once_per_core, 事件本身留作 0 时长
    # 的标记 (名字仍被 dispatch_ready 的 meta 引用)。
    if "AIV1" in pooled:
        for ev in events:
            if str(ev.meta.get("stage", "")) == "dispatch_call":
                ev.duration_us = 0.0

    # 相位拆分事件 (.lg/.ld/.cb/fix 以及 AIV 的 .ld/main) 自己不持核资源 —— 它们代表
    # 同一个核里不同引擎 (MTE / Cube / Fixpipe) 的工作, 在时间上重叠, 所以不能各自
    # 独占核资源 (那就被迫串行, 拆了等于没拆)。它们"属于哪个核"原先靠名字里写死核号的
    # 按核计数信号量 (QUEUE:mte_aic:c7) 记着, 而晚绑定下核号到派发时刻才定, 于是:
    #   * 回填不了 -> 工作在 3 号核跑、L1 槽从 7 号核扣, 约束等于失效 (偏快);
    #   * 这就是原先直接拒绝两者同用的原因。
    #
    # 现在用**核组**解决: 同一个 tile 的几个相位编成一组, 核号由该组最先派发的那个
    # 事件选定, 同组其余事件跟随 (Event.core_group, 引擎的 group_core)。为什么不能用
    # colocate_with: 它要求锚点先绑定, 而先跑的恰恰是不持核资源的那一相。
    _group_phase_events(events, pooled, pre)


def _group_phase_events(events: List[Event], pooled: Sequence[str], pre: str) -> None:
    """给相位拆分出来的事件编核组, 并把它们按核的 token 换成占位 c*.

    组名 = 去掉相位后缀的事件名 (同一个 tile 的各相位同组)。角色取自同组里**持核
    资源**的那个相位 (AIC 的 cube 相、AIV 的主相), 它决定候选核表。

    没有相位拆分时本函数什么都不做 (没有不持核资源却带按核 token 的事件)。
    """
    def group_of(name: str) -> str:
        for suffix in (".lg", ".ld", ".cb"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    # 组 -> 角色: 由持核资源的那个相位给出
    role_of: Dict[str, str] = {}
    for ev in events:
        for r in ev.resources:
            role, _, core_s = r.partition(":")
            if core_s and role in pooled:
                role_of.setdefault(group_of(ev.name), role)
    need = {ev.name for ev in events
            if not ev.resources
            and any(_CORE_SUFFIX.search(t) for t, _ in ev.acquires + ev.releases)}
    if not need:
        return
    for ev in events:
        gid = group_of(ev.name)
        role = role_of.get(gid)
        if role is None:
            continue
        # 组里**每个**相位都要带核组: 不光是扣槽的那几个 —— 否则不扣槽的相位 (.ld)
        # 的 meta["core"] 会留着建图时的占位核号, 下游按核分组时就对不上。
        ev.core_group = (pre + gid, pre + role)
        if ev.name in need:
            ev.acquires = tuple((_CORE_SUFFIX.sub("c*", t), k) for t, k in ev.acquires)
            ev.releases = tuple((_CORE_SUFFIX.sub("c*", t), k) for t, k in ev.releases)
    missing = sorted(n for n in need if role_of.get(group_of(n)) is None)
    if missing:
        raise NotImplementedError(
            f"相位事件 {missing[0]} 所在的组里没有持核资源的相位, 无从确定候选核表")


def _reconcile_act_to_gmm2(costs: PrimitiveCosts, mode: str) -> PrimitiveCosts:
    """让 gmm2 公式的 A 流口径与 activation->gmm2 这条边的落点一致.

    gmm2_tile 是自定义 callable 时无从改写, 原样返回 (调用方自负一致)。
    """
    import dataclasses

    from .costs import AnalyticalGmmCosts
    owner = getattr(costs.gmm2_tile, "__self__", None)
    if not isinstance(owner, AnalyticalGmmCosts):
        return costs
    want = (mode == "gm")
    if owner.gmm2_a_from_gm == want:
        return costs
    new_g = AnalyticalGmmCosts(
        bw_bytes_per_us=owner.bw,
        weight_nz=owner.weight_nz,
        bw_b_nz_bytes_per_us=(owner.bw_b if owner.weight_nz else 0.0),
        l1_buf_num=1 if owner.serial else 2,
        cube_mac_per_us=owner.cube_rate,
        tile_restart_us=owner.chunk_restart,
        l1_tile_k=owner._k_l1,
        gmm1_weight_blocks=owner.wb,
        gmm2_a_from_gm=want,
        load_overlap=owner.load_overlap)
    return dataclasses.replace(costs, gmm1_tile=new_g.gmm1_tile,
                               gmm2_tile=new_g.gmm2_tile)


def _apply_onchip_act_to_gmm2(events: List[Event], pooled: Sequence[str]) -> None:
    """activation->gmm2 这条边落片上时的共位约束: 一个 m-group 的全部工作同核.

    为什么是整个 m-group: GMM2 的 K 就是 GMM1 切分的 N 轴, 一个 GMM2 tile 要累完
    整个 K, 即吃该 m-group 的全部 ACT。A 不落 GM 就只能在产它的那个核的片上, 所以
    产它的 ACT 和吃它的 GMM2 必须同核; 对一个 m-group 的所有 ACT 同时成立 =>
    这些 ACT (以及产它们的 GMM1) 彼此同核。

    后果 (这就是该编排要被评估的那个代价): 一个 m-group 的全部工作串在一个核上,
    并行度上限 = m-group 数。核数多于 m-group 数时, 多出来的核**无活可做** ——
    不是违反"有就绪的活就不空闲", 是这个编排本身没有可并行的活。

    ACT 的 GM 写出同时取消 (store_bytes 申报清零): 它本来就是为了让 GMM2 从 GM
    读回。注意 activation_tile 的时长里含写出那部分, 当前公式不可分, 所以时长**没有**
    相应变短 —— 这一项是已声明的保守近似。
    """
    if "AIC" not in set(pooled):
        raise ValueError(
            'StageLink("activation","gmm2", location="onchip") 需要 late_bind_pools '
            '含 "AIC": 共位靠派发时刻绑定表达, 建图时静态钉核无从表达 '
            "(钉死的核号本来就各不相同)")
    anchor: Dict[tuple, str] = {}
    for ev in events:
        stage = str(ev.meta.get("stage", ""))
        if stage != "gmm1":
            continue
        key = (ev.meta.get("wave"), ev.meta.get("expert"), ev.meta.get("slice"),
               ev.meta.get("mgroup"))
        anchor.setdefault(key, ev.name)
    for ev in events:
        stage = str(ev.meta.get("stage", ""))
        if stage == "activation":
            ev.channel_bytes = tuple(c for c in ev.channel_bytes
                                     if c[0] != CH_HBM_WRITE)
            ev.meta = dict(ev.meta, store_bytes=0)
            continue
        if stage not in ("gmm1", "gmm2"):
            continue
        key = (ev.meta.get("wave"), ev.meta.get("expert"), ev.meta.get("slice"),
               ev.meta.get("mgroup"))
        a = anchor.get(key)
        if a is not None and a != ev.name:
            ev.colocate_with = a


def completion_event(scheduled: Sequence[ScheduledEvent]) -> Optional[ScheduledEvent]:
    """执行时间的终点事件: 最晚结束的 COMBINE.

    执行时间记到最后一个 COMBINE 结束为止; 其后的尾段 (counts_export /
    core_sync / rank_sync / unpermute / finalize, 以及共享专家的 GMM2) 仍在
    事件图里照常调度, 但不计入执行时间. 没有 COMBINE 事件时退回最晚结束的事件.
    """
    if not scheduled:
        return None
    combines = [e for e in scheduled if e.meta.get("stage") == "combine"]
    return max(combines or scheduled, key=lambda e: (e.end_us, e.order, e.name))


class A8W8WaveCostModel:
    def __init__(self, costs: PrimitiveCosts, options: ModelOptions = ModelOptions()):
        # COMBINE 的量化 (CombineQuantMode) 由 KernelConfig.combine_quant_mode 表达, 已建模
        # (写侧每元素字节随之变)。这里原有一道 ModelOptions.combine_no_quant 的门, 拒绝
        # "量化 combine" —— 与 combine_quant_mode=1 能跑互相矛盾, 同一个事实两个说法。
        # 2026-10-05 删掉那个字段与门, 只留 combine_quant_mode 一个真相。
        # 编排与公式必须同口径: activation->gmm2 这条边落片上时 GMM2 的 A 不付 GM
        # 字节, 落 GM 时要付。调用方给的 costs 可能两边都不是, 这里按选定的编排改写
        # 公式, 不让两套口径混在一张图里 (混着就会把物化算成近乎免费)。
        self.costs = _reconcile_act_to_gmm2(
            costs, options.link("activation", "gmm2").location)
        self.options = options
        self._order = 0
        self._rank = 0
        self.cursor_traces: Dict[int, List[CursorTrace]] = {}
        self._wave_plans: Dict[tuple, WavePlan] = {}

    def _kernel_cfg(self, shape: MegaMoeShape) -> KernelConfig:
        return shape.kernel if shape.kernel is not None else KernelConfig()

    def compile_config(self, shape: MegaMoeShape) -> CompileConfig:
        """本次运行的编译点. 由 shape.kernel (KernelConfig) 的编译轴得出, 带指纹."""
        return CompileConfig.from_kernel_config(self._kernel_cfg(shape))

    def adapter(self, shape: MegaMoeShape):
        """这个 shape 用哪份实现的适配器.

        优先 shape.orchestration (场景文件里写 orchestration = "layered" 之类, 或直接给
        适配器/建图器类), 否则按编译点的 comm_mode 选 —— 即原先的 `km.topo_urma` 分支,
        现在走 CompileConfig.comm_mode (对应 kernel 的 TILINGKEY_COMM_MODE)。
        """
        want = getattr(shape, "orchestration", None)
        if want is not None:
            # 旧路径: registry 里注册的是**建图器类**, 不是适配器。两者都要能用, 所以
            # 先问 registry, 认不出来再问适配器表。
            cls = registry.builder_class(want)
            if cls is not None:
                return _BuilderShim(cls)
            return resolve_adapter(want)
        return adapter_for(self.compile_config(shape))

    def wave_plan(self, shape: MegaMoeShape) -> WavePlan:
        """波计划, 每个 (rank, 形状) 只算一次.

        为什么缓存: _postprocess 要把 wave_count / m_groups_per_wave / waves 放进结果
        (wave_count 进 golden 指纹), 而它原先**第二次调用** self.waves(shape) 重算。
        算两遍就有两条路径可以漂, 所以这里算一次, 建图与后处理共用同一个对象。
        """
        key = (shape.rank_id, id(shape))
        got = self._wave_plans.get(key)
        if got is None:
            got = self.adapter(shape).plan(shape, self.compile_config(shape), self.options)
            self._wave_plans[key] = got
        return got

    def m_groups_per_wave(self, shape: MegaMoeShape) -> int:
        return self.wave_plan(shape).m_groups_per_wave

    def waves(self, shape: MegaMoeShape) -> List[Wave]:
        return list(self.wave_plan(shape).waves)

    def build_events(self, shape: MegaMoeShape) -> Tuple[List[Event], List[CursorTrace]]:
        """降解: 选适配器 -> 校验编译点 -> 用缓存的波计划展开事件图."""
        adapter = self.adapter(shape)
        adapter.accepts(self.compile_config(shape), self.options)
        return adapter.lower(shape, self.wave_plan(shape), self.costs, self.options)


    def simulate(self, shape: MegaMoeShape, restructure=None) -> Dict[str, object]:
        return self.simulate_multi([shape], restructure=restructure)[shape.rank_id]

    def simulate_multi(self, shapes: Sequence[MegaMoeShape],
                       restructure=None) -> Dict[int, Dict[str, object]]:
        """多 rank 调度: 核/队列资源按 rank 前缀隔离.

        信道模型 (速率服务器) 已于 2026-10-03 停用: 事件的 channel_bytes 仍然申报
        字节, 但只在 rank_results["traffic_bytes"] 里汇总成访存量, 不参与准入、
        不影响任何时长。片间 fab 通路同理 —— 它的两个常数本来就不同尺度
        (聚合 BW_WINDOW=33000 是整卡值, 逐事件 BW_REMOTE_GM=31000 是从 28 核并发
        反解的单核值, 已含平均争用), 叠速率服务器会把争用计两遍。

        各 rank 之间无共享资源时逐 rank 独立调度 (结果与合并调度逐位一致,
        见 _ranks_independent); 否则全部事件进同一个调度器.
        """
        # 路由守恒是算法事实, 在**这里**查而不是只在 api 里查: run_shapes 这类直达
        # simulate_multi 的入口原先绕过了它 (golden 有两个 case 一直在给不可能的输入建图)。
        # 与 attach_bounds 放在这里是同一个理由 —— 没有入口能绕过护栏。
        bad = check_shape_conservation(shapes)
        if bad:
            raise ValueError("routing 不守恒 (每个源 rank 应发出 token_num x topk 行):\n  "
                             + "\n  ".join(bad))
        per: List[Tuple[MegaMoeShape, List[Event], Dict, List[CursorTrace]]] = []
        self._sched_policy = getattr(shapes[0], 'scheduling_policy', None) if shapes else None
        if shapes:
            mism = [d for d, sh in enumerate(shapes)
                    if getattr(sh, 'scheduling_policy', None) is not self._sched_policy
                    and getattr(sh, 'scheduling_policy', None) != self._sched_policy]
            if mism:
                raise ValueError(
                    f"各 rank 的 scheduling_policy 不一致 (rank {mism} 与 rank 0 不同); "
                    "多 rank 单调度器只支持同一策略")
        for shape in shapes:
            events, trace = self.build_events(shape)
            self.cursor_traces[shape.rank_id] = trace
            if self.options.barriers:
                events = apply_barriers(
                    events, self.options.barriers,
                    ub_depth=self.options.gmm1_act_link(shape.kernel).depth,
                    aic_num=shape.aic_num,
                    pooled=bool(self.options.late_bind_pools))
            caps: Dict = {}
            if self.options.pipeline is not None:
                events, caps = apply_pipeline(
                    events, self.options.pipeline,
                    aic_num=shape.aic_num, h=shape.h,
                    gmm1_act_depth=self.options.gmm1_act_link(shape.kernel).depth,
                    kernel=shape.kernel)
            per.append((shape, events, caps, trace))

        # 每 rank 一组 (事件, 容量, 资源池)
        groups: List[Tuple[List[Event], Dict[str, int],
                           Dict[str, Tuple[str, ...]]]] = []
        late = tuple(self.options.late_bind_pools or ())
        for shape, events, caps, trace in per:
            pre = f"R{shape.rank_id}."
            pools: Dict[str, Tuple[str, ...]] = {}
            if late:
                _rewrite_for_late_binding(
                    events, late, shape.aic_num, pre, pools,
                    act_role=self.options.roles.role_of("activation"))
            if late:
                _charge_late_bind_fetch(events, late, self.costs)
            if self.options.link("activation", "gmm2").location == "onchip":
                # 共位在加 rank 前缀之前打: colocate_with 记的是事件名, 不带前缀。
                _apply_onchip_act_to_gmm2(events, late)
            capacities: Dict[str, int] = {}
            for ev in events:
                ev.resources = tuple(pre + r for r in ev.resources)
                ev.acquires = tuple((pre + a, k) for a, k in ev.acquires)
                ev.releases = tuple((pre + r, k) for r, k in ev.releases)
                # 访存量申报全部保留 (不再按"开了哪些信道"过滤): 它只做统计。
                # 片间通路跨 rank 共享, 不加前缀; 其余按 rank 前缀隔离。
                ev.channel_bytes = tuple(
                    (c, b, rt) if c.startswith("fab_") else (pre + c, b, rt)
                    for c, b, rt in ev.channel_bytes)
            # 每核引擎队列的容量恒为 1, 不设旋钮。原因是这个事件代数里 "更深的
            # 队列" 没有可表达的后果: 持核事件独占 AIC/AIV0/AIV1, 同核在途数恒 <= 1,
            # 所以容量 2 与 1 等价; 相位拆分后的 load 相位又刻意不继承 Q:* (继承会让
            # 容量 1 的引擎信号量卡死 L1 缓冲深度, 见 pipeline_expand 的说明), 所以
            # 拆相位也不会让它咬上。要表达 "更深的队列" 必须先有发射开销或在途计数的
            # 物理后果, 模型里没有, 给个旋钮只会让扫描得到 "深了也没用" 的假结论。
            # 2026-10-05 之前这里是 EngineQueueDepths(aic/vec0/aiv1): 四类形状逐个扫过,
            # 任何取值都与容量 1 逐位相同 (golden 的 pipeline_engine_queue2 与
            # pipeline_split 指纹全同可证), 所以它是个无法生效的旋钮, 已删。
            ub_depth = self.options.gmm1_act_link(shape.kernel).depth
            for core in range(shape.aic_num):
                capacities[pre + f"Q:aic:c{core}"] = 1
                capacities[pre + f"Q:vec0:c{core}"] = 1
                capacities[pre + f"Q:aiv1:c{core}"] = 1
                # GMM1->ACT 的 UB 槽位数 (C2: 容量, 不是程序序边)。深度 0 时
                # builders 不申报这个 token, 容量也就不必声明。
                if ub_depth > 0:
                    capacities[pre + f"UB:gmm1act:c{core}"] = ub_depth
            for k, v in caps.items():
                capacities[pre + k] = v
            groups.append((events, capacities, pools))

        sched_pol = getattr(self, '_sched_policy', None)
        if not self._ranks_independent(shapes, restructure, sched_pol):
            # 合并调度: 全部 rank 的事件/容量进同一个调度器
            merged_caps: Dict[str, int] = {}
            merged_pools: Dict[str, Tuple[str, ...]] = {}
            for events, capacities, pools in groups:
                merged_caps.update(capacities)
                merged_pools.update(pools)
            groups = [([ev for events, _, _ in groups for ev in events],
                       merged_caps, merged_pools)]
        scheduled: List[ScheduledEvent] = []
        for events, capacities, pools in groups:
            _, part = MultiResourceScheduler().schedule(
                events, capacities=capacities or None,
                restructure=restructure, policy=sched_pol, pools=pools or None)
            scheduled.extend(part)

        # 访存量汇总: 事件申报的 channel_bytes 不参与准入 (信道模型已停用), 只在
        # 这里按通路累加成字节数, 供"哪条通路搬了多少"的核算用。
        traffic: Dict[int, Dict[str, float]] = {sh.rank_id: {} for sh in shapes}
        for shape, events, _caps, _trace in per:
            tr = traffic[shape.rank_id]
            for ev in events:
                for cname, nbytes, _rate in ev.channel_bytes:
                    tr[cname] = tr.get(cname, 0.0) + float(nbytes)

        # 全部 rank 的容量表 (空闲分解要用它判"这个空闲核的槽还有余量吗")
        all_caps: Dict[str, int] = {}
        for _evs, caps_, _pools in groups:
            all_caps.update(caps_)

        results: Dict[int, Dict[str, object]] = {}
        for shape in shapes:
            rank = shape.rank_id
            evs = [e for e in scheduled if e.meta.get("rank") == rank]
            results[rank] = self._postprocess(shape, evs, all_caps)
            results[rank]["traffic_bytes"] = dict(sorted(traffic[rank].items()))
            # 下界挂在**这一层**, 不是 api 层 —— 护栏不能被绕过。2026-10-05 发现
            # tests/golden_cases.run_shapes 直达本函数并手工拼结果, 于是四个 golden
            # case 完全没跑下界断言。platform 只有 api 层知道, 所以这里按 platform=None
            # 挂 (带宽下界只用每核带宽 x 核数), api 收到 platform 时会重算一遍。
            results[rank]["bounds"] = attach_bounds(
                shape, results[rank], costs=self.costs,
                kernel=shape.kernel, active_cores=shape.aic_num)
            # 每条结果都带上**是谁算的**: 实现 id + 编译指纹 + 运行拓扑。
            # 在这之前, 一个时长数字离开 Python 之后就无从知道它对应哪份 kernel、哪个
            # 编译点、几张卡几个核 —— 而 config/hardware.py 的标定注释恰恰说明这些数
            # 换了编排/拓扑未必还成立。放在这里 (不是 api 层) 是因为 run_shapes 这类
            # 入口直达本函数, 与 bounds/provenance 同一个理由。
            adapter = self.adapter(shape)
            compile_cfg = self.compile_config(shape)
            results[rank]["implementation"] = {
                "id": adapter.identity().key,
                "source_refs": list(adapter.identity().source_refs),
                "compile_fingerprint": compile_cfg.fingerprint,
                "compile_point": compile_cfg.describe(),
                "measured_end_stage": adapter.measured_end_stage(),
                "topology": dataclasses.asdict(
                    RuntimeConfig.from_shape(
                        shape, world_size=len(shapes) if len(shapes) > 1 else None,
                    ).topology),
            }
        return results

    def _ranks_independent(self, shapes: Sequence[MegaMoeShape], restructure,
                           sched_pol) -> bool:
        """各 rank 能否独立调度而不改变结果.

        调度器每步提交就绪集里排序键最小的事件. 资源/容量按 rank 前缀隔离、
        依赖不跨 rank 时, 一个 rank 的就绪集与资源状态只被本 rank 的提交改变,
        合并调度里该 rank 的提交子序列就等于它单独调度的序列.
        以下情况该前提不成立, 走合并调度:
          * 重构钩子 — 上下文是全局视图, 可跨 rank 转移任务;
          * 自定义调度策略 — event_key 能读到全局 tbase/end_by_name;
          * rank_id 重复 — 交给调度器报重名.
        """
        if restructure is not None:
            return False
        if sched_pol is not None and type(sched_pol) not in (
                EarliestStart, CriticalPathFirst, PriorityByStage,
                WorkConservingCriticalPath):
            return False
        ranks = [sh.rank_id for sh in shapes]
        return len(set(ranks)) == len(ranks)

    def _postprocess(self, shape: MegaMoeShape,
                     scheduled: List[ScheduledEvent],
                     capacities: Optional[Dict[str, int]] = None) -> Dict[str, object]:
        waves = self.waves(shape)
        resource_busy: Dict[str, float] = {}
        resource_first: Dict[str, float] = {}
        resource_last: Dict[str, float] = {}
        stage_busy: Dict[str, float] = {}
        stage_first: Dict[str, float] = {}
        stage_last: Dict[str, float] = {}
        stage_dependency_wait: Dict[str, float] = {}
        stage_resource_queue: Dict[str, float] = {}

        for ev in scheduled:
            duration = ev.end_us - ev.start_us
            for resource in ev.resources:
                resource_busy[resource] = resource_busy.get(resource, 0.0) + duration
                resource_first[resource] = min(resource_first.get(resource, ev.start_us), ev.start_us)
                resource_last[resource] = max(resource_last.get(resource, ev.end_us), ev.end_us)
            stage = str(ev.meta.get("stage", "other"))
            # overlapped: 与同 tile 的另一相位并行且较短 (相位流水), 不重复计入
            if not ev.meta.get("overlapped"):
                stage_busy[stage] = stage_busy.get(stage, 0.0) + duration
            else:
                stage_busy.setdefault(stage, 0.0)
            stage_first[stage] = min(stage_first.get(stage, ev.start_us), ev.start_us)
            stage_last[stage] = max(stage_last.get(stage, ev.end_us), ev.end_us)
            stage_dependency_wait[stage] = stage_dependency_wait.get(stage, 0.0) + ev.dependency_wait_us
            stage_resource_queue[stage] = stage_resource_queue.get(stage, 0.0) + ev.resource_queue_us

        resource_span = {r: resource_last[r] - resource_first[r] for r in resource_busy}
        resource_idle = {r: max(0.0, resource_span[r] - resource_busy[r]) for r in resource_busy}
        resource_utilization = {
            r: (resource_busy[r] / resource_span[r] if resource_span[r] > 0 else 0.0)
            for r in resource_busy}

        # 空闲分解 (每个角色池一份): forced = 此刻全局无就绪活, 消不掉;
        # avoidable = 有就绪活却有核空着, 即 work-conservation 违规。约 simulate 的 1%。
        from .analysis.idle import idle_decomposition
        idle_reports = {}
        for role in ("AIC:", "AIV0:", "AIV1:"):
            for rank_tag, rep in idle_decomposition(
                    scheduled, role, capacities=capacities).items():
                idle_reports[f"{rank_tag}.{role.rstrip(':')}" if rank_tag
                             else role.rstrip(":")] = rep

        scheduled_by_name = {ev.name: ev for ev in scheduled}
        gmm1_by_group: Dict[Tuple[int, int], List[ScheduledEvent]] = {}
        for ev in scheduled:
            if ev.meta.get("stage") == "gmm1":
                gmm1_by_group.setdefault(
                    (int(ev.meta["expert"]), int(ev.meta["mgroup"])), []).append(ev)

        dispatch_ready_tiles: List[Dict[str, object]] = []
        for ev in scheduled:
            if ev.meta.get("stage") != "dispatch_ready":
                continue
            expert = int(ev.meta["expert"])
            mgroup = int(ev.meta["mgroup"])
            g1 = gmm1_by_group.get((expert, mgroup), [])
            g1_first = min((x.start_us for x in g1), default=None)
            dispatch_ready_tiles.append({
                "dst_rank": int(ev.meta["dst_rank"]),
                "wave": int(ev.meta["wave"]),
                "expert": expert, "mgroup": mgroup,
                "required_rows": int(ev.meta["required_rows"]),
                "contributed_rows": int(ev.meta["contributed_rows"]),
                "contributor_count": int(ev.meta["contributor_count"]),
                "t_dispatchReady_us": ev.end_us,
                "gmm1_first_start_us": g1_first,
            })
        dispatch_ready_tiles.sort(key=lambda x: (x["t_dispatchReady_us"], x["expert"], x["mgroup"]))

        critical_path: List[Dict[str, object]] = []
        tail = completion_event(scheduled)
        if tail is not None:
            seen = set()
            cur: Optional[ScheduledEvent] = tail
            while cur is not None and cur.name not in seen:
                seen.add(cur.name)
                critical_path.append({
                    "name": cur.name, "stage": str(cur.meta.get("stage", "other")),
                    "start_us": cur.start_us, "end_us": cur.end_us,
                    "duration_us": cur.end_us - cur.start_us,
                    "critical_reason": cur.critical_reason,
                    "critical_parent": cur.critical_parent,
                    "resources": cur.resources})
                cur = scheduled_by_name.get(cur.critical_parent) if cur.critical_parent else None
            critical_path.reverse()

        return {
            "total_us": tail.end_us if tail is not None else 0.0,
            "dag_end_us": max((e.end_us for e in scheduled), default=0.0),
            "m_groups_per_wave": self.m_groups_per_wave(shape),
            "wave_count": len(waves),
            "waves": waves,
            "cursor_trace": self.cursor_traces.get(shape.rank_id, []),
            "events": scheduled,
            "resource_busy_us": resource_busy,
            "resource_span_us": resource_span,
            "resource_idle_us": resource_idle,
            "resource_utilization": resource_utilization,
            "dispatch_ready_tiles": dispatch_ready_tiles,
            "critical_path": critical_path,
            "stage_busy_us": stage_busy,
            "stage_first_start_us": stage_first,
            "stage_last_end_us": stage_last,
            "stage_dependency_wait_us": stage_dependency_wait,
            "stage_resource_queue_us": stage_resource_queue,
            "gmm2_lag_active": (shape.policy.effective_gmm2_lag(shape.token_num) > 0
                                and not self._kernel_cfg(shape).topo_urma),
            # 空闲分解: 区分"DAG 逼出来的"与"有活却空着"。后者才是 work-conservation
            # 违规, 换 tile->核 的绑定方式可回收; 前者只能靠改 DAG 结构 (加深流水、改波
            # 的组成)。见 analysis/idle.py —— avoidable 是上界, 未计共位与队列约束。
            "idle_decomposition": idle_reports,
        }



def _charge_late_bind_fetch(events, late_pools, costs) -> None:
    """晚绑定要付"动态取活"的代价: 每个落到池化角色上的事件加一次取活开销.

    为什么必须计费
    --------------
    静态分核 (late_bind_pools=()) 里"我干哪些 tile"是编译期算出来的, 运行时零开销。
    晚绑定是运行时从一个共享游标里抢活 —— 真实 kernel 得做一次原子加 (或一次核间
    同步标志的读改写)。**模型里不计这笔, 晚绑定就永远显得更好**: 它只拿到了收益
    (就绪的活能漂到空闲核), 没付代价。那是模型的结构偏置, 不是结论。

    这个值没有标定
    --------------
    PrimitiveCosts.late_bind_fetch_us 缺省 0.0, 出处 assumed —— **0 不表示"没有代价",
    表示"本模型没有声称代价是多少"**。所以缺省下换晚绑定的那个收益仍是不可信的,
    analysis/design_space.py 会把这类行标出来。要定这个值, 见
    docs/calibration_runs.md 的 R7 (同一形状跑静态分核与动态取活两版, 差分)。
    """
    fetch = float(getattr(costs, "late_bind_fetch_us", 0.0) or 0.0)
    if fetch <= 0.0:
        return
    pooled = set(late_pools)
    for ev in events:
        for r in ev.resources:
            role, _, core_s = r.partition(":")
            if core_s and role in pooled:
                ev.duration_us += fetch
                break
