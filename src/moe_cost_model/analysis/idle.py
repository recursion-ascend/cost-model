"""第 6 层: AIC/AIV 空闲分解 — 区分"DAG 逼出来的"与"真浪费".

为什么要分开
------------
"核不能有空闲"作为绝对约束在逻辑上不可能: 事件必须等前置完成才能开始, 而 t=0 时刻
GMM1 还在等 dispatch_ready (AIV1 产出), 所以开头那段**所有 AIC 必须空着**。这是依赖
约束直接推出来的, 不是模型缺陷。

能成立的不变量只有一个 —— **work-conserving**: 核不得在"存在已就绪的活"时空闲。
于是空闲要分成两类:

  forced    此刻该资源池里**没有**已就绪未开始的事件 → 空闲无法消除, 只能靠改 DAG
            结构 (加深流水、改波的组成) 来减少
  avoidable 此刻**有**已就绪未开始的事件, 却还有核空着 → work-conservation 违规,
            换 tile→核 的绑定方式 (如派发时刻晚绑定) 就能回收

"已就绪"必须是"真能动", 不只是"依赖齐了"
--------------------------------------
一个事件的前置虽然跑完了, 它仍可能被别的物理约束挡住 —— 那种等待和"依赖未完成"同类,
不该算成 work-conservation 违规。调度器的准入有四关, 这里逐一对应:

  ① 依赖齐备          dependency_ready_us
  ②  自己的核空闲      ← **这一关就是本模块要抓的**: 核空着却因为绑定挪不过去
  ③ 计数信号量够      ScheduledEvent.actionable_us (如 UB:gmm1act 的槽、QUEUE:mte_aic 的 L1 槽)
  ⑦ 非核独占资源空闲   如 DISPATCH_COMM (跨卡通道一次只许一个核用)

所以"就绪未开始"的窗口起点取 **actionable_us** = ①③⑦ 三者中最晚的那个, 而不是 ①。
不这么算的话, 等 UB 槽、等跨卡通道都会被报成"有活不干" —— 实测 serialize_dispatch_comm
下虚报 5814.8 核·us, UB 深度改成容量后又虚报 262.8 核·us。

(信道那一关 ④ 已随信道模型于 2026-10-03 一起去掉。)

avoidable 仍是**上界**
----------------------
①③ 由引擎精确给出 (actionable_us), ⑦ 从排好的时间线精确反推。但有**两种"核挪不动"
的物理约束没有建模进来**, 它们都会让这里虚报:

  共位约束      GMM1 落核 X 时 ACT 必须落 AIV0:X, 而 AIV0:X 可能正忙。
  相位组绑定    开了相位流水 (ModelOptions.pipeline) 之后, 一个 tile 被拆成
                .ld / .cb / .fix 几个相位, 它们共用一个 core_group: load 相位把 A/B
                搬进**某个核的 L1**, cube 相位只能在那个核上算 —— 数据在那儿。所以
                cube 相位即使"前置齐备且别处有空闲核"也挪不过去。

所以真实可回收量 <= avoidable_idle_us。实测 (2026-10-04, examples/scenario_basic.toml):
开了晚绑定之后所有配置的 avoidable 都是 0, **只有相位流水那一档剩 1637.5 核·us** ——
逐段挖进去看, 等着的全是 .cb 相位, 且每一个都在它那个核空出来的**同一时刻**就开跑
(例: W2.E8.S0.gmm1.m0.n0.cb 绑在 AIC:12, AIC:12 的上一个活跑到 168.97, 它就在
168.97 开始), 同时另有 16 个核空着。那不是调度没做到位, 是相位组绑定使然。

→ **判读规则**: 静态钉核 (late_bind_pools=()) 下的 avoidable 是真帐, 换晚绑定能回收;
  相位流水下的 avoidable 含相位组那部分虚报, 不能直接当成违规。要把它收紧, 得把
  core_group 透进本模块, 把"组已绑定到忙核"的事件从 ready 里剔掉。
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: 按核的信号量 token 后缀 ("R0.UB:gmm1act:c7" 的 "c7")
_TOK_CORE = re.compile(r"c\d+$")

#: 时间比较的容差. 调度器的时刻是浮点, 相邻事件的 end 与下一个 start 常常只差 ulp。
_EPS = 1e-9


@dataclass(frozen=True)
class IdleSegment:
    """一段 avoidable 空闲: [t_begin, t_end) 内有核空着而有活已就绪."""
    t_begin: float
    t_end: float
    idle_resources: Tuple[str, ...]
    waiting_ready: Tuple[str, ...]      # 已就绪但还没开始的事件名 (最多留几条做样本)

    @property
    def core_us(self) -> float:
        return len(self.idle_resources) * (self.t_end - self.t_begin)


@dataclass(frozen=True)
class IdleReport:
    """一个资源池 (如某个 rank 的全部 AIC) 的空闲分解."""
    pool: Tuple[str, ...]
    horizon_us: float                   # 统计区间 [0, horizon)
    capacity_us: float                  # len(pool) * horizon, 单位 核·us
    busy_us: float
    forced_idle_us: float
    avoidable_idle_us: float
    segments: Tuple[IdleSegment, ...]   # 只收 avoidable 的段

    @property
    def idle_us(self) -> float:
        return self.forced_idle_us + self.avoidable_idle_us

    @property
    def utilization(self) -> float:
        return self.busy_us / self.capacity_us if self.capacity_us > 0 else 0.0

    @property
    def work_conserving(self) -> bool:
        """work-conserving 不变量: 没有一寸空闲是"有活却空着"."""
        return self.avoidable_idle_us <= _EPS


def _split_rank(resource: str) -> Tuple[str, str]:
    if "." in resource:
        head, tail = resource.split(".", 1)
        return head, tail
    return "", resource


def _free_at_or_after(ivs: Sequence[Tuple[float, float]], t0: float,
                      own: Tuple[float, float]) -> float:
    """>= t0 的最早时刻, 该资源不被**别人**占用 (排除本事件自己的那一段)."""
    t = t0
    for s, e in ivs:                      # ivs 按 s 升序
        if (s, e) == own:
            continue
        if s - _EPS <= t < e - _EPS:
            t = e
    return t


def _actionable_us(e, pool_res: str, res_busy: Dict[str, List[Tuple[float, float]]],
                   resource_prefix: str) -> float:
    """事件"真能动"的最早时刻: 依赖齐 + 信号量可准入 + 非核独占资源空出.

    不含"自己的核空出来"—— 那一关正是本模块要抓的违规。
    """
    # ①③ 由引擎给出: actionable_us = 从依赖齐备起算的容量可行点。
    #     不能用 capacity_wait_us 加在 dependency_ready_us 上 —— 它是从探测起点
    #     (已含"等自己的核") 算的, 那样会严重低估 (实测 20.7 vs 真值 71.1)。
    t = max(e.dependency_ready_us, getattr(e, "actionable_us", 0.0) or 0.0)
    # ⑦ 非核独占资源 (DISPATCH_COMM 等): 精确算
    own = (e.start_us, e.end_us)
    for r in e.resources:
        if r == pool_res or _split_rank(r)[1].startswith(resource_prefix):
            continue
        t = max(t, _free_at_or_after(res_busy.get(r, ()), t, own))
    # 不可能晚于它实际开始的时刻
    return min(t, e.start_us)


class _SemLedger:
    """从排好的时间线反推每个计数信号量 token 的在途量.

    一个 token 可以**跨事件**持有 (UB:gmm1act 由 GMM1 取、配对 ACT 还), 所以必须按
    "谁在何时取、谁在何时还"重建, 不能按单个事件的区间算。
    """

    def __init__(self, scheduled: Sequence):
        delta: Dict[str, Dict[float, int]] = {}
        for e in scheduled:
            for tok, k in getattr(e, "acquires", ()):
                delta.setdefault(tok, {})[e.start_us] = \
                    delta.setdefault(tok, {}).get(e.start_us, 0) + k
            for tok, k in getattr(e, "releases", ()):
                delta.setdefault(tok, {})[e.end_us] = \
                    delta.setdefault(tok, {}).get(e.end_us, 0) - k
        self._t: Dict[str, List[float]] = {}
        self._c: Dict[str, List[int]] = {}
        for tok, d in delta.items():
            ts = sorted(d)
            run, acc = 0, []
            for t in ts:
                run += d[t]
                acc.append(run)
            self._t[tok], self._c[tok] = ts, acc

    def outstanding(self, tok: str, t: float) -> int:
        ts = self._t.get(tok)
        if not ts:
            return 0
        i = bisect_right(ts, t + _EPS) - 1
        return self._c[tok][i] if i >= 0 else 0


def _fits_on(e, core_res: str, t: float, sem: "_SemLedger",
             capacities: Mapping[str, int]) -> bool:
    """把 e 放到 core_res 这个核上, t 时刻它的按核信号量还有余量吗.

    这是"空闲核能不能接这个活"的判据。按核的 token 要先换成**那个核**的名字:
    一个核空着不等于它的 UB 槽空着 —— GMM1 跑完、配对 ACT 还在读 UB 时, 核空着
    但槽没还, 新的 GMM1 落不进来。
    """
    # 资源名是 "R0.AIC:27" (冒号分核号), 按核 token 是 "R0.UB:gmm1act:c27" (c 前缀)
    num = core_res.rsplit(":", 1)[-1]
    if not num.isdigit():
        return True
    suffix = "c" + num
    for tok, k in getattr(e, "acquires", ()):
        tc = _TOK_CORE.search(tok)
        probe = (tok[: tc.start()] + suffix) if tc else tok
        cap = capacities.get(probe)
        if cap is None:
            continue                      # 没声明容量 = 不构成约束
        if sem.outstanding(probe, t) + k > cap:
            return False
    return True


def idle_decomposition(scheduled: Sequence, resource_prefix: str = "AIC:",
                       horizon_us: Optional[float] = None,
                       max_segments: int = 64,
                       max_waiting_sample: int = 4,
                       capacities: Optional[Mapping[str, int]] = None) -> Dict[str, IdleReport]:
    """按 rank 分组给出资源池的空闲分解.

    scheduled:        ScheduledEvent 序列 (rank_results[r]["events"])
    resource_prefix:  池的资源前缀, "AIC:" / "AIV0:" / "AIV1:"
    horizon_us:       统计区间上界; 缺省取**传入全部事件**的最大 end_us (≈ dag_end)
    max_segments:     最多收集几段 avoidable (只影响 segments, 不影响统计量)
    capacities:       计数信号量容量表 (simulate 传入)。给了才能判"**这个**空闲核
                      的槽还有没有余量" —— 不给则退化成"存在某处可准入"的上界。
    """
    # horizon 缺省取**传入的全部事件**的最大 end_us (≈ dag_end), 不是池内事件的 ——
    # 尾段 (epilogue/unpermute 在 AIV 上) 期间 AIC 确实没活干, 那段空闲属于 forced,
    # 工程上要算进核利用率里。只按池内事件取 horizon 会把尾段整段藏掉。
    all_end = max((e.end_us for e in scheduled if getattr(e, "end_us", None) is not None),
                  default=0.0)
    # 每个资源的占用区间 (含非核资源如 DISPATCH_COMM), 供 actionable 计算 ⑦
    res_busy: Dict[str, List[Tuple[float, float]]] = {}
    for e in scheduled:
        if e.end_us - e.start_us <= _EPS:
            continue
        for r in e.resources:
            res_busy.setdefault(r, []).append((e.start_us, e.end_us))
    for ivs in res_busy.values():
        ivs.sort()
    sem = _SemLedger(scheduled) if capacities else None
    pools: Dict[str, List[str]] = {}
    members: Dict[str, List] = {}       # rank -> 该池的事件
    for e in scheduled:
        if not e.resources:
            continue
        # 池事件只占一个池资源; 真有多占的按首个算。
        # (这里原先举例说 moe_stage_done 同时占 AIC 与 AIV —— 那是旧行为; 现在排空栅栏与
        #  barriers 都是 resources=() 的零时长事件, 上面那个 `if not e.resources` 就跳过了。)
        hit = [r for r in e.resources if _split_rank(r)[1].startswith(resource_prefix)]
        if not hit:
            continue
        rank = _split_rank(hit[0])[0]
        pools.setdefault(rank, [])
        members.setdefault(rank, []).append((e, hit[0]))
        for r in hit:
            if r not in pools[rank]:
                pools[rank].append(r)

    out: Dict[str, IdleReport] = {}
    for rank, pool in pools.items():
        pool_sorted = tuple(sorted(pool))
        evs = members[rank]
        horizon = float(horizon_us) if horizon_us is not None else all_end
        if horizon <= 0:
            out[rank] = IdleReport(pool_sorted, 0.0, 0.0, 0.0, 0.0, 0.0, ())
            continue

        # actionable: 依赖齐 + 信号量可准入 + 非核独占资源空出, 三者取最晚
        actionable: Dict[str, float] = {
            e.name: _actionable_us(e, hit_res, res_busy, resource_prefix)
            for e, hit_res in evs}

        # 时间断点: 事件起止 + actionable 时刻 (它会改变"有无能动的活"的真值)
        pts = {0.0, horizon}
        for e, _ in evs:
            pts.update((e.start_us, e.end_us, actionable[e.name]))
        marks = sorted(t for t in pts if 0.0 <= t <= horizon)

        # 事件在 [start, end) 占资源; 在 [dependency_ready, start) 处于"就绪未开始"
        starts: Dict[float, List] = {}
        ends: Dict[float, List] = {}
        ready_delta: Dict[float, int] = {}
        for e, res in evs:
            # 零时长事件 (dispatch_ready / moe_stage_done 的 duration 都是 0) 不进占用
            # 统计: 它们占的是测度零的时间, 对 busy/idle 的积分无贡献。而且同一时刻
            # "先处理 end 再处理 start"的次序会让它们入集后永不移除 —— 实测把 AIC busy
            # 从 5455.0 虚报到 6438.7。
            if e.end_us - e.start_us <= _EPS:
                continue
            starts.setdefault(e.start_us, []).append((e, res))
            ends.setdefault(e.end_us, []).append((e, res))
            a = actionable[e.name]
            if e.start_us - a > _EPS:
                ready_delta[a] = ready_delta.get(a, 0) + 1
                ready_delta[e.start_us] = ready_delta.get(e.start_us, 0) - 1

        busy_res: set = set()
        ready_cnt = 0
        busy_us = forced = avoidable = 0.0
        segs: List[IdleSegment] = []
        for i, t in enumerate(marks[:-1]):
            # 先处理在 t 结束的 (资源即时释放), 再处理在 t 开始的
            for e, res in ends.get(t, ()):
                busy_res.discard(res)
            for e, res in starts.get(t, ()):
                busy_res.add(res)
            ready_cnt += ready_delta.get(t, 0)

            t_next = marks[i + 1]
            dt = t_next - t
            if dt <= _EPS:
                continue
            busy_us += len(busy_res) * dt
            idle = [r for r in pool_sorted if r not in busy_res]
            if not idle:
                continue
            core_us = len(idle) * dt
            placeable = ()
            if ready_cnt > 0:
                pend = [e for e, _ in evs
                        if actionable[e.name] <= t + _EPS < e.start_us]
                if sem is None:
                    placeable = tuple(e.name for e in pend)
                else:
                    # 逐个空闲核问: 它的按核槽还容得下这个活吗? 一个都放不下 -> forced
                    placeable = tuple(
                        e.name for e in pend
                        if any(_fits_on(e, r, t, sem, capacities) for r in idle))
            if placeable:
                avoidable += core_us
                if len(segs) < max_segments:
                    segs.append(IdleSegment(t, t_next, tuple(idle),
                                            placeable[:max_waiting_sample]))
            else:
                forced += core_us

        out[rank] = IdleReport(
            pool=pool_sorted, horizon_us=horizon,
            capacity_us=len(pool_sorted) * horizon,
            busy_us=busy_us, forced_idle_us=forced, avoidable_idle_us=avoidable,
            segments=tuple(segs))
    return out

class WorkConservationViolation(RuntimeError):
    """调度器在"有就绪的活"时让一个**池化**的核空着 —— 不变量没守住.

    与下界 (BoundViolation) 对举: 下界问"物理上可能吗", 这一条问"这个调度自己有没有
    浪费"。两者都在模型层抛, 不在入口层 —— 护栏不能被绕过。

    只看派发时刻绑定 (ModelOptions.late_bind_pools) 的那些角色池。静态钉核的池不在
    其列: 工作钉死在某个核上, 那个核忙而别处空着时搬不过去, 那是**那种分核方式的
    代价** (量出来就是结论), 不是调度器没做到位。
    """

    def __init__(self, detail: Mapping[str, float], segments=()):
        self.detail = dict(detail)
        self.segments = tuple(segments)
        worst = ", ".join(f"{k} {v:.1f} 核·us" for k, v in sorted(self.detail.items()))
        sample = "; ".join(
            f"[{sg.t_begin:.2f},{sg.t_end:.2f}) 空闲 {len(sg.idle_resources)} 个核, "
            f"已就绪: {', '.join(sg.waiting_ready[:2])}"
            for sg in self.segments[:3])
        super().__init__(
            f"work-conservation 不变量被打破: {worst}"
            + (f"\n  前几段: {sample}" if sample else ""))


def work_conservation_violations(reports: Mapping[str, object],
                                 late_bind_pools: Sequence[str],
                                 *, tol: float = 1e-6) -> Dict[str, float]:
    """哪些**池化**角色违了不变量: {角色名: 可避免空闲 核·us}.

    reports: rank_result["idle_decomposition"], 键形如 "R0.AIC" / "AIC"。
    late_bind_pools: 这次运行哪些角色池是派发时刻绑定的。空 = 全静态钉核 = 不检查。

    共位是硬件强制的那一对 (GMM1 -> ACT 必须同核, L0C->UB 的 Fixpipe 只在绑定对内)
    不随 AIC 入池而自由: 所以 "AIC" 入池隐含 AIV0 随动, AIV0 自身也按池化看待。
    """
    pools = set(late_bind_pools or ())
    if not pools:
        return {}
    if "AIC" in pools:
        pools.add("AIV0")                  # 共位随动: ACT 跟着它的 GMM1 漂
    out: Dict[str, float] = {}
    for key, rep in (reports or {}).items():
        role = str(key).rsplit(".", 1)[-1]
        if role not in pools:
            continue
        got = float(getattr(rep, "avoidable_idle_us", 0.0) or 0.0)
        if got > tol:
            out[role] = got
    return out
