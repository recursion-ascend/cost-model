"""第 6 层: 下界 — 墙钟在物理上不可能低于这些值.

为什么要这一层
--------------
排程给出的是"这样编排会跑多久"。但一个只会推演的模型**无法证伪自己**: 它吐出的数
不管对错都长得一样。下界把算法、物理、硬件三类事实变成可检查的断言:

  算法事实   MegaMoE 要算多少次乘加、要搬多少字节 —— 只由形状 (每专家行数、h、
             hidden_dim、topk) 决定, 与 tile 怎么切、活怎么分到核上**无关**。
  物理事实   一次乘加要占 Cube 一拍; 一个字节要占带宽一次; 一条依赖链上的事情
             只能一件接一件。并行度再高也不能让这三件事变少。
  硬件事实   每核 Cube 速率、每核载入带宽、聚合 HBM 带宽、可用核数 (规格值,
             见 config/platform.py 与 config/hardware.py)。

于是:

    墙钟 >= max(算力下界, 带宽下界, 依赖下界)

**违反这个不等式的墙钟一定是模型漏算了什么**, 不是优化。2026-10-04 用它抓到的第一个
实例: 开相位流水之后载入相位不占任何资源 (信道模型 2026-10-03 停用后 channel_bytes
只做申报、不参与准入), 于是 512 个载入可以无限并行, 墙钟 1221.80us 低于自己的载入
带宽下界 1697.12us **28%** —— 之前被当成"相位流水省了 30%"。

三个下界各自的口径
------------------
算力下界  总乘加 / (可用核数 x 每核 Cube 速率)。假设完美均衡、零空闲。
          GMM1: 每专家 m_e x h x hidden_dim 次乘加 (hidden_dim = 2I, SwiGLU 的
                gate 与 up 两块投影都要算)。
          GMM2: 每专家 m_e x I x h, I = hidden_dim / activation_n_half。
          两者都是 A(m x K) x B(K x N) 的定义值, 不含任何 tile 口径。

带宽下界  必须跨 GM->L1 的字节 / 带宽。每个字节**至少搬一次**是下界;
          L2 命中、B 流复用只能让实际更接近下界, 不能低于它。
          取 min(每核带宽 x 可用核数, 聚合 HBM 带宽) 作为可用速率 —— 两个都是硬件
          规格, 谁小谁管。

依赖下界  一个 token 必须依次经过 dispatch -> GMM1 -> ACT -> GMM2 -> COMBINE,
          这条链上的事情不能并行。取各 stage 的**最小一份**工作时长之和。
          这是弱下界 (真实链更长), 但它是硬的。

**下界不是预测**: 它不含任何编排信息, 所以不会因为换编排而变 (除了换 dtype/算法)。
墙钟与下界的差额就是编排带来的那部分, 它的构成由 analysis/idle.py 的
busy / forced_idle / avoidable_idle 给出 (三者精确闭合到 horizon x 核数)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

#: 判定"墙钟低于下界"的相对容差. 浮点累加与 tile 边界取整会带来 ulp 级误差。
TOL = 1e-6


@dataclass(frozen=True)
class WorkloadFacts:
    """算法事实: 只由形状决定, 与 tile 切法、分核方式、编排全部无关.

    乘加数按 A(m x K) x B(K x N) 的定义算; 字节数按"每个元素至少跨 GM->L1 一次"算。
    """

    gmm1_mac: int                  #: 所有专家的 GMM1 乘加 (含 SwiGLU 的 gate+up)
    gmm2_mac: int                  #: 所有专家的 GMM2 乘加
    gmm1_a_bytes: int              #: GMM1 读激活 (dispatch 落 GM 的量化激活)
    gmm1_b_bytes: int              #: GMM1 读权重 (每专家一份, 至少一次)
    gmm2_a_bytes: int              #: GMM2 读激活 (物化编排下存在)
    gmm2_b_bytes: int              #: GMM2 读权重
    rows_total: int                #: 本卡所有专家收到的行数之和
    experts_active: int            #: 行数 > 0 的专家数

    @property
    def total_mac(self) -> int:
        return self.gmm1_mac + self.gmm2_mac

    @property
    def gm_to_l1_bytes(self) -> int:
        return (self.gmm1_a_bytes + self.gmm1_b_bytes
                + self.gmm2_a_bytes + self.gmm2_b_bytes)


def workload_facts(shape, *, kernel=None, gmm2_a_from_gm: bool = True,
                   act_bytes_per_elem: float = 1.0,
                   weight_bytes_per_elem: float = 1.0) -> WorkloadFacts:
    """从形状算出算法事实.

    act_bytes_per_elem / weight_bytes_per_elem: 数据格式 (A8W8 两者都是 1B/元素)。
    gmm2_a_from_gm: ACT 的输出是否物化到 GM 再被 GMM2 读回 (编排选择, 见 StageLink)。
    """
    km = kernel if kernel is not None else getattr(shape, "kernel", None)
    half = int(getattr(km, "activation_n_half", 2) or 2)
    h, hidden = int(shape.h), int(shape.hidden_dim)
    inter = hidden // half                      # I = 单块投影宽度
    rows = [int(x) for x in shape.expert_tokens]
    m_total = sum(rows)
    n_active = sum(1 for x in rows if x > 0)
    return WorkloadFacts(
        # A(m x h) x B(h x hidden): hidden 已含 gate+up 两块, 所以不再乘 2
        gmm1_mac=m_total * h * hidden,
        # A(m x I) x B(I x h)
        gmm2_mac=m_total * inter * h,
        gmm1_a_bytes=int(m_total * h * act_bytes_per_elem),
        gmm1_b_bytes=int(n_active * h * hidden * weight_bytes_per_elem),
        gmm2_a_bytes=int(m_total * inter * act_bytes_per_elem) if gmm2_a_from_gm else 0,
        gmm2_b_bytes=int(n_active * inter * h * weight_bytes_per_elem),
        rows_total=m_total,
        experts_active=n_active,
    )


@dataclass(frozen=True)
class Bounds:
    """三个下界与谁绑定. 全部单位 us."""

    compute_us: float
    bandwidth_us: float
    dependency_us: float
    facts: WorkloadFacts
    #: 带宽下界用的可用速率 (B/us) 与它是被谁限住的 ("per_core" / "aggregate")
    bandwidth_rate: float = 0.0
    bandwidth_limited_by: str = ""

    @property
    def lower_us(self) -> float:
        return max(self.compute_us, self.bandwidth_us, self.dependency_us)

    @property
    def binding(self) -> str:
        """谁决定下界. 三者取最大, 并列时按 compute > bandwidth > dependency 报."""
        best = self.lower_us
        for name, v in (("compute", self.compute_us),
                        ("bandwidth", self.bandwidth_us),
                        ("dependency", self.dependency_us)):
            if v >= best - TOL * max(1.0, best):
                return name
        # 构造上不可达: best = max(三者), 所以必有一个满足上面的比较。
        # 留着是为了在"三者之一变成 NaN"这种被破坏的状态下有个确定答案而不是
        # 隐式返回 None。覆盖率会把它报成未覆盖行, 那是对的。
        raise AssertionError(  # pragma: no cover
            f"lower_us={best} 不等于三个下界中的任何一个 (NaN?): "
            f"{self.compute_us} / {self.bandwidth_us} / {self.dependency_us}")

    def as_dict(self) -> Dict[str, object]:
        return {
            "compute_us": self.compute_us,
            "bandwidth_us": self.bandwidth_us,
            "dependency_us": self.dependency_us,
            "lower_us": self.lower_us,
            "binding": self.binding,
            "bandwidth_rate_bytes_per_us": self.bandwidth_rate,
            "bandwidth_limited_by": self.bandwidth_limited_by,
            "total_mac": self.facts.total_mac,
            "gm_to_l1_bytes": self.facts.gm_to_l1_bytes,
        }


def compute_bound_us(facts: WorkloadFacts, *, cube_mac_per_us: float,
                     active_cores: int) -> float:
    """总乘加 / (核数 x 每核 Cube 速率). cube_mac_per_us <= 0 时返回 0 (未标定)."""
    if cube_mac_per_us <= 0 or active_cores <= 0:
        return 0.0
    return facts.total_mac / (cube_mac_per_us * active_cores)


def bandwidth_bound_us(facts: WorkloadFacts, *, bw_per_core_bytes_per_us: float,
                       active_cores: int,
                       aggregate_bytes_per_us: Optional[float] = None
                       ) -> Tuple[float, float, str]:
    """必须搬的字节 / 可用速率. 返回 (下界, 用的速率, 被谁限住).

    可用速率 = min(每核带宽 x 核数, 聚合带宽) —— 两个都是硬件规格, 谁小谁管。
    """
    if bw_per_core_bytes_per_us <= 0 or active_cores <= 0:
        return 0.0, 0.0, ""
    per_core_total = bw_per_core_bytes_per_us * active_cores
    rate, who = per_core_total, "per_core"
    if aggregate_bytes_per_us and aggregate_bytes_per_us < per_core_total:
        rate, who = float(aggregate_bytes_per_us), "aggregate"
    return facts.gm_to_l1_bytes / rate, rate, who


def dependency_bound_us(stage_min_us: Mapping[str, float]) -> float:
    """一个 token 必经链上各 stage 的最小一份工作时长之和 (弱但硬的下界)."""
    return float(sum(v for v in stage_min_us.values() if v > 0))


class BoundViolation(AssertionError):
    """墙钟低于物理下界 —— 模型漏算了某项代价, 不是优化."""


def check_wall_clock(bounds: Bounds, wall_us: float, *, where: str = "",
                     tol: float = TOL) -> None:
    """墙钟必须 >= max(三个下界); 否则抛 BoundViolation 并指出是哪个界被穿透.

    这不是品味问题: 下界只用算法/物理/硬件事实, 不含编排。穿透它只能是模型少算了
    某项资源占用 (典型: 某个相位不占任何资源)。
    """
    low = bounds.lower_us
    if low <= 0 or wall_us >= low * (1.0 - tol):
        return
    deficit = (low - wall_us) / low * 100.0
    raise BoundViolation(
        f"{where or '墙钟'} {wall_us:.3f}us 低于物理下界 {low:.3f}us ({deficit:.1f}%), "
        f"被穿透的是 {bounds.binding} 界 "
        f"(算力 {bounds.compute_us:.1f} / 带宽 {bounds.bandwidth_us:.1f} / "
        f"依赖 {bounds.dependency_us:.1f})。"
        f"下界只用算法+物理+硬件事实, 不含编排 —— 穿透它说明某项资源占用没被计入, "
        f"典型是某个相位不占任何资源。")


def idle_split_us(idle_report, active_cores: int) -> Dict[str, float]:
    """把 analysis/idle.py 的核·us 分解折算成 us (除以核数), 便于与下界对齐.

    恒等式 (精确闭合): horizon x 核数 = busy + forced_idle + avoidable_idle,
    所以 busy/核数 + (forced+avoidable)/核数 = horizon。
    """
    if idle_report is None or active_cores <= 0:
        return {}
    n = float(active_cores)
    return {
        "busy_us": idle_report.busy_us / n,
        "forced_idle_us": idle_report.forced_idle_us / n,
        "avoidable_idle_us": idle_report.avoidable_idle_us / n,
        "horizon_us": idle_report.horizon_us,
    }


def attach_bounds(shape, rank_result, *, costs, kernel=None, platform=None,
                  active_cores=None, check: bool = True):
    """给一个 rank 算三个下界并断言墙钟没穿透它们.

    放在这一层 (而不是 api) 是因为**护栏不能被绕过**: 2026-10-05 发现
    tests/golden_cases.run_shapes 直达 A8W8WaveCostModel.simulate_multi 并手工拼结果,
    于是四个 golden case 完全没跑下界断言。任何入口 (api / model / 手工) 都该挂上。

    算法事实 (乘加数、必搬字节) 只由形状决定; 速率取硬件规格。所以同一形状换任何
    编排, 下界都不变 —— 它是用来**检查**编排结果的尺子, 不是预测。
    """
    link = None
    opts = getattr(shape, "options", None)
    if opts is not None:
        try:
            link = opts.link("activation", "gmm2")
        except Exception:
            link = None
    if active_cores is None:
        active_cores = int(getattr(shape, "aic_num", 0) or 0)
    if kernel is None:
        kernel = getattr(shape, "kernel", None)
    facts = workload_facts(
        shape, kernel=kernel,
        gmm2_a_from_gm=True if link is None else bool(link.materialised))
    cube = _cube_rate_of(costs)
    bw = _load_bw_of(costs)
    agg = getattr(platform, "hbm_bytes_per_us", None) if platform is not None else None
    bw_us, rate, who = bandwidth_bound_us(
        facts, bw_per_core_bytes_per_us=bw, active_cores=active_cores,
        aggregate_bytes_per_us=agg)
    b = Bounds(
        compute_us=compute_bound_us(facts, cube_mac_per_us=cube,
                                    active_cores=active_cores),
        bandwidth_us=bw_us,
        dependency_us=_dependency_bound_of(rank_result),
        facts=facts, bandwidth_rate=rate, bandwidth_limited_by=who)
    out = b.as_dict()
    out["violation"] = None
    if check:
        try:
            check_wall_clock(b, float(rank_result["total_us"]),
                             where=f"rank{getattr(shape, 'rank_id', '?')} 墙钟")
        except BoundViolation as exc:
            # 记进结果再抛: 调用方给 check_bounds=False 时能拿到同一条诊断
            out["violation"] = str(exc)
            raise
    else:
        low = b.lower_us
        wall = float(rank_result["total_us"])
        if low > 0 and wall < low * (1.0 - 1e-6):
            out["violation"] = (
                f"墙钟 {wall:.3f}us 低于 {b.binding} 下界 {low:.3f}us "
                f"({(low - wall) / low * 100:.1f}%)")
    out["idle_split_us"] = idle_split_us(
        (rank_result.get("idle_decomposition") or {}).get(
            f"R{getattr(shape, 'rank_id', 0)}.AIC"), active_cores)
    return out


def _cube_rate_of(costs) -> float:
    """从代价对象上取每核 Cube 速率 (MAC/us); 自定义 callable 取不到则返回 0."""
    owner = getattr(getattr(costs, "gmm1_tile", None), "__self__", None)
    return float(getattr(owner, "cube_rate", 0.0) or 0.0)


def _load_bw_of(costs) -> float:
    """从代价对象上取每核 GM->L1 载入带宽 (B/us)."""
    owner = getattr(getattr(costs, "gmm1_tile", None), "__self__", None)
    return float(getattr(owner, "bw", 0.0) or 0.0)


def _dependency_bound_of(rank_result) -> float:
    """一个 token 必经链 dispatch->GMM1->ACT->GMM2->COMBINE 上各 stage 的最小一份.

    从已排出的事件里取每个 stage 的最短事件时长 —— 那就是"这个 stage 最小一份工作"
    的时长。链上的事不能并行, 所以它们的和是硬下界 (弱, 但不会错)。
    """
    CHAIN = ("dispatch", "gmm1", "activation", "gmm2", "combine")
    best = {}
    for e in rank_result.get("events", ()):
        st = (e.meta or {}).get("stage")
        if st in CHAIN:
            d = e.end_us - e.start_us
            if d > 0 and (st not in best or d < best[st]):
                best[st] = d
    return dependency_bound_us(best)
