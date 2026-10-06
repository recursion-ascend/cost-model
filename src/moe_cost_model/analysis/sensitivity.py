"""第 6 层: 未标定输入 -> 结论的区间. 不能判定的比较要**说出来**, 不要报点值.

为什么要这一层
--------------
模型吐出 "换这个旋钮省 0.57%" 这样的点值, 看起来像个结论。但它下面垫着几个**没有
标定**的输入: 跨卡写带宽只定到量级 (4.5~9.5 GB/s 取 8.6)、晚绑定的取活开销干脆没有
数。一个 0.57% 的差, 在这些输入的合理范围内很可能变号 —— 那它就不是结论, 是噪声。

所以比较要带区间: Δ = -0.57% [-1.2%, +0.3%]。区间跨 0 = **不可判定**, 别拿去做决策。

两类输入要分开
--------------
有区间 (ranged)   实测给出了范围, 只是没定到一个点。可以传播成 Δ 的区间。
无区间 (unknown)  连范围都没有 (比如晚绑定取活开销, 它甚至没被测过)。
                  **不能**编一个范围传播 —— 那是把无知包装成精度。
                  只能把"这个结论依赖某个没测过的量"如实标出来。

口径
----
一次变一个 (one-at-a-time): 每个有区间的输入分别取两端重跑, Δ 的区间取这些结果的
min/max。这是局部敏感度, 不覆盖输入之间的交互 —— 交互要全因子扫, 代价是 2^n 次。
对"这个结论稳不稳"这个问题, OAT 已经够用: 只要有一个输入能让它变号, 它就不稳。
"""
from __future__ import annotations

from ..config.hardware import BW_L1_GM, BW_REMOTE_WRITE

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Ranged:
    """有实测区间、但没定到一个点的输入."""

    name: str                   #: 场景文件里的点分路径 (如 "calibration.bw_combine_remote")
    nominal: float
    low: float
    high: float
    source: str                 #: 区间是怎么来的 —— 必须写, 否则无从判断可不可信

    def __post_init__(self) -> None:
        if not (self.low <= self.nominal <= self.high):
            raise ValueError(
                f"{self.name}: nominal {self.nominal} 不在 [{self.low}, {self.high}] 内")
        if not self.source:
            raise ValueError(f"{self.name}: 区间必须写出处")

    @property
    def endpoints(self) -> Tuple[float, float]:
        return (self.low, self.high)


@dataclass(frozen=True)
class Unknown:
    """连范围都没有的输入. 不参与区间传播, 只做"这个结论依赖它"的标注."""

    name: str
    reason: str                 #: 为什么没有范围 (没测过 / 机制未定 / ...)
    calibration: str = ""       #: 要定它得做哪个 run (docs/calibration_runs.md 的编号)


#: 本模型当前**没有标定到点**的输入. 加一条就要写出处或原因 —— 这张表就是
#: "模型自己知道自己不知道什么"。
UNCERTAIN_INPUTS: Tuple[object, ...] = (
    Ranged(
        name="calibration.bw_combine_remote",
        nominal=float(BW_REMOTE_WRITE), low=4500.0, high=9500.0,
        source=("20260930 三个 noshared run 的最快 COMBINE tile 反扣: "
                "bs8192 9.5 / bs36 7.8 / bs128 4.5 GB/s per core "
                "(bs128 全程被挤, 是下界)。见 config/hardware.BW_REMOTE_WRITE"),
    ),
    Ranged(
        name="calibration.bw_l1_gm",
        nominal=float(BW_L1_GM), low=49200.0, high=54400.0,
        source=("同一批 run 按 A 流 / B 流分别反解: A 49.2, B 54.4 GB/s; "
                "标定值 51.9 取中。随并发核数变, 待按并发分档 (R2)"),
    ),
    Unknown(
        name="calibration.late_bind_fetch_us",
        reason=("晚绑定每取一次活的开销 (原子加 / 核间同步标志的读改写) **从未测过**。"
                "缺省 0 不表示没有代价, 表示本模型没有声称它是多少 —— 所以任何"
                "'晚绑定更快'的结论都依赖这个没测过的量"),
        calibration="R7",
    ),
    Unknown(
        name="calibration.cube_mac_per_us",
        reason=("Cube 速率给的是规格峰值 (fp8 1.35e7 MAC/us/核), 真实可达效率没测过。"
                "计算绑定的形状上结论会随它变"),
        calibration="R1",
    ),
)

RANGED = tuple(u for u in UNCERTAIN_INPUTS if isinstance(u, Ranged))
UNKNOWNS = tuple(u for u in UNCERTAIN_INPUTS if isinstance(u, Unknown))


@dataclass(frozen=True)
class Interval:
    """一个量的点值与区间. unknown_inputs 非空时区间**不完整**."""

    nominal: float
    low: float
    high: float
    #: 哪个输入把区间撑到这么宽 (贡献最大的那个)
    driver: str = ""
    #: 这个结论还依赖哪些**连范围都没有**的输入 —— 有的话区间不完整
    unknown_inputs: Tuple[str, ...] = ()

    @property
    def straddles_zero(self) -> bool:
        return self.low <= 0.0 <= self.high

    @property
    def decidable(self) -> bool:
        """符号确定, 且不依赖任何没范围的输入."""
        return not self.straddles_zero and not self.unknown_inputs

    def format(self, unit: str = "%") -> str:
        s = f"{self.nominal:+.2f}{unit} [{self.low:+.2f}, {self.high:+.2f}]"
        if self.unknown_inputs:
            s += f"  依赖未测量: {', '.join(self.unknown_inputs)}"
        elif self.straddles_zero:
            s += "  不可判定 (区间跨 0)"
        return s


def propagate(metric: Callable[[Mapping[str, float]], float],
              *, ranged: Sequence[Ranged] = RANGED,
              unknowns: Sequence[Unknown] = UNKNOWNS,
              depends_on_unknown: Sequence[str] = ()) -> Interval:
    """把有区间的输入一次变一个地推到 metric 上, 返回点值与区间.

    metric: 接收 {输入路径: 取值} 的覆盖字典, 返回要比较的那个量 (如 Δ%)。
    depends_on_unknown: 调用方知道这个结论还依赖哪些没范围的输入 (按名字给)。
    """
    nominal = float(metric({}))
    lo = hi = nominal
    driver, spread = "", 0.0
    for r in ranged:
        vals = [float(metric({r.name: v})) for v in r.endpoints]
        lo, hi = min(lo, *vals), max(hi, *vals)
        width = max(vals) - min(vals)
        if width > spread:
            driver, spread = r.name, width
    names = {u.name for u in unknowns}
    dep = tuple(n for n in depends_on_unknown if n in names)
    return Interval(nominal=nominal, low=lo, high=hi, driver=driver,
                    unknown_inputs=dep)


def report() -> str:
    """把"模型知道自己不知道什么"列出来 —— 用在报告与 README 里."""
    lines = ["未标定到点的输入:", ""]
    for r in RANGED:
        lines.append(f"  [有区间] {r.name} = {r.nominal:g}  "
                     f"[{r.low:g}, {r.high:g}]")
        lines.append(f"           {r.source}")
    for u in UNKNOWNS:
        tag = f" (要定它: {u.calibration})" if u.calibration else ""
        lines.append(f"  [无区间] {u.name}{tag}")
        lines.append(f"           {u.reason}")
    return "\n".join(lines)
