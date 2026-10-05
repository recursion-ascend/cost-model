"""实现身份: 一个具体 kernel 变体在模型里的名字, 以及标定值能用在哪里.

为什么需要这一层 —— 在它之前, 这个项目只能表达"某一份 A8W8 Wave 实现":

  * 变体选择是一个布尔 (`KernelConfig.topo_urma`) 加一张两项的表
    (`registry._ORCHESTRATION = {"mte": ..., "layered": ...}`), 名字里没有硬件、
    没有版本、没有编译点;
  * `ReferenceProfile` 有 `name` / `source` (自由文本), 没有任何机器可核对的身份;
  * 全部标定常数是模块级全局量 (`config/hardware.py`), 一套数覆盖所有编排 ——
    而实际上 `config/hardware.py` 里几乎每个 `measured:` 常数的注释都写明了它是在哪一个
    形状、哪一个核数、哪一条编排下量出来的。换一套编排, 那些数未必还成立。

本模块只做**命名与域**, 不做任何计算:

  ImplementationId   (hardware_id, implementation_id, variant) 三段式名字
  CalibrationDomain  一组标定值声称有效的范围 (形状域 + 运行拓扑 + 编译指纹)

命名三段的含义:
  hardware_id        硬件平台, 例如 "ascend950"。决定物理规格 (带宽/算力/容量)。
  implementation_id  算法与实现族, 例如 "megamoe.a8w8_wave"。决定 DAG 的形状。
  variant            同一实现的版本, 例如 "v1"。源码改了编排就要换它。

为什么不把编译参数编进名字: 编译点有十几个轴 (见 config/hardware.KernelConfig 与
mega_moe/include/CMakeLists.txt 的 MEGAMOE_* 宏), 塞进名字会得到一个没人能念的字符串,
而且 kernel 自己的 tiling key 也只编码 5 个轴 (mega_moe_tiling_key.h:33-45), 不含
TILE_M/TILE_N/L1_BUF_NUM/IsGmm1Interleaved。所以编译点用**指纹**表达, 见 compile.py。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

#: 名字各段允许的字符: 小写字母、数字、下划线、点 (点用于分层, 如 megamoe.a8w8_wave)。
#: 收紧是为了让 "id 字符串" 能直接做文件名、表键与报告里的列名, 不必再转义。
_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789_.")


def _check(part: str, what: str) -> str:
    part = str(part)
    if not part:
        raise ValueError(f"{what} 不能为空")
    bad = sorted(set(part) - _ALLOWED)
    if bad:
        raise ValueError(
            f"{what} {part!r} 含不允许的字符 {bad}; 只允许小写字母/数字/下划线/点")
    return part


@dataclass(frozen=True)
class ImplementationId:
    """一个 kernel 变体的名字. 字符串形式 = "hardware.implementation.variant"."""

    hardware_id: str
    implementation_id: str
    variant: str
    #: 源码依据: 指向仓内文件的相对路径 (可带行号), 说明这个身份对应哪份实现。
    #: 自由文本不行 —— 它是这一层存在的理由之一, 所以要求能被 tools 核对的路径。
    source_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check(self.hardware_id, "hardware_id")
        _check(self.implementation_id, "implementation_id")
        _check(self.variant, "variant")

    @property
    def key(self) -> str:
        return f"{self.hardware_id}.{self.implementation_id}.{self.variant}"

    def __str__(self) -> str:        # 报告里直接当列名用
        return self.key

    @classmethod
    def parse(cls, text: str) -> "ImplementationId":
        """从 "ascend950.megamoe.a8w8_wave.v1" 还原.

        中间段可以带点 (megamoe.a8w8_wave), 所以切分规则是: 第一段是硬件, 最后一段是
        版本, 中间全部属于实现族。
        """
        parts = str(text).split(".")
        if len(parts) < 3:
            raise ValueError(
                f"实现 id {text!r} 至少要三段 (hardware.implementation.variant)")
        return cls(hardware_id=parts[0], implementation_id=".".join(parts[1:-1]),
                   variant=parts[-1])


@dataclass(frozen=True)
class ShapeDomain:
    """一组标定值声称覆盖的形状范围. None = 这一维没有声明边界 (不是"任意")."""

    token_num: Optional[Tuple[int, int]] = None      # 每 rank token 数 [低, 高]
    h: Optional[Tuple[int, int]] = None
    hidden_dim: Optional[Tuple[int, int]] = None
    topk: Optional[Tuple[int, int]] = None
    local_experts: Optional[Tuple[int, int]] = None

    def covers(self, **actual) -> Tuple[bool, Tuple[str, ...]]:
        """实际形状是否落在声明范围内. 返回 (是否覆盖, 越界的维度名).

        没有声明边界的维度**不参与判断**, 并且不算"覆盖" —— 调用方要能区分
        "量过这个点" 与 "没说过这个维度", 所以越界与未声明分开报 (后者见 undeclared)。
        """
        out = []
        for name, rng in self.__dict__.items():
            if rng is None or name not in actual or actual[name] is None:
                continue
            lo, hi = rng
            if not (lo <= actual[name] <= hi):
                out.append(name)
        return (not out), tuple(out)

    def undeclared(self, **actual) -> Tuple[str, ...]:
        """实际给了值、而本域没有声明范围的维度."""
        return tuple(sorted(n for n, v in actual.items()
                            if v is not None and getattr(self, n, None) is None))


@dataclass(frozen=True)
class RuntimeTopology:
    """标定时的运行拓扑. 这些是**运行期事实**, 不是编译参数, 也不是硬件规格.

    为什么要记: config/hardware.py 的注释已经把它们写出来了, 只是没地方存 ——
    例如 BW_L1_GM 的标定注释说"28 核 45300 / 18 核 37000", URMA_GET_LAT_US 说
    "4 卡 / 3 条流, 超过 world-1 > 3 未验证"。这些数换个拓扑就不成立, 而模型此前
    用一个全局常数覆盖所有拓扑。
    """

    world_size: Optional[int] = None          # 参与的 rank 数
    active_cores: Optional[int] = None        # 实际用的 AI Core 数 (如 28, 非规格 32)
    ranks_per_server: Optional[int] = None    # 单机内的 rank 数 (跨机跳数由它决定)
    concurrent_streams: Optional[int] = None  # 并发流数 (URMA 标定里有这一维)

    def mismatch(self, **actual) -> Tuple[str, ...]:
        """与实际拓扑不符的维度 (None 的维度跳过)."""
        return tuple(sorted(
            n for n, v in self.__dict__.items()
            if v is not None and actual.get(n) is not None and actual[n] != v))


@dataclass(frozen=True)
class CalibrationDomain:
    """一组标定值的适用域: 在**哪个实现、哪个编译点、哪些形状、哪种拓扑**下量的.

    五个键正是"换一套编排后这些常数未必还有效"这句话的五个方面。compile_fingerprint
    为空串表示"没有记录编译点" —— 那本身是一条要暴露的缺口, 不是通配符。
    """

    implementation: ImplementationId
    compile_fingerprint: str = ""
    shape: ShapeDomain = field(default_factory=ShapeDomain)
    topology: RuntimeTopology = field(default_factory=RuntimeTopology)
    #: 数据来源 (打点 run 目录名 / 文档路径), 供复现。
    evidence: Tuple[str, ...] = ()

    @property
    def key(self) -> Tuple[str, str]:
        return (self.implementation.key, self.compile_fingerprint)
