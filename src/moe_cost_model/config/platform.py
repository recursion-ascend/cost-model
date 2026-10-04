"""硬件规格 (spec): 从官方架构白皮书来的**上界**, 与实测值分开记.

为什么要和 measured 分开: 规格是峰值, 用它算出来的是**时间下界**, 所以该配一个效率
系数; 实测值已经含了争用与开销, 不该再乘效率。两者混在一个标签里早晚用错。

出处: 《昇腾 950 NPU 架构白皮书》(华为)。本仓的 kernel 是 __NPU_ARCH__==3510
(第三代达芬奇 / arch35)。950PR 与 950DT 同源同 die, 差别在存储与互连档位,
每核计算峰值相同 —— 所以 compute 侧只有一组数, 存储/互连侧两组。

**核对状态**: 下面的数字取自白皮书规格表的转述, 本仓未能直接打开官方 PDF 逐字核对
(容器网络策略拦了 hiascend.com 与华为 OBS)。拿到原件请逐项核对, 尤其是:
  * Cube 部分与 Vector 部分的算力如何拆分 (白皮书给的 919/486/1784 是合计值);
  * 950PR 的 HBM 容量 (转述有 112GB 与 128GB 两说; 本模型不用容量, 只用带宽)。
"""
from __future__ import annotations

from dataclasses import dataclass

from .provenance import SourcedInt, SourcedValue

#: 白皮书规格表 (32 Cube / 64 Vector 配置) 的**合计**算力, TFLOPS。含 Vector 部分。
TOTAL_TFLOPS_FP8 = SourcedValue(919.0, 'spec:白皮书 HiF8/MXFP8/FP8 合计算力, TFLOPS')
TOTAL_TFLOPS_FP16 = SourcedValue(486.0, 'spec:白皮书 BF16/FP16 合计算力, TFLOPS')
TOTAL_TFLOPS_MXFP4 = SourcedValue(1784.0, 'spec:白皮书 MXFP4 合计算力, TFLOPS')
TOTAL_TFLOPS_TF32 = SourcedValue(243.0, 'spec:白皮书 TF32 合计算力, TFLOPS')
#: Vector 部分的算力 (从合计里减掉它才是 Cube 部分)
VECTOR_TFLOPS_FP16 = SourcedValue(54.0, 'spec:白皮书 Vector 部分 FP16/BF16 算力, TFLOPS')
VECTOR_TFLOPS_FP32 = SourcedValue(27.0, 'spec:白皮书 Vector 部分 FP32 算力, TFLOPS')

#: 规格配置里的 Cube 核数 (整颗最大 36, 950PR 档 32)。注意这是**算力分母**:
#: 峰值按核给, 不随实际启用核数变。可用核数是另一个量 (MegaMoeShape.aic_num)。
SPEC_CUBE_CORES = SourcedInt(32, 'spec:白皮书 950PR 档 Cube 核数 (整颗最大 36)')

#: Cube 部分 FP16 算力 = 合计 - Vector 部分。
CUBE_TFLOPS_FP16 = float(TOTAL_TFLOPS_FP16) - float(VECTOR_TFLOPS_FP16)       # 432
#: 白皮书: HiF8/MXFP8/FP8 同频给 FP16 的 2 倍, MXFP4 给 4 倍。
#: 自洽核对: 2*432 + 54 = 918 ≈ 合计 919; 4*432 + 54 = 1782 ≈ 合计 1784。
CUBE_TFLOPS = {
    "fp8": CUBE_TFLOPS_FP16 * 2.0,      # 864
    "mxfp8": CUBE_TFLOPS_FP16 * 2.0,
    "hif8": CUBE_TFLOPS_FP16 * 2.0,
    "int8": CUBE_TFLOPS_FP16 * 2.0,
    "fp16": CUBE_TFLOPS_FP16,
    "bf16": CUBE_TFLOPS_FP16,
    "mxfp4": CUBE_TFLOPS_FP16 * 4.0,    # 1728
    "tf32": float(TOTAL_TFLOPS_TF32),
}


def cube_mac_per_us(dtype: str = "fp8", efficiency: float = 1.0,
                    cube_cores: int = int(SPEC_CUBE_CORES)) -> float:
    """每个 Cube 核的 MAC/µs 峰值 x 效率系数.

    为什么要除 2: 规格给的是 FLOPS, 而一次 MAC 算两个 FLOP; 本模型的计算量分子是
    **MAC 数** (GMM1 = 2·m·cols·K, 那个 2 是 gate/up 两个投影, 不是 FLOP/MAC;
    GMM2 = m·cols·K2)。单位搞错会让计算时间整整差一倍。

    A8W8 主路径对应 dtype="fp8": 864 TFLOPS / 32 核 / 2 = 1.35e7 MAC/µs。

    efficiency: 峰值的达成率。1.0 = 峰值 = 计算时间的**下界**。实测反解远低于峰值
    (bs36 反解 3.3e6, 是峰值的 24%), 但那批形状本来就是载入绑定的 —— 所以这里该填
    峰值让模型自己判断谁绑定, 不该把"当时恰好载入绑定"烧进常数。
    """
    key = dtype.lower()
    if key not in CUBE_TFLOPS:
        raise ValueError(f"未知 dtype {dtype!r}; 可选: {', '.join(sorted(CUBE_TFLOPS))}")
    if cube_cores <= 0:
        raise ValueError("cube_cores 必须为正")
    if efficiency <= 0:
        raise ValueError("efficiency 必须为正")
    per_core_flops_per_us = CUBE_TFLOPS[key] * 1e12 / cube_cores / 1e6
    return per_core_flops_per_us / 2.0 * float(efficiency)


@dataclass(frozen=True)
class PlatformSpec:
    """一个存储/互连档位的规格上界. 计算侧两档相同 (同源同 die)."""

    name: str
    source: str
    hbm_bytes_per_us: float        # 聚合 HBM 带宽 (B/µs; 1 TB/s = 1e6 B/µs)
    fabric_bytes_per_us: float     # 聚合片间互连带宽 (灵衢 2.0)
    hbm_capacity_gb: float = 0.0   # 本模型不用, 只为出处完整

    def gm_bw_per_core(self, active_cores: int,
                       per_core_limit: float) -> float:
        """单核 GM→L1 有效带宽 = min(单核上限, 聚合带宽 / 活跃核数).

        单核上限是实测量 (BW_L1_GM); 聚合是规格上界。两者取小 —— 一个常数在核数
        足够多时会突破聚合带宽, 那是物理上不可能的。
        """
        if active_cores <= 0:
            raise ValueError("active_cores 必须为正")
        return min(float(per_core_limit), self.hbm_bytes_per_us / active_cores)

    def hbm_utilisation(self, active_cores: int, per_core_bw: float) -> float:
        """这个单核带宽在该核数下占聚合带宽的比例 (>1 即物理上不可能)."""
        return active_cores * float(per_core_bw) / self.hbm_bytes_per_us


ASCEND_950PR = PlatformSpec(
    name="Ascend 950PR",
    source="spec:白皮书 950PR 档 (自研 HBM HiBL 1.0)",
    hbm_bytes_per_us=1.6e6,        # 1.6 TB/s
    fabric_bytes_per_us=2.0e6,     # 灵衢 2.0, 2 TB/s
    hbm_capacity_gb=128.0,         # 转述有 112/128 两说; 模型不用
)

ASCEND_950DT = PlatformSpec(
    name="Ascend 950DT",
    source="spec:白皮书 950DT 档 (自研 HBM HiZQ 2.0)",
    hbm_bytes_per_us=4.0e6,        # 4 TB/s
    fabric_bytes_per_us=2.0e6,     # 灵衢 2.0, 2 TB/s
    hbm_capacity_gb=144.0,
)

PLATFORMS = {"950pr": ASCEND_950PR, "950dt": ASCEND_950DT}


def resolve_platform(name: str) -> PlatformSpec:
    key = str(name).strip().lower().replace("ascend", "").replace(" ", "").replace("-", "")
    if key not in PLATFORMS:
        raise ValueError(f"未知平台 {name!r}; 可选: {', '.join(sorted(PLATFORMS))}")
    return PLATFORMS[key]
