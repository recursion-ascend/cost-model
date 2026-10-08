"""实现层: 身份 / 编译点 / 运行期 / 适配器.

四层架构里的中间两层 (Compile 与 Kernel Implementation Lowering)。入口:

  ImplementationId      一份 kernel 变体的名字 (hardware.implementation.variant)
  CompileConfig         一个编译点的全部轴 + 编译指纹
  RuntimeConfig         一次运行的拓扑与波推进策略
  CalibrationDomain     一组标定值声称有效的范围 (身份 + 指纹 + 形状域 + 拓扑)
  ImplementationAdapter 适配器接口; megamoe.py 里是仓内两份实现
"""
from .adapter import ImplementationAdapter, Unsupported, WavePlan
from .calibration import (CORPUS_COMPILE, CORPUS_POINT, CORPUS_SHAPE,
                          CORPUS_TOPOLOGY,
                          CalibrationRecord, CalibrationTable, Lookup,
                          audit_run, default_table)
from .compile import FINGERPRINT_AXES, CompileConfig
from .identity import (CalibrationDomain, ImplementationId, RuntimeTopology,
                       ShapeDomain)
from .megamoe import (ADAPTERS, ALIASES, A8W4WaveV1Declared, A8W8WaveV1,
                      LayeredV1, adapter_for, resolve)
from .runtime import RuntimeConfig

__all__ = [
    "ADAPTERS", "ALIASES", "A8W4WaveV1Declared", "A8W8WaveV1", "CORPUS_COMPILE",
    "CORPUS_POINT", "CORPUS_SHAPE", "CORPUS_TOPOLOGY", "CalibrationRecord", "CalibrationTable",
    "Lookup", "audit_run", "default_table", "CalibrationDomain", "CompileConfig",
    "FINGERPRINT_AXES", "ImplementationAdapter", "ImplementationId", "LayeredV1",
    "RuntimeConfig", "RuntimeTopology", "ShapeDomain", "Unsupported", "WavePlan",
    "adapter_for", "resolve",
]
