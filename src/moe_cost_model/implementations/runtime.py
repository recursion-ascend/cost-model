"""运行期配置: 每次运行才知道的事实.

与编译期的分界线: 改它**不需要重编 kernel**。所以这里放的是 token 分布、波推进策略、
拓扑, 而不是 tile 几何或量化格式。

这一层目前是**把已有的东西归位**, 不新增行为: policy (波偏移/滞后/游标修正) 已经在
config/policy.InstancePolicy 里, 拓扑散落在 MegaMoeShape.aic_num 与 Workload.world,
核数 28 是场景文件里的手写事实 (规格是 32, 见 config/platform.SPEC_CUBE_CORES)。
把它们聚到一个对象上, 第 8 步的标定键才有东西可取。

kernel 侧的对应物 (运行期从 tiling data 读, 不在 ABI 之外):
  numMaxTokensPerRank   tiling @212
  rankNumPerServer      tiling @216 (仅 URMA 用)
  epWorldSize           tiling @16
  aicNum / blockAivNum  tiling @32 / @36
取值出处见上游 op_host/op_tiling/arch35/mega_moe_tiling.cpp。
这四项里前两项与 ranks_per_server 当前**不在** config/pipeline.TILING_FIELDS 里, 所以
tiling 真值护栏核不到它们 —— 记在这里, 第 4 步补。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .identity import RuntimeTopology


@dataclass(frozen=True)
class RuntimeConfig:
    """一次运行的运行期事实 (拓扑 + 波推进策略的取值来源).

    topology   硬件实例怎么用: 几张卡、每卡几核、单机几卡、几条并发流。
    policy     波推进策略 (InstancePolicy): 预取几波、GMM2 滞后几波、游标修正。
               不复制它的字段, 直接持有对象 —— 只有一个真相。
    """

    topology: RuntimeTopology = field(default_factory=RuntimeTopology)
    policy: object = None            # config.policy.InstancePolicy

    @classmethod
    def from_shape(cls, shape, *, world_size: Optional[int] = None,
                   ranks_per_server: Optional[int] = None) -> "RuntimeConfig":
        """从 MegaMoeShape 取运行期事实.

        world_size 不在 shape 上 (shape 是**单 rank** 的视角), 由调用方给:
        api.simulate_routing_counts 知道它 = len(routing_counts)。
        """
        src = getattr(shape, "expert_source_tokens", ()) or ()
        inferred_world = len(src[0]) if src and src[0] else None
        return cls(
            topology=RuntimeTopology(
                world_size=world_size if world_size is not None else inferred_world,
                active_cores=int(getattr(shape, "aic_num", 0)) or None,
                ranks_per_server=ranks_per_server,
            ),
            policy=getattr(shape, "policy", None),
        )
