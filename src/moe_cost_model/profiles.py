"""实例层: "某一份实现是设计空间里的哪个点".

建模原则: **没有任何参数的取值可以用"等于某一版 kernel"来定义自己的含义**。
编排参数的缺省值是"最少假设" (不声称实现做了什么特殊的事); 一份具体实现的取值
集中记在这里, 要复现它就显式引用, 要扫设计空间就不引用。

这样做的好处:
  * 缺省跑出来的数是模型的数, 不是某一版实现的数 —— 两者不同是信息, 不是 bug。
  * 实现升级只改这一个对象, 不改模型。
  * "这个取值的出处是什么"有地方可写 (source 字段), 不必散在各处注释里。

用法:
    from moe_cost_model.profiles import MEGAMOE_A8W8 as P
    simulate_routing_counts(..., **P.shape_kw(), options=P.options)
    # 只改一项:
    simulate_routing_counts(..., **P.shape_kw(), options=P.with_options(gmm2_k_segments=0))
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict

from .config.hardware import EpilogueOverheads, KernelConfig
from .config.policy import InstancePolicy
from .planning.core_assignment import StaticRoundRobin
from .planning.tile_grid import SwizzledTileGrid
from .planning.wave_packing import SequentialGreedy
from .shape import ModelOptions


@dataclass(frozen=True)
class ReferenceProfile:
    """一份具体实现在设计空间里的坐标.

    options / kernel / policy 是模型入口直接吃的三个对象; 其余是 MegaMoeShape 上的
    策略字段, 经 shape_kw() 一次性给出。
    """

    name: str
    source: str
    options: ModelOptions
    kernel: KernelConfig
    policy: InstancePolicy
    tile_grid: Any = None
    core_assignment: Any = None
    wave_packing: Any = None

    def with_options(self, **overrides) -> ModelOptions:
        """以本 profile 的编排为底, 只改指定几项 (扫设计空间用)."""
        return dataclasses.replace(self.options, **overrides)

    def with_policy(self, **overrides) -> InstancePolicy:
        return dataclasses.replace(self.policy, **overrides)

    def with_kernel(self, **overrides) -> KernelConfig:
        return dataclasses.replace(self.kernel, **overrides)

    def scenario_fields(self) -> Dict[str, Any]:
        """Scenario 的字段名 → 本 profile 的取值 (场景文件 profile= 用)."""
        out = {"options": self.options, "kernel": self.kernel, "policy": self.policy}
        for name in ("tile_grid", "core_assignment", "wave_packing"):
            val = getattr(self, name)
            if val is not None:
                out[name] = val
        return out

    def shape_kw(self, **overrides) -> Dict[str, Any]:
        """MegaMoeShape / api 入口的策略字段 (kernel / policy / 三个 planning 策略)."""
        kw: Dict[str, Any] = {
            "kernel": self.kernel,
            "policy": self.policy,
            "tile_grid": self.tile_grid,
            "core_assignment": self.core_assignment,
            "wave_packing": self.wave_packing,
        }
        kw.update(overrides)
        return {k: v for k, v in kw.items() if v is not None}


# MegaMoe A8W8 Wave 融合算子 (arch35) 的坐标。
#
# 每一项都是"那份实现这么做", 不是"物理只能这么做" —— 所以它在这里而不在缺省值里。
MEGAMOE_A8W8 = ReferenceProfile(
    name="megamoe-a8w8-wave-arch35",
    source=(
        "mega_moe/op_kernel/arch35 + megamoe_profile/CMakeLists.txt:28-32; "
        "固定开销来自 20260930 的实测 trace"
    ),
    options=ModelOptions(
        # ACT 写 GM、GMM2 从 GM 读回 (epilogue 写 workspaceInfo.activationQuantDataPtr,
        # stage/mega_moe_gmm2_combine.h:770 从 Location::GM 取同一个指针)
        act_to_gmm2="gm",
        # GMM2 沿 K 两段就绪: 首个 kL1 块一段 (只等 1 个 ACT), 其余合成一段
        gmm2_k_segments=2,
        # dispatch 按核预切: 均衡分配 + startBlockIdx 轮转
        dispatch_partition="kernel",
        # AIV1 循环体先 combine 再下一波 dispatch
        dispatch_pacing="per_core",
        # 尾段五项固定开销 = 实测残留
        epilogue_overheads=EpilogueOverheads(),
        # tile->核 在建图时定死 (startBlockIdx 旋转), 不是派发时挑
        late_bind_pools=(),
        # 波宽由 p1/p2 推导 (kernel 自己按 token 数查表, 见
        # planning.waves.resolve_gmm1_min_logical_tiles_per_core)
        m_groups_per_wave=0,
    ),
    kernel=KernelConfig(),
    policy=InstancePolicy(),
    tile_grid=SwizzledTileGrid(),
    core_assignment=StaticRoundRobin(),
    wave_packing=SequentialGreedy(),
)


# 名字 → profile (场景文件里 profile = "<名字>" 用这张表解析)
PROFILES: Dict[str, ReferenceProfile] = {
    "megamoe-a8w8": MEGAMOE_A8W8,
}


def resolve_profile(name: str) -> ReferenceProfile:
    """按名字取 profile; 未知名字直接报错并列出可选值 (零猜测)."""
    key = str(name).strip()
    if key not in PROFILES:
        raise ValueError(
            f"未知 profile {name!r}; 可选: {', '.join(sorted(PROFILES))}")
    return PROFILES[key]
