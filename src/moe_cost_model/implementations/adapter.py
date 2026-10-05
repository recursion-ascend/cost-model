"""实现适配器 (lowering adapter): 把一份具体 kernel 的编排展开成事件图.

分层里的位置与职责边界:

  Workload / Runtime / Compile  ——  输入事实 (谁、在什么机器上、用什么二进制)
  **Adapter**                   ——  这一份实现的编排: 波怎么推进、哪个 stage 落哪个核、
                                    中间结果放哪、什么时候同步
  DAG (scheduler/events)        ——  统一事件图
  Scheduler / Timing            ——  只模拟事件在硬件资源上的执行, **不认识任何 stage 名**

为什么要这个接口 —— 在它之前, "换一份 kernel" 这件事在代码里的表达是
`KernelConfig.topo_urma` 这个布尔加 `registry._ORCHESTRATION` 两项表, 而
`m_groups_per_wave` / `waves` / "执行时间记到最后一个 combine" 这三件同样随实现而变的事
散在 model.py 里按 `topo_urma` 分支。第三份实现 (A8W4 多一个权重解压阶段, A4W4 换数据
格式) 无处落脚。

**本步只做归位, 不动行为。** 适配器把现有建图器原样包起来: `lower()` 的实现就是调用
`MteEventBuilder.build` / `LayeredEventBuilder.build`, 一行代码都没有搬进搬出。判据是
golden 的 39 个指纹逐位不变 (tools/gen_golden.py --check --explain: 行为类零差异)。
把 DRAIN_STAGES / 尾段链 / 共位规则这些真正属于适配器的东西搬过来, 是后面的步骤 ——
它们会动 Event.order 与事件名, 必须单独一步、单独一次 golden 重生成。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .compile import CompileConfig
from .identity import ImplementationId


class Unsupported(NotImplementedError):
    """这个适配器不支持给定的编译点/编排组合.

    继承 NotImplementedError 是为了与现存的拒绝保持同一个类型 (model.py 原先直接
    raise NotImplementedError), 调用方的 except 不必改。
    """


@dataclass(frozen=True)
class WavePlan:
    """波计划: 波宽 + 波序列. 适配器算一次, 建图与后处理都用这一份.

    为什么要缓存成对象: model._postprocess 会**再算一遍** waves 并把结果放进输出
    (wave_count 进 golden 指纹)。两次计算走不同路径就有漂移风险, 所以算一次传下去。
    """

    m_groups_per_wave: int
    waves: Tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.waves)


class ImplementationAdapter(Protocol):
    """一份具体实现在模型里的全部接口.

    方法分三组:
      身份      identity()                 —— 名字与源码依据
      接受性    accepts(compile_cfg, opts) —— 这个编译点/编排组合支持吗 (不支持抛 Unsupported)
      降解      plan(...) / lower(...)     —— 波计划与事件图
      观察点    measured_end_stage()       —— 执行时间记到哪个 stage 结束
    """

    def identity(self) -> ImplementationId: ...

    def accepts(self, compile_cfg: CompileConfig, options: Any) -> None: ...

    def plan(self, shape: Any, compile_cfg: CompileConfig, options: Any) -> WavePlan: ...

    def lower(self, shape: Any, plan: WavePlan, costs: Any, options: Any) -> Tuple[List[Any], List[Any]]: ...

    def measured_end_stage(self) -> str: ...
