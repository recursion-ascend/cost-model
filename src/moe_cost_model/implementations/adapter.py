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

适配器把建图器包起来: `lower()` 调用 `MteEventBuilder.build` /
`LayeredEventBuilder.build`。DRAIN_STAGES / 尾段链 / 共位规则这些同样属于适配器的
东西还留在 model.py 与 builders/base.py —— 搬过来会动 Event.order 与事件名。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from ..config.stages import StageVocabulary
from .compile import CompileConfig
from .identity import ImplementationId


class Unsupported(NotImplementedError):
    """这个适配器不支持给定的编译点/编排组合.

    继承 NotImplementedError 是为了与 model.py 里的拒绝保持同一个类型, 调用方的
    except 不必改。
    """


@dataclass(frozen=True)
class WavePlan:
    """波计划: 波宽 + 波序列. 适配器算一次, 建图与后处理都用这一份.

    为什么要缓存成对象: model._postprocess 会**再算一遍** waves 并把结果放进输出
    (wave_count 进结果)。两次计算走不同路径就有漂移风险, 所以算一次传下去。
    """

    m_groups_per_wave: int
    waves: Tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.waves)


class ImplementationAdapter(Protocol):
    """一份具体实现在模型里的全部接口.

    绑定纪律 (binding) 为什么在这里而不是在 ModelOptions 里: "tile 落哪个核" 由谁决定
    是**那份实现的事实**, 不是使用者的偏好。编译期分好块的 kernel (如仓内两份 MegaMoE,
    滚动游标 startBlockIdx_) 运行时零开销地知道自己干哪些 tile; 从共享游标动态取活的
    实现要付一次原子加。两者是不同的实现, 不是同一件事的两种近似。

    它仍然可以被 ModelOptions.late_bind_pools 覆盖 —— 那是**比较**所需: 同一份实现换一
    种绑定纪律跑一遍, 差值就是这个选择的代价。覆盖与声明不一致时结果里记一条告警
    (rank_results["implementation"]["binding_note"]), 因为那时模型描述的已经不是声明
    的那台机器。

    方法分五组:
      身份      identity()                 —— 名字与源码依据
      词汇表    stages()                   —— 这份实现划了哪些 stage、它们之间有哪些边
      绑定纪律  binding()                  —— tile->核 由谁决定: 建图期还是派发时刻
      接受性    accepts(compile_cfg, opts) —— 这个编译点/编排组合支持吗 (不支持抛 Unsupported)
      降解      plan(...) / lower(...)     —— 波计划与事件图
      观察点    measured_end_stage()       —— 执行时间记到哪个 stage 结束
    """

    def identity(self) -> ImplementationId: ...

    def stages(self) -> StageVocabulary: ...

    def binding(self) -> Tuple[str, ...]: ...

    def accepts(self, compile_cfg: CompileConfig, options: Any) -> None: ...

    def plan(self, shape: Any, compile_cfg: CompileConfig, options: Any) -> WavePlan: ...

    def lower(self, shape: Any, plan: WavePlan, costs: Any, options: Any) -> Tuple[List[Any], List[Any]]: ...

    def measured_end_stage(self) -> str: ...
