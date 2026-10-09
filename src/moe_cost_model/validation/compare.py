"""预测 DAG 与实测 trace 的**结构**比对.

先把能比什么说清楚, 因为这一步最容易自欺:

  可比   波数、逐专家分布的形状、用了几个核、每核事件数的均衡度
  **粒度不同, 不能直接比条数**: trace 的 GMM1 标记是每次 `RunGmm1Generic` 调用一个
         (一个 (核, 波, 专家) 切片一个, bs128 那个 run 是 108 个), 而模型是每个 **tile**
         一个事件 (同一个 run 是 27 个)。两个数都对, 口径不同。所以本模块报**两个数与
         它们的比**, 并要求比值在各专家间一致 —— 一致说明两边描述的是同一个结构,
         只是刻度不同; 不一致才是真问题。
  不可比 **字节**: trace 的 args 只有 rank/local_id/payload/cycles/wave/expert, 没有任何
         搬运字节字段 (见 validation/trace 的说明)。
  不可比 **buffer 生命周期**: trace 里只有等待事件这个影子, 没有槽位取/还。

还有一个数据事实: 两个 bs8192 run 的 8 个 trace 文件都被截断 (同一个字节数), 事件数只是
下界。比对结果里会标出来 —— 否则"模型比实测多"会被当成模型的问题。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config.stages import default_vocabulary
from ..ir.vocabulary import classify_resource
from .trace import TraceFile

#: 比对覆盖哪些 stage 由**实现声明的词汇表**给 (config/stages.py): 一份实现的
#: stage 划分决定哪些事件在模型里存在。MegaMoE 不列 dispatch_call 的理由写在
#: implementations/megamoe_stages。
COMPARED_STAGES = default_vocabulary().compared


def time_groups(starts: Sequence[float]) -> int:
    """实测事件按时间空档切出的段数 (诊断用, **不**当作轮数).

    为什么只做诊断: "段数 = 轮数"靠不住 —— 同一个 run 的不同 stage 用 8 倍中位间隔切出来
    是 7/10/3/5/14 段, 彼此矛盾 (28 个核的事件交错, 轮内间隔本身差异很大)。所以这里只报
    段数这个**证据**, 归一化交给人: 把一个靠不住的推断写进比值, 比不写更糟。

    **"采集含多轮"这个假设已被证伪**: bs128 rank0 的 108 条 GMM1 标记里,
    54 个 (波,专家,核,引擎) 元组各出现两次, 两次的 ts 与 dur **逐位相同** —— 多轮会有不同
    的时刻。真正的原因是那个 trace 文件含**同一次运行的两个视图** (两个 pid: "完整流水"
    与 "隐藏 WAIT"), 而读取器把两个都收了; 已在 validation/trace.py 改成只读一个视图。
    config.json5 的 warmup: 3 也注明"正式固定一次", 本就不该进 trace。
    """
    if len(starts) < 4:
        return 1
    ordered = sorted(starts)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    median = sorted(gaps)[len(gaps) // 2] or 1e-9
    return 1 + sum(1 for g in gaps if g > 8 * median and g > 1.0)


@dataclass
class StageComparison:
    """一个 stage 的比对结果."""

    stage: str
    trace_count: int
    model_count: int
    trace_experts: Dict[int, int] = field(default_factory=dict)
    model_experts: Dict[int, int] = field(default_factory=dict)
    trace_cores: int = 0
    model_cores: Optional[int] = None
    trace_waves: Tuple[int, ...] = ()
    model_waves: Tuple[int, ...] = ()
    #: 实测事件按时间空档切出的段数 (诊断用, 不是轮数 —— 见 time_groups)
    trace_time_groups: int = 1

    @property
    def ratio(self) -> Optional[float]:
        """实测条数 / 模型条数 (原始)."""
        return (self.trace_count / self.model_count) if self.model_count else None



    @property
    def expert_ratios(self) -> Dict[int, Optional[float]]:
        out: Dict[int, Optional[float]] = {}
        for expert in sorted(set(self.trace_experts) | set(self.model_experts)):
            model = self.model_experts.get(expert, 0)
            out[expert] = (self.trace_experts.get(expert, 0) / model) if model else None
        return out

    @property
    def comparable_by_expert(self) -> bool:
        """实测侧带了专家号, 逐专家比值才有意义 (DISPATCH_* 没带)."""
        return bool(self.trace_experts) and bool(self.model_experts)

    @property
    def consistent(self) -> bool:
        """逐专家的比值彼此一致 (容差 1%) —— 一致 = 两边是同一个结构的两种刻度."""
        if not self.comparable_by_expert:
            return False
        got = [r for r in self.expert_ratios.values() if r is not None]
        if not got or self.ratio is None:
            return False
        return max(got) - min(got) <= 0.01 * max(1e-9, self.ratio)

    def issues(self) -> List[str]:
        out = []
        if self.model_count == 0 and self.trace_count:
            out.append(f"{self.stage}: 实测有 {self.trace_count} 个, 模型一个都没有")
        if self.trace_count == 0 and self.model_count:
            out.append(f"{self.stage}: 模型有 {self.model_count} 个, 实测一个都没有")
        if not self.trace_experts and self.model_experts:
            # 实测这一类事件根本没带专家号 (DISPATCH_XFER 的 args 只有
            # rank/local_id/payload/cycles), 所以"逐专家"这一项**不可比**, 不是问题。
            pass
        elif set(self.trace_experts) != set(self.model_experts):
            out.append(f"{self.stage}: 专家集合不同 "
                       f"实测 {sorted(self.trace_experts)} vs 模型 {sorted(self.model_experts)}")
        elif not self.consistent:
            out.append(f"{self.stage}: 逐专家条数比不一致 {self.expert_ratios} "
                       f"—— 两边描述的结构不一样, 不只是刻度不同")
        if self.ratio is not None and abs(self.ratio - 1.0) > 0.01:
            out.append(
                f"{self.stage}: 条数比 {self.ratio:.2f} (实测 {self.trace_count} vs 模型 "
                f"{self.model_count}), 逐专家一致, 时间上分 {self.trace_time_groups} 段。"
                f"两边都按 tile 计数 (kernel 的 MOE_PROFILE_BEGIN 带 "
                f"ProfileTile(mLoc,nLoc)), 所以这个比值要有解释才能读时长对比: "
                f"候选是 tile 网格真的不同, 或两边的事件粒度口径不同 "
                f"(dispatch 就是这一类: 模型按 dispatch_rows_per_item 成批, "
                f"kernel 按行级软流水)。'采集含多轮'已被证伪, 见 time_groups 的说明")
        if self.trace_waves and self.model_waves and \
                len(self.trace_waves) != len(self.model_waves):
            out.append(f"{self.stage}: 波数不同 实测 {len(self.trace_waves)} "
                       f"vs 模型 {len(self.model_waves)}")
        return out


@dataclass
class RunComparison:
    run: str
    truncated: bool
    note: str
    stages: Tuple[StageComparison, ...]
    #: 实测里有、但模型完全不建图的事件 (前导/等待), 以及等待事件计数
    not_modelled: Dict[str, int] = field(default_factory=dict)
    waits: Dict[str, int] = field(default_factory=dict)

    def issues(self) -> List[str]:
        out: List[str] = []
        for stage in self.stages:
            out.extend(stage.issues())
        return out

    def report(self) -> str:
        head = f"# {self.run}" + ("  [trace 被截断, 事件数是下界]" if self.truncated else "")
        lines = [head,
                 f"{'stage':12s}{'实测':>7s}{'模型':>7s}{'比':>7s}{'时间段':>7s}"
                 f"{'核(实测/模型)':>15s}{'波':>8s}  逐专家比一致"]
        for s in self.stages:
            ratio = "-" if s.ratio is None else f"{s.ratio:.2f}"
            cores = ('晚绑定' if s.model_cores is None else str(s.model_cores))
            lines.append(
                f"{s.stage:12s}{s.trace_count:7d}{s.model_count:7d}{ratio:>7s}"
                f"{s.trace_time_groups:7d}"
                f"{f'{s.trace_cores}/{cores}':>15s}"
                f"{f'{len(s.trace_waves)}/{len(s.model_waves)}':>8s}"
                f"  {'是' if s.consistent else ('不可比' if not s.comparable_by_expert else '否')}")
        issues = self.issues()
        lines.append("  结构一致 (口径差异已按逐专家比值归一)" if not issues
                     else "  问题:\n" + "\n".join(f"    - {i}" for i in issues))
        return "\n".join(lines)


def _model_counts(events, stage: str):
    """模型侧: 该 stage 的条数 / 逐专家条数 / 用了几个核 / 波集合."""
    rows = [e for e in events if (e.meta or {}).get("stage") == stage]
    experts: Dict[int, int] = {}
    cores = set()
    waves = set()
    late_bound = False
    for ev in rows:
        meta = ev.meta or {}
        if meta.get("expert") is not None:
            experts[int(meta["expert"])] = experts.get(int(meta["expert"]), 0) + 1
        # 用 IR 的分类器, 不自己解字符串: 核号有 "AIC:0" 与 "AIC:c7" 两种写法, 自己解
        # 两种都要认: 只认带 c 的那种会让模型侧核数恒为 0。
        for res in getattr(ev, "resources", ()) or ():
            got = classify_resource(res)
            if got is None:
                continue
            if got.late_bound:
                late_bound = True          # 晚绑定: 建图期还没有核号
            else:
                cores.add(got.core)
        if meta.get("wave") is not None:
            waves.add(int(meta["wave"]))
    # 晚绑定下核号在**派发时刻**才定, 建图期的事件带的是占位符 —— 此时"用了几个核"在
    # 模型侧无从得知, 报 None 而不是 0 (0 会被读成"一个核都没用")。
    n_cores = None if (late_bound and not cores) else len(cores)
    return len(rows), dict(sorted(experts.items())), n_cores, tuple(sorted(waves))


def compare_run(trace: TraceFile, model_events, run_name: str = "") -> RunComparison:
    """一个 rank 的实测 trace vs 模型事件图."""
    stages = []
    for stage in COMPARED_STAGES:
        t_rows = [e for e in trace.events if e.stage == stage]
        t_experts: Dict[int, int] = {}
        for ev in t_rows:
            if ev.expert is not None:
                t_experts[ev.expert] = t_experts.get(ev.expert, 0) + 1
        m_count, m_experts, m_cores, m_waves = _model_counts(model_events, stage)
        stages.append(StageComparison(
            stage=stage, trace_count=len(t_rows), model_count=m_count,
            trace_experts=dict(sorted(t_experts.items())), model_experts=m_experts,
            trace_cores=len({e.core for e in t_rows}), model_cores=m_cores,
            trace_waves=trace.waves(stage), model_waves=m_waves,
            trace_time_groups=time_groups([e.start_us for e in t_rows])))
    not_modelled: Dict[str, int] = {}
    for ev in trace.events:
        if ev.stage is None and not ev.is_wait:
            not_modelled[ev.raw_name] = not_modelled.get(ev.raw_name, 0) + 1
    return RunComparison(run=run_name or trace.path, truncated=trace.truncated,
                         note=trace.note, stages=tuple(stages),
                         not_modelled=dict(sorted(not_modelled.items())),
                         waits=trace.waits())
