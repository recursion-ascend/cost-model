"""第 4 层: 事件图构建公共基类 — 与算子无关的那部分.

剩下的三件事都不认识任何 stage 名: 事件工厂 ``_event``、均衡轮转
``_rotated_balanced_range``、排空栅栏 ``_add_completion`` (要排空哪些 stage 取自
词汇表的 drain 声明)。另加路由张量上的两个算术 (count_remote_rows /
rows_by_source_rank), 任何 EP 实现都要用。

MegaMoE 两条路径共用的实现专属建图在 megamoe_common.py: 尾段六个事件、共享专家
两段、dispatch 段枚举 IR、以及 GMM1 调度宽度取 hidden_dim/activation_n_half 这个
SwiGLU 假设。放在基类里会让任何新建图器只要继承就**白得一条 MegaMoE 的尾段**。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..config.stages import default_vocabulary
from ..scheduler.events import Event
from ..costs import PrimitiveCosts
from ..shape import CursorTrace, ModelOptions


def rows_by_source_rank(source_counts, row_begin: int, row_end: int) -> Tuple[int, ...]:
    """行区间 [row_begin, row_end) 按源卡拆出的逐卡行数.

    专家内的行按源卡顺序排布 (dispatch 就是按这个顺序分段写的, 见
    build_dispatch_expert_ir), 所以任意行区间的归属可以精确数出来 —— 不是按
    比例摊。COMBINE 要把每行写回它的来源卡, 这组数就是该窗发往各卡的行数,
    既定时长 (本卡/跨卡两个带宽), 也定片间信道流量 (逐目的卡一条边)。
    """
    out = []
    cur = 0
    for cnt in source_counts:
        nxt = cur + cnt
        lo = max(row_begin, cur)
        hi = min(row_end, nxt)
        out.append(hi - lo if hi > lo else 0)
        cur = nxt
    return tuple(out)


def count_remote_rows(source_counts, dst_rank: int, row_begin: int, row_end: int) -> int:
    """rows_by_source_rank 里源卡 != dst_rank 的行数合计."""
    by_src = rows_by_source_rank(source_counts, row_begin, row_end)
    return sum(n for src, n in enumerate(by_src) if src != dst_rank)


class EventBuilderBase:
    """持有建图工作状态, 按波序列生成事件图.

    状态生命周期 = 一次 build() 调用; 建成后本对象可丢弃.
    """

    def __init__(self, costs: PrimitiveCosts, options: ModelOptions):
        self.costs = costs
        self.options = options
        self.events: List[Event] = []
        self.cursor_trace: List[CursorTrace] = []
        self._order = 0
        self._rank = 0
        # combine 传输后端 (builders/comm): 由编排循环装配;
        # gmm2 stage 通过 on_gmm2_tile 钩子调用
        self.combine_backend = None
        # gmm2 tail 事件按 (expert, global_group) 归档 — layered combine 的
        # 组级就绪依赖 (kernel: GMM2 sync counter ≥ nTilesPerGroup)
        self.gmm2_tail_by_group: Dict[Tuple[int, int], List[str]] = {}
        # 共享专家 ACT 事件按 m-group 归档 — 共享 GMM2 的数据依赖
        self.shared_act_by_group: Dict[int, List[str]] = {}

    

    def _event(self, name: str, resources, duration_us: float,
               deps=(), meta=None,
               acquires=(), releases=(), channel_bytes=()) -> str:
        _rp = f"R{self._rank}."
        dep_tuple = tuple(
            dict.fromkeys(d if d.startswith(_rp) else _rp + d for d in deps if d))
        name = f"R{self._rank}.{name}"
        self.events.append(Event(
            name=name, resources=tuple(resources),
            duration_us=max(0.0, float(duration_us)),
            deps=dep_tuple, order=self._order,
            meta=dict(meta or {}, rank=self._rank),
            acquires=tuple(acquires), releases=tuple(releases),
            channel_bytes=tuple(channel_bytes),
        ))
        self._order += 1
        return name

    @staticmethod
    def _rotated_balanced_range(total: int, worker: int, workers: int,
                                global_prefix: int) -> Tuple[int, int]:
        if workers <= 0 or worker < 0 or worker >= workers:
            return (0, 0)
        first_owner = global_prefix % workers
        logical = worker - first_owner if worker >= first_owner else worker + workers - first_owner
        base, rem = divmod(total, workers)
        extra_before = logical if logical < rem else rem
        start = logical * base + extra_before
        count = base + (1 if logical < rem else 0)
        return start, count

    # stage 建图函数在同包各文件: gmm1.py / activation.py / gmm2.py, 通信与归约在
    # comm/{mte,urma}.py (dispatch 与 combine 都在那里, 没有 dispatch.py / combine.py) —
    # 状态经 BuildContext (context.py) 传递.

    # ---- 完成事件 ----

    #: 每个引擎上跑哪些 stage —— 排空节点按此归集本核该引擎的全部事件。
    #: 这张表属于具体实现 (哪些 stage 存在、落哪个引擎), 所以取自词汇表;
    #: 另一份实现的建图器覆盖这个类属性即可。
    DRAIN_STAGES = default_vocabulary().drain

    def _add_completion(self, p):
        """MoE 阶段的排空栅栏: **一个**零时长节点, 依赖全部 MoE 事件.

        C3: 不用每核每引擎一个节点 (28 x 3 = 84 个, 复刻内核的 WAIT_GMM_DRAIN)。
        但 cost model 真正需要表达的只是"尾段要等这批工作全做完" —— 尾段本来就依赖
        全部 84 个节点, 而每个节点依赖本核该引擎的全部事件, 所以传递闭包就是"依赖
        全部 MoE 事件"。一个栅栏与 84 个逐核节点**对尾段完全等价**, 却少 83 个节点、
        83 条出边, 而且名字不带核号 (见 C1)。

        它也不再占核资源: 零时长事件占资源只会让"同一时刻先处理 end 再处理 start"
        的次序出问题 (analysis/idle.py 里记过这个坑), 而排空语义不需要占核。

        想表达"波间全核对齐"(分段式执行) 用 ModelOptions.barriers, 见 builders/
        barriers.py —— 那是编排选择, 和这里的排空栅栏是两回事。
        """
        stages = {st for _role, sts in self.DRAIN_STAGES for st in sts}
        deps = tuple(ev.name for ev in self.events
                     if str(ev.meta.get("stage", "")) in stages)
        return (self._event("moe_stage_done", (), 0.0, deps=deps,
                            meta={"stage": "moe_stage_done"}),)