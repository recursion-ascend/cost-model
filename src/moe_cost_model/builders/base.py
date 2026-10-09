"""第 4 层: 事件图构建公共基类 — IR/公共方法/共享专家/GMM1+ACT/GMM2+COMBINE/尾段.

MTE 与 URMA Layered 两个建图器共享本基类; 路径差异在各自子类.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..config.hardware import (
    BW_UNPERMUTE_AGG, T_CORE_SYNC_BARRIER_US, T_COUNTS_EXPORT_US,
    T_FINALIZE_US, T_OUTPUT_INIT_US, T_RANK_SYNC_RTT_US, ceil_div,
)
from ..config.stages import default_vocabulary
from ..scheduler.events import Event
from ..costs import DispatchDataLayout, PrimitiveCosts
from ..shape import BlockCursor, CursorTrace, MegaMoeShape, ModelOptions
from ..planning.waves import ExpertSlice, Wave, swizzle_coord


# =====================================================================
# Dispatch 段枚举 IR 
# =====================================================================


@dataclass(frozen=True)
class DispatchExpertIR:
    expert: int = 0
    dst_rank: int = 0
    row_begin: int = 0
    row_end: int = 0
    segments: Tuple[Tuple[int, int], ...] = ()   # (src_rank, rows) in execution order
    local_segments: int = 0
    remote_segments: int = 0
    rows: int = 0


@dataclass(frozen=True)
class DispatchCallIR:
    dst_rank: int = 0
    wave: int = 0
    aiv1: int = 0
    global_row_begin: int = 0
    global_row_end: int = 0
    experts: Tuple[DispatchExpertIR, ...] = ()


def build_dispatch_expert_ir(*, expert, dst_rank, source_counts, row_begin, row_end, layout):
    """Segments of one (core-local) expert row range, src-major order."""
    segs: List[Tuple[int, int]] = []
    cur = 0
    for src, cnt in enumerate(source_counts):
        nxt = cur + cnt
        lo = max(row_begin, cur)
        hi = min(row_end, nxt)
        if hi > lo:
            segs.append((src, hi - lo))
        cur = nxt
    local = [(s, r) for s, r in segs if s == dst_rank]
    remote = [(s, r) for s, r in segs if s != dst_rank]
    return DispatchExpertIR(
        expert=expert, dst_rank=dst_rank,
        row_begin=row_begin, row_end=row_end,
        segments=tuple(segs),
        local_segments=len(local), remote_segments=len(remote),
        rows=sum(r for _, r in segs))


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

    @staticmethod
    def _gmm1_device_scheduler_n(shape: MegaMoeShape, act_half: int = 2) -> int:
        return ceil_div(shape.hidden_dim, act_half)

    def _dispatch_call_ir(self, shape: MegaMoeShape, w: Wave, core: int) -> DispatchCallIR:
        if not shape.expert_source_tokens:
            raise ValueError("mechanistic Dispatch requires exact expert_source_tokens")
        layout = shape.dispatch_layout
        if layout is None:
            layout = DispatchDataLayout.from_hidden(shape.h)
        rel_begin, count = self._rotated_balanced_range(w.rows, core, shape.aic_num,
                                                        w.begin.global_row)
        core_global_begin = w.begin.global_row + rel_begin
        core_global_end = core_global_begin + count
        expert_irs = []
        if count:
            for sl in w.slices:
                overlap_begin = max(core_global_begin, sl.global_row_begin)
                overlap_end = min(core_global_end, sl.global_row_end)
                if overlap_begin >= overlap_end:
                    continue
                local_begin = sl.row_begin + (overlap_begin - sl.global_row_begin)
                local_end = sl.row_begin + (overlap_end - sl.global_row_begin)
                expert_irs.append(build_dispatch_expert_ir(
                    expert=sl.expert, dst_rank=shape.rank_id,
                    source_counts=shape.expert_source_tokens[sl.expert],
                    row_begin=local_begin, row_end=local_end, layout=layout))
        return DispatchCallIR(
            dst_rank=shape.rank_id, wave=w.index, aiv1=core,
            global_row_begin=core_global_begin, global_row_end=core_global_end,
            experts=tuple(expert_irs))

    # ---- 主入口 ----


    def _build_shared_expert(self, shape, km, ACT_HALF, TILE_M, TILE_N, p, c):
        """共享专家前半段: GMM1 + ACT tile, 排在 MoE dispatch 之前.

        返回门控事件名 (全部共享 ACT 完成), dispatch_call 依赖它.
        后半段 (共享 GMM2) 在尾段, 见 _add_shared_gmm2.
        """
        self.shared_act_by_group = {}
        if shape.shared_expert_num <= 0:
            return None
        m_tot_s = shape.token_num
        sched_n_s = self._gmm1_device_scheduler_n(shape, ACT_HALF)
        nt1_s = ceil_div(sched_n_s, TILE_N)
        mg_s = ceil_div(m_tot_s, TILE_M)
        g1s, as_events = [], []
        sc1 = BlockCursor(p, 0)
        for ti in range(mg_s * nt1_s):
            mg, nt = swizzle_coord(ti, mg_s, nt1_s, km.swizzle_offset, km.swizzle_direction)
            m_rows = min(TILE_M, m_tot_s - mg * TILE_M)
            logical_n = min(TILE_N, sched_n_s - nt * TILE_N)
            core = sc1.owners(1)[0]
            g1s.append(self._event(
                f"shared.gmm1.m{mg}.n{nt}", (self.options.role_resource("shared_gmm1", core),),
                c.gmm1_tile(m_rows, shape.h, logical_n),
                meta={"stage": "shared_gmm1", "m_rows": m_rows}))
            as_events.append(self._event(
                f"shared.act.m{mg}.n{nt}", (self.options.role_resource("shared_act", core),),
                c.activation_tile(m_rows, logical_n),
                deps=(g1s[-1],), meta={"stage": "shared_act", "m_rows": m_rows}))
            self.shared_act_by_group.setdefault(mg, []).append(as_events[-1])
        return self._event("shared.head_done", (), 0.0, deps=tuple(as_events),
                           meta={"stage": "shared_head_done"})

    def _add_shared_gmm2(self, shape, km, ACT_HALF, p, c, after: str) -> str:
        """共享专家后半段: GMM2 按 tile 建事件, 占 AIC 核, 时长取 GMM2 公式 (纯计算).

        每个 tile 依赖 after (尾段前序事件) 与本 m-group 的全部共享 ACT
        (GMM2 的输入是该 m-group 的 ACT 产出, 覆盖整个 K).
        返回汇合事件名 (全部共享 GMM2 tile 完成).
        """
        tile_m, tile_n = km.tile_m, km.tile_n
        k_gmm2 = shape.hidden_dim // ACT_HALF
        n_tiles = ceil_div(shape.h, tile_n)
        m_groups = ceil_div(shape.token_num, tile_m)
        cursor = BlockCursor(p, 0)
        tiles = []
        for ti in range(m_groups * n_tiles):
            mg, nt = swizzle_coord(ti, m_groups, n_tiles,
                                   km.swizzle_offset, km.swizzle_direction)
            m_rows = min(tile_m, shape.token_num - mg * tile_m)
            logical_n = min(tile_n, shape.h - nt * tile_n)
            core = cursor.owners(1)[0]
            tiles.append(self._event(
                f"shared.gmm2.m{mg}.n{nt}", (self.options.role_resource("shared_gmm2", core),),
                c.gmm2_tile(m_rows, k_gmm2, logical_n),
                deps=(after, *self.shared_act_by_group.get(mg, ())),
                meta={"stage": "shared_gmm2", "m_rows": m_rows, "mgroup": mg,
                      "ntile": nt, "logical_n": logical_n, "core": core}))
        return self._event("epilogue.shared_gmm2_done", (), 0.0, deps=tuple(tiles),
                           meta={"stage": "epilogue", "part": "shared_gmm2_done"})

    # stage 建图函数在同包各文件: gmm1.py / activation.py / gmm2.py, 通信与归约在
    # comm/{mte,urma}.py (dispatch 与 combine 都在那里, 没有 dispatch.py / combine.py) —
    # 状态经 BuildContext (context.py) 传递.

    # ---- 尾段 ----

    def _add_epilogue(self, shape, km, ACT_HALF, p, c, drains=()):
        """尾段链. 门是每核三引擎的排空节点 (drains), 不是"每核最后一个 COMBINE".

        内核里 WAIT_GMM_DRAIN (实测 trace 恰好 84 个 = 28 核 x AIC/AIV0/AIV1) 在
        WAIT_OUTPUT_CORE_SYNC → COUNTS_EXPORT 之前, 三个引擎都要各自排空。只等
        COMBINE 会漏掉 AIC 的 GMM2 尾块与 AIV0 的 ACT: bs=36 用例里核 2~5 的最后
        一个 ACT (209.6us) 晚于该核最后一个 COMBINE (149.9us), 只是恰好被核 0/1 的
        晚 COMBINE (231.0us) 盖住 —— 换个路由就会让尾段起得太早。
        """
        # C5: 五项固定开销可配置 (ModelOptions.epilogue_overheads); 缺省沿用实测常数
        oh = getattr(self.options, "epilogue_overheads", None)

        def _oh(field: str, fallback: float) -> float:
            if oh is None:
                return fallback
            v = getattr(oh, field, 0.0)
            return v if (v or getattr(oh, "literal", False)) else fallback

        counts_export = self._event("epilogue.counts_export", (),
                                    _oh("counts_export_us", T_COUNTS_EXPORT_US),
                                    deps=tuple(drains),
                                    meta={"stage": "epilogue", "part": "counts_export"})
        core_sync = self._event("epilogue.output_core_sync", (),
                                _oh("core_sync_us", T_CORE_SYNC_BARRIER_US),
                                deps=(counts_export,),
                                meta={"stage": "epilogue", "part": "output_core_sync"})
        tail_head = core_sync
        if shape.shared_expert_num > 0:
            tail_head = self._add_shared_gmm2(shape, km, ACT_HALF, p, c, after=core_sync)
        rank_sync = self._event("epilogue.output_rank_sync", (),
                                _oh("rank_sync_us", T_RANK_SYNC_RTT_US), deps=(tail_head,),
                                meta={"stage": "epilogue", "part": "output_rank_sync"})
        out_init = self._event("epilogue.output_buffer_init", (),
                               _oh("output_init_us", T_OUTPUT_INIT_US), deps=(rank_sync,),
                               meta={"stage": "epilogue", "part": "output_buffer_init"})
        unpermute_bytes = shape.token_num * (shape.topk * shape.h * 2 + shape.h * 2)
        if shape.shared_expert_num > 0:
            unpermute_bytes += shape.shared_expert_num * shape.token_num * shape.h * 2
        unpermute = self._event("epilogue.unpermute", (), unpermute_bytes / BW_UNPERMUTE_AGG,
                                deps=(out_init,), meta={"stage": "epilogue", "part": "unpermute"})
        self._event("epilogue.finalize", (), _oh("finalize_us", T_FINALIZE_US),
                    deps=(unpermute,), meta={"stage": "epilogue", "part": "finalize"})

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