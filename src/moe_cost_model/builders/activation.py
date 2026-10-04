"""第 4 层: activation stage — ACT 事件 (SwiGLU + MX 量化).

ACT 钉在产它的 GMM1 同核的向量角色上 (L0C->UB 的 Fixpipe 是物理约束), 数据经核内
UB 传递。

**事件粒度** (ModelOptions.granularity 的 "activation"): 一个 ACT 事件可以覆盖
多个 GMM1 事件的输出 —— 用更大的 UB 驻留换更少的同步点。合并只发生在
**同一核、同一 (专家, m-group)、行范围相同、列范围相邻** 的连续项之间:

  * 同核是 Fixpipe 的物理要求;
  * 同 (专家, m-group) 是因为 ``ctx.activation_ready`` 以它为键, 下游 GMM2 按这个键
    取依赖;
  * 行同列邻是因为 GMM2 按**连续列区间**挑 ACT (``col_begin < k_hi and col_end > k_lo``),
    不连续的并集表达不出来。

粒度 g 与 UB 槽数 (StageLink("gmm1","activation").depth) 有物理耦合: g 个 GMM1 事件
各占一个槽, 要等它们全到齐才发 ACT, 所以 depth 必须 >= g (或 0 = 不设限), 否则死锁。
这个检查在 ActBatcher 里强制。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..config.hardware import BW_LOCAL_GM
from .context import ActRecord, BuildContext
from .pipeline_expand import CH_HBM_WRITE


class _Pending:
    """一个核上正在攒的 ACT 项."""

    __slots__ = ("row_begin", "row_end", "out_begin", "out_end", "rows",
                 "gnames", "ub_slot", "ntile", "n_items", "group")

    def __init__(self, t, out_begin, out_end, gname, ub_slot, ntile, group):
        self.row_begin, self.row_end = t.row_begin, t.row_end
        self.rows = t.rows
        self.out_begin, self.out_end = out_begin, out_end
        self.gnames: List[str] = [gname]
        self.ub_slot = ub_slot
        self.ntile = ntile
        self.n_items = 1
        # 这一项属于哪个 (专家, m-group) —— 必须跟着项走, 不能用 flush 时刻的循环变量:
        # 一个切片里可以有多个 m-group, 换组会触发 flush, 那时循环变量已经是新组了。
        self.group = group

    def extend(self, t, out_end, gname) -> None:
        self.out_end = out_end
        self.gnames.append(gname)
        self.n_items += 1

    def can_extend(self, t, out_begin) -> bool:
        return (t.row_begin == self.row_begin and t.row_end == self.row_end
                and out_begin == self.out_end)

    @property
    def out_cols(self) -> int:
        return self.out_end - self.out_begin


class ActBatcher:
    """按事件粒度攒 ACT. 粒度 1 时每来一项立刻发 (与攒批前逐字节等价)."""

    def __init__(self, items_per_event: int, ub_depth: int):
        if items_per_event < 0:
            raise ValueError("activation 的事件粒度不能为负")
        if items_per_event > 1 and 0 < ub_depth < items_per_event:
            raise ValueError(
                f"activation 粒度 {items_per_event} 要求 UB 槽数 >= {items_per_event}, "
                f'但 StageLink("gmm1","activation").depth = {ub_depth} —— '
                "一个 ACT 要等 g 个 GMM1 的结果都在 UB 里, 槽不够会死锁。"
                "把 depth 调大或设 0 (不设限), 或把粒度调小。")
        self.cap = items_per_event
        self._pending: Dict[int, _Pending] = {}

    def add(self, builder, ctx, w, si, sl, t, label, ntile, core, global_group,
            gname, out_div: int = 1, ub_slot=None) -> None:
        out_begin, out_end = t.col_begin // out_div, t.col_end // out_div
        if self.cap == 1:
            _emit(builder, ctx, w, si, sl, core, global_group,
                  t.row_begin, t.row_end, t.rows, out_begin, out_end,
                  [gname], ub_slot, ntile, label=label, n_items=1)
            return
        cur = self._pending.get(core)
        if cur is not None and not cur.can_extend(t, out_begin):
            self.flush_core(builder, ctx, w, si, sl, core)
            cur = None
        if cur is None:
            self._pending[core] = _Pending(t, out_begin, out_end, gname,
                                           ub_slot, ntile, global_group)
        else:
            cur.extend(t, out_end, gname)
            cur = self._pending[core]
        cap = self.cap if self.cap > 0 else None
        if cap is not None and self._pending[core].n_items >= cap:
            self.flush_core(builder, ctx, w, si, sl, core)

    def flush_core(self, builder, ctx, w, si, sl, core) -> None:
        cur = self._pending.pop(core, None)
        if cur is None:
            return
        _emit(builder, ctx, w, si, sl, core, cur.group,
              cur.row_begin, cur.row_end, cur.rows, cur.out_begin, cur.out_end,
              cur.gnames, cur.ub_slot, cur.ntile, label=None,
              n_items=cur.n_items)

    def flush_all(self, builder, ctx, w, si, sl) -> None:
        """切片收尾: 攒批不跨切片 (ctx.activation_ready 以 (专家, m-group) 为键)."""
        for core in list(self._pending):
            self.flush_core(builder, ctx, w, si, sl, core)


def _emit(builder, ctx: BuildContext, w, si, sl, core, global_group,
          row_begin, row_end, rows, out_begin, out_end, gnames, ub_slot, ntile,
          *, label: Optional[str], n_items: int) -> None:
    out_cols = out_end - out_begin
    q_vec = (f"Q:vec0:c{core}", 1)
    # C1: 名字不带核号 (见 gmm1.py 的说明); 落哪个核由 resources + 共位约束决定
    tag = label if label is not None else f"m{row_begin}_{row_end}.n{out_begin}_{out_end}"
    aname = f"W{w.index}.E{sl.expert}.S{si}.act.{tag}"
    c = builder.costs
    # ACT 的写出走 GM (StoreQuantOutput + StoreQuantScaleCompact, 各 m 次带 stride
    # 的 UB→GM 突发) —— 申报到写信道上, 28 核同时写的降速由速率服务器算出来。
    # 实测: 28 并发中位 3.899us vs 8~10 并发 3.554us (+9.7%, 同专家受控对比)。
    store_bytes = c.activation_store_bytes(rows, out_cols)
    ch_bytes = ((CH_HBM_WRITE, store_bytes, float(BW_LOCAL_GM)),) if store_bytes else ()
    # 粗粒度事件归还它覆盖的**每一个** GMM1 占下的 UB 槽 (计数信号量按次数归还)。
    rel = (q_vec,)
    if ub_slot is not None:
        rel = (q_vec, (ub_slot[0], n_items))
    builder._event(aname, (builder.options.role_resource("activation", core),),
                   c.activation_tile(rows, out_cols) + c.activation_ready_publish_us,
                   deps=list(gnames), acquires=(q_vec,),
                   releases=rel,
                   channel_bytes=ch_bytes,
                   meta={"stage": "activation", "wave": w.index,
                         "expert": sl.expert, "slice": si,
                         "mgroup": global_group, "ntile": ntile,
                         "col_begin": out_begin, "col_end": out_end,
                         "row_begin": row_begin, "row_end": row_end,
                         "logical_n": out_cols, "core": core, "m_rows": rows,
                         "gmm1_events_in_event": n_items,
                         "store_bytes": store_bytes})
    ctx.activation_ready.setdefault((sl.expert, global_group), []).append(
        ActRecord(row_begin, row_end, out_begin, out_end, aname))


def add_activation_tile(builder, ctx: BuildContext, w, si, sl, t, label, ntile,
                        core, global_group, gname, out_div: int = 1,
                        ub_slot=None) -> None:
    """粒度 1 的直发路径 (保留给不经 ActBatcher 的调用方)."""
    out_begin, out_end = t.col_begin // out_div, t.col_end // out_div
    _emit(builder, ctx, w, si, sl, core, global_group, t.row_begin, t.row_end,
          t.rows, out_begin, out_end, [gname], ub_slot, ntile, label=label,
          n_items=1)
