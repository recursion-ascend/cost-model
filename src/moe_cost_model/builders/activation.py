"""第 4 层: activation stage — ACT 事件 (SwiGLU + MX 量化).

ACT 固定在产它的 GMM1 同核的向量角色上 (L0C->UB 的 Fixpipe 是物理约束), 数据经核内
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
from .pipeline_expand import CH_ACT_READBACK, CH_HBM_WRITE


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

class ActBatcher:
    """按事件粒度攒 ACT. 粒度 1 时每来一项立刻发 (与合并前逐字节等价)."""

    def __init__(self, items_per_event: int, ub_depth: int,
                 epilogue_rows: int = 0, prefetch: bool = False):
        if items_per_event < 0:
            raise ValueError("activation 的事件粒度不能为负")
        if items_per_event > 1 and 0 < ub_depth < items_per_event:
            raise ValueError(
                f"activation 粒度 {items_per_event} 要求 UB 槽数 >= {items_per_event}, "
                f'但 StageLink("gmm1","activation").depth = {ub_depth} —— '
                "一个 ACT 要等 g 个 GMM1 的结果都在 UB 里, 槽不够会死锁。"
                "把 depth 调大或设 0 (不设限), 或把粒度调小。")
        self.cap = items_per_event
        # epilogue 的行块高度 (kernel 的 EPILOGUE_TILE_M)。0 = 不按行切。
        # prefetch 开启时它是 128 而不是 256, 于是一个 GMM1 tile 的 epilogue 分成两个
        # 行块各自执行 —— 事件数翻倍、每个半高 (见 config.hardware.epilogue_tile_m)。
        self.epilogue_rows = int(epilogue_rows)
        self.prefetch = bool(prefetch)
        self._pending: Dict[int, _Pending] = {}

    def add(self, builder, ctx, w, si, sl, t, label, ntile, core, global_group,
            gname, out_div: int = 1, ub_slot=None) -> None:
        out_begin, out_end = t.col_begin // out_div, t.col_end // out_div
        if self.cap == 1:
            _emit(builder, ctx, w, si, sl, core, global_group,
                  t.row_begin, t.row_end, t.rows, out_begin, out_end,
                  [gname], ub_slot, ntile, label=label, n_items=1,
                  epilogue_rows=self.epilogue_rows, prefetch=self.prefetch)
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
              n_items=cur.n_items, epilogue_rows=self.epilogue_rows,
              prefetch=self.prefetch)

    def flush_all(self, builder, ctx, w, si, sl) -> None:
        """切片收尾: 合并不跨切片 (ctx.activation_ready 以 (专家, m-group) 为键)."""
        for core in list(self._pending):
            self.flush_core(builder, ctx, w, si, sl, core)


def _emit(builder, ctx: BuildContext, w, si, sl, core, global_group,
          row_begin, row_end, rows, out_begin, out_end, gnames, ub_slot, ntile,
          *, label: Optional[str], n_items: int,
          epilogue_rows: int = 0, prefetch: bool = False) -> None:
    """发这一份 ACT 工作. epilogue 行块高度小于行范围时拆成多个事件.

    为什么拆: kernel 的 epilogue 按 EPILOGUE_TILE_M 行一块循环执行
    (stage/mega_moe_gmm1_activation.h:428-459 的 `for subOffset`), 每块跑完就
    `NotifyGmm2InputReady`。TopkWeightsPrefetch 把这个高度从 256 降到 128, 于是
    一个 GMM1 tile 的 epilogue 是两个前后相继的行块 —— 事件数、同步点数、以及
    GMM2 能多早看到前半行, 都跟着变。这是编译参数的结构后果, 不是时长系数。

    下游依赖不变: 通知用的 flag 下标仍是 `subMLoc / L1_TILE_M_256` (同文件 457),
    即 m-group; 本模型的 ``ctx.activation_ready`` 键也仍是 (专家, m-group),
    只是同一个键下多了一条行范围更窄的记录, GMM2 按行相交取到全部行块。
    """
    step = int(epilogue_rows) if epilogue_rows and epilogue_rows < rows else 0
    if not step:
        _emit_block(builder, ctx, w, si, sl, core, global_group,
                    row_begin, row_end, rows, out_begin, out_end, gnames,
                    ub_slot, ntile, label=label, n_items=n_items,
                    suffix="", release_ub=True, prefetch=prefetch)
        return
    starts = list(range(row_begin, row_end, step))
    for idx, rb in enumerate(starts):
        re_ = min(rb + step, row_end)
        _emit_block(builder, ctx, w, si, sl, core, global_group,
                    rb, re_, re_ - rb, out_begin, out_end, gnames,
                    ub_slot, ntile, label=label, n_items=n_items,
                    suffix=f".e{idx}", release_ub=(idx == len(starts) - 1),
                    prefetch=prefetch)


def _emit_block(builder, ctx: BuildContext, w, si, sl, core, global_group,
                row_begin, row_end, rows, out_begin, out_end, gnames, ub_slot,
                ntile, *, label: Optional[str], n_items: int, suffix: str,
                release_ub: bool, prefetch: bool) -> None:
    out_cols = out_end - out_begin
    # 令牌跟着角色走 (roles 可以把 activation 挪到 AIV1): 写死 Q:vec0 会让事件占着 AIV1
    # 的核却扣 AIV0 的队列 —— 见 config/roles.queue_token 的说明。
    q_vec = (builder.options.role_queue_token("activation", core), 1)
    # C1: 名字不带核号 (见 gmm1.py 的说明); 落哪个核由 resources + 共位约束决定
    tag = label if label is not None else f"m{row_begin}_{row_end}.n{out_begin}_{out_end}"
    aname = f"W{w.index}.E{sl.expert}.S{si}.act.{tag}{suffix}"
    c = builder.costs
    # ACT 的写出走 GM (StoreQuantOutput + StoreQuantScaleCompact, 各 m 次带 stride
    # 的 UB→GM 突发) —— 申报到写信道上, 28 核同时写的降速由速率服务器算出来。
    # 实测: 28 并发中位 3.899us vs 8~10 并发 3.554us (+9.7%, 同专家受控对比)。
    store_bytes = c.activation_store_bytes(rows, out_cols)
    channels = []
    if store_bytes:
        channels.append((CH_HBM_WRITE, store_bytes, float(BW_LOCAL_GM)))
    duration = c.activation_tile(rows, out_cols) + c.activation_ready_publish_us
    readback_bytes = 0.0
    if prefetch:
        # prefetch 路径: 输入不是 AIC 经 Fixpipe 放进 UB 的, 要自己从 GM 读回
        # (GMM1 输出 + 本行块的 topk 权重)。时长串行相加 —— kernel 用 MTE2_V 标志
        # 把搬运与计算隔开, 见 costs.AnalyticalActCosts.readback_us。
        if c.activation_readback_us is None or c.activation_readback_bytes is None:
            raise ValueError(
                "KernelConfig.topk_weights_prefetch=True 需要 costs 给出 "
                "activation_readback_us / activation_readback_bytes —— 这份 "
                "PrimitiveCosts 没有描述 prefetch 路径的读回。"
                "用 build_analytical_costs 构造, 或自己补上这两项; "
                "缺省不按'读回免费'算。")
        readback_bytes = float(c.activation_readback_bytes(rows, out_cols))
        duration += float(c.activation_readback_us(rows, out_cols))
        if readback_bytes:
            channels.append((CH_ACT_READBACK, readback_bytes, float(BW_LOCAL_GM)))
    # 粗粒度事件归还它覆盖的**每一个** GMM1 占下的 UB 槽 (计数信号量按次数归还)。
    # 拆成多个行块时只有最后一块归还 (槽是按 GMM1 事件占的, 不按行块)。
    rel = (q_vec,)
    if ub_slot is not None and release_ub:
        rel = (q_vec, (ub_slot[0], n_items))
    meta = {"stage": "activation", "wave": w.index,
            "expert": sl.expert, "slice": si,
            "mgroup": global_group, "ntile": ntile,
            "col_begin": out_begin, "col_end": out_end,
            "row_begin": row_begin, "row_end": row_end,
            "logical_n": out_cols, "core": core, "m_rows": rows,
            "gmm1_events_in_event": n_items,
            "store_bytes": store_bytes}
    if prefetch:
        # 只在 prefetch 下进 meta: 非 prefetch 路径没有这一项, 记个 0 等于声称
        # "量过, 是零"。
        meta["readback_bytes"] = readback_bytes
    builder._event(aname, (builder.options.role_resource("activation", core),),
                   duration,
                   deps=list(gnames), acquires=(q_vec,),
                   releases=rel,
                   channel_bytes=tuple(channels),
                   meta=meta)
    ctx.activation_ready.setdefault((sl.expert, global_group), []).append(
        ActRecord(row_begin, row_end, out_begin, out_end, aname))
