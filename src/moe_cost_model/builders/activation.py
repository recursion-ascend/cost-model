"""第 4 层: activation stage — ACT tile 事件 (SwiGLU + MX 量化).

ACT 钉在配对 GMM1 同核的 AIV0 上, 数据经核内 UB 传递; 行列范围与 GMM1 tile 相同.
"""
from __future__ import annotations

from ..config.hardware import BW_LOCAL_GM
from .context import ActRecord, BuildContext
from .pipeline_expand import CH_HBM_WRITE


def add_activation_tile(builder, ctx: BuildContext, w, si, sl, t, label, ntile,
                        core, global_group, gname, out_div: int = 1,
                        ub_slot=None) -> None:
    """out_div: GMM1 tile 的列数 -> ACT 输出列数的缩减比。

    交织路径 (KernelConfig.gmm1_interleaved) 下 gate/up 在 tile 内按列交织, 一个宽
    tileN 的 GMM1 tile 只产出 tileN/activation_n_half 个输出列 (kernel 的
    epilogueN = N/ACTIVATION_N_HALF, epilogueNLoc = nLoc/ACTIVATION_N_HALF)。
    ACT 的列区间必须记在**输出列空间**里, 否则 GMM2 的 K 段映射会错。
    """
    out_cols = t.cols // out_div
    out_begin = t.col_begin // out_div
    out_end = t.col_end // out_div
    q_vec = (f"Q:vec0:c{core}", 1)
    aname = f"W{w.index}.E{sl.expert}.S{si}.act.{label}.c{core}"
    c = builder.costs
    # ACT 的写出走 GM (StoreQuantOutput + StoreQuantScaleCompact, 各 m 次带 stride
    # 的 UB→GM 突发) —— 申报到写信道上, 28 核同时写的降速由速率服务器算出来。
    # 实测: 28 并发中位 3.899us vs 8~10 并发 3.554us (+9.7%, 同专家受控对比)。
    store_bytes = c.activation_store_bytes(t.rows, out_cols)
    ch_bytes = ((CH_HBM_WRITE, store_bytes, float(BW_LOCAL_GM)),) if store_bytes else ()
    builder._event(aname, (f"AIV0:{core}",),
                   c.activation_tile(t.rows, out_cols) + c.activation_ready_publish_us,
                   deps=[gname], acquires=(q_vec,),
                   releases=(q_vec, ub_slot) if ub_slot else (q_vec,),
                   channel_bytes=ch_bytes,
                   meta={"stage": "activation", "wave": w.index,
                         "expert": sl.expert, "slice": si,
                         "mgroup": global_group, "ntile": ntile,
                         "col_begin": out_begin, "col_end": out_end,
                         "row_begin": t.row_begin, "row_end": t.row_end,
                         "logical_n": out_cols, "core": core, "m_rows": t.rows,
                         "store_bytes": store_bytes})
    ctx.activation_ready.setdefault((sl.expert, global_group), []).append(
        ActRecord(t.row_begin, t.row_end, out_begin, out_end, aname))
