"""第 4 层: activation stage — ACT tile 事件 (SwiGLU + MX 量化).

ACT 钉在配对 GMM1 同核的 AIV0 上, 数据经核内 UB 传递; 行列范围与 GMM1 tile 相同.
"""
from __future__ import annotations

from .context import ActRecord, BuildContext


def add_activation_tile(builder, ctx: BuildContext, w, si, sl, t, label, ntile,
                        core, global_group, gname) -> None:
    q_vec = (f"Q:vec0:c{core}", 1)
    aname = f"W{w.index}.E{sl.expert}.S{si}.act.{label}.c{core}"
    c = builder.costs
    builder._event(aname, (f"AIV0:{core}",),
                   c.activation_tile(t.rows, t.cols) + c.activation_ready_publish_us,
                   deps=[gname], acquires=(q_vec,), releases=(q_vec,),
                   meta={"stage": "activation", "wave": w.index,
                         "expert": sl.expert, "slice": si,
                         "mgroup": global_group, "ntile": ntile,
                         "col_begin": t.col_begin, "col_end": t.col_end,
                         "row_begin": t.row_begin, "row_end": t.row_end,
                         "logical_n": t.cols, "core": core, "m_rows": t.rows})
    ctx.gmm1_act_history[core].append(aname)
    ctx.activation_ready.setdefault((sl.expert, global_group), []).append(
        ActRecord(t.row_begin, t.row_end, t.col_begin, t.col_end, aname))
