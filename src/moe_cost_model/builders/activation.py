"""第 4 层: activation stage — ACT tile 事件 (SwiGLU + MX 量化).

ACT 钉在配对 GMM1 同核的 AIV0 上, 数据经核内 UB 传递.
"""
from __future__ import annotations

from .context import BuildContext


def add_activation_tile(builder, ctx: BuildContext, w, si, sl, mg, nt, core,
                        m_rows, logical_n, global_group, gname) -> None:
    q_vec = (f"Q:vec0:c{core}", 1)
    aname = f"W{w.index}.E{sl.expert}.S{si}.act.m{mg}.n{nt}.c{core}"
    c = builder.costs
    builder._event(aname, (f"AIV0:{core}",),
                   c.activation_tile(m_rows, logical_n) + c.activation_ready_publish_us,
                   deps=[gname], acquires=(q_vec,), releases=(q_vec,),
                   meta={"stage": "activation", "wave": w.index,
                         "expert": sl.expert, "slice": si,
                         "mgroup": global_group, "ntile": nt,
                         "logical_n": logical_n, "core": core, "m_rows": m_rows})
    ctx.gmm1_act_history[core].append(aname)
    ctx.activation_ready.setdefault((sl.expert, global_group), []).append((nt, aname))
