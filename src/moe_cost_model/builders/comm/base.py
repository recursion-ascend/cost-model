"""通信协议接口定义.

DispatchTransport 契约: 为 wave 内每个 (expert, m-group) 在
ctx.dispatch_ready_event 登记就绪标记事件名, 下游 GMM1 依赖它;
行数守恒由实现自校验.

CombineTransport 契约: on_gmm2_tile 在 GMM2 tile 建好后逐个调用 (t 是该 tile
的行列范围, label 是它的名字片段),
flush_wave 在该波全部 GMM2 建完后调用一次; 实现自行决定
聚合粒度 (配对 tile / 批量 PUT).
"""
from __future__ import annotations


class DispatchTransport:
    """dispatch 后端接口: 把一个 wave 的 token 行搬运建成事件."""

    def add_wave(self, builder, ctx, w, shape, km, p, c, policy, shared_gates):
        raise NotImplementedError


class CombineTransport:
    """combine 后端接口: 为 GMM2 产出建聚合事件."""

    def on_gmm2_tile(self, builder, ctx, w, si, sl, t, label, ntile, core,
                     gname, global_group, call_iteration):
        raise NotImplementedError

    def flush_wave(self, builder, ctx, w, shape, km, p):
        pass
