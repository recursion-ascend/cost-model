"""第 4 层: URMA Layered 路径建图器 — 接收/批量 PUT combine (topo_urma=True).

URMA Layered 路径事件图构建 (MegaMoeLayered, topoType=URMA).

  Layered 路径结构:
    宏 Wave = 连续专家范围, AIV1 程序序 = recv(w+1) → combine(w)
    接收: 按 relay 通道扫专家 → mask/count → 本地拷贝或批量 GET
    聚合: 按 dst 通道归属核, 批量 PUT (跨专家积累, 满批/波尾提交)

建模边界:
  * 单 Server (serverNum=1): 全部 src rank 直连通道; 跨 Server 一级中继 PUT
    (BuildDispatchRelayQueues/SendDispatchRelayQueues) 未建模。
  * 前导 (SendMaskCal/量化本地 token/建队列/CrossRankSync/PrepareDispatch/
    PublishExpertTokenCount) 不入图 — 与 MTE 模型前导退场口径一致; 远端就绪
    由每批一次 flag 窗口轮询开销承载, 轮询重试 (跨 rank 偏斜) 不建模。
  * flag 轮询窗口数按 1 窗/批 近似 (token id 分布未建模; B=64 实测全命中窗 0)。
  * mask 扫描只计 mask 槽字节流量; 标量/GatherMask 开销无实测常数, 未建模。
  * PUT (WriteNni) 复用 GET (ReadNbi) 实测常数 (fab probe 仅测 GET, assumed)。
  * 并发域: 每 rank 通道并发 = world-1, fab probe 实测域为 3 流 (保持率 0.96);
    world ≥ 5 的并发外推未验证。
  * maxOutputSize 截断 (kernel 行上界钳制) 不建模。
"""
from __future__ import annotations

from typing import List, Tuple

from .base import EventBuilderBase
from .comm import UrmaTransport
from .context import BuildContext
from .gmm1 import add_gmm1_wave
from .gmm2 import add_gmm2_wave
from ..config.hardware import KernelConfig
from ..config.policy import InstancePolicy
from ..costs import UrmaMechanisticLatency
from ..shape import BlockCursor, CursorTrace, MegaMoeShape, ModelOptions
from ..planning.waves import Wave

_ALIGN_32 = 32
#: 原先这里写死 2048 并在注释里声明"= URMA_FLAG_WINDOW_TOKENS x URMA_FLAG_BYTES" ——
#: 复制出来的派生值, 改常数不会跟着变。现在从公式容器取 (UrmaMechanisticLatency
#: 的 flag_window_bytes), 它是某实现的选择, 可覆盖。


class LayeredEventBuilder(EventBuilderBase):
    """topo_urma=True 的建图器: 复用 GMM1/ACT/GMM2/尾段, 替换 dispatch/combine."""

    def __init__(self, costs, options: ModelOptions):
        super().__init__(costs, options)
        self.combine_mode = "layered"
        self.urma: UrmaMechanisticLatency = (
            costs.urma_mechanistic if getattr(costs, "urma_mechanistic", None) is not None
            else UrmaMechanisticLatency())

    # ---- 程序序链: AIV1 同核事件按 kernel 调用序串联 ----
    # _event 返回带 R{rank}. 前缀的实名; aiv1_last[core] 持有该核上一事件实名。

    # ---- 主入口 ----

    def build(self, shape: MegaMoeShape,
              waves: List[Wave]) -> Tuple[List, List[CursorTrace]]:
        km = shape.kernel if shape.kernel is not None else KernelConfig()
        core_assign = getattr(shape, "core_assignment", None)
        policy = shape.policy if shape.policy is not None else InstancePolicy()
        TILE_M = km.tile_m
        TILE_N = km.tile_n
        ACT_HALF = km.activation_n_half
        self._order = 0
        self._rank = shape.rank_id
        p = shape.aic_num
        c = self.costs
        ctx = BuildContext.fresh(p, BlockCursor(p, 0))
        cursor = ctx.cursor
        transport = UrmaTransport(self.urma)
        self.combine_backend = transport.combine

        shared_gates = self._build_shared_expert(shape, km, ACT_HALF, TILE_M, TILE_N, p, c)

        # receive(W0): kernel ProcessMoeExpertWave 前置 (PrepareDispatch 后立即接收)
        if waves:
            transport.dispatch.add_wave(self, ctx, waves[0], shape, km, p, c,
                                        policy, shared_gates)

        # ---- 宏 Wave 主循环: recv(w+1) → GMM1/ACT(w) → GMM2(w) → combine(w) ----
        for iteration, w in enumerate(waves):
            if iteration + 1 < len(waves):
                transport.dispatch.add_wave(self, ctx, waves[iteration + 1],
                                            shape, km, p, c, policy, None)

            cursor_before = cursor.start
            add_gmm1_wave(self, ctx, w, shape, km, p, c, core_assign, policy,
                          TILE_M, TILE_N, ACT_HALF)
            cursor_after_gmm1 = cursor.start
            add_gmm2_wave(self, ctx, w, shape, km, p, c, core_assign, policy,
                          TILE_M, TILE_N, ACT_HALF, iteration)
            cursor_after_gmm2 = cursor.start
            transport.combine.flush_wave(self, ctx, w, shape, km, p)

            has_next = iteration + 1 < len(waves)
            resonance = (has_next and cursor_after_gmm2 == cursor_before
                         and cursor_after_gmm1 != cursor_before)
            if resonance and policy.cursor_resonance_fix:
                cursor.set(cursor_after_gmm1)

            self.cursor_trace.append(CursorTrace(
                iteration=iteration, gmm1_wave=w.index,
                cursor_before_gmm1=cursor_before,
                cursor_after_gmm1=cursor_after_gmm1,
                gmm2_wave=w.index,
                cursor_after_gmm2=cursor_after_gmm2,
                resonance_fix_applied=resonance,
                cursor_after_fix=cursor.start))

        # ---- 完成事件 + 尾段链: 都在全部 stage 事件之后; 排空先建, 尾段的门就是它 ----
        drains = self._add_completion(p)
        if waves:
            self._add_epilogue(shape, km, ACT_HALF, p, c, drains)

        # ---- 守恒校验 (与 MTE 建图器同口径) ----
        missing = []
        for w in waves:
            for sl in w.slices:
                fg = sl.row_begin // TILE_M
                for lg in range(sl.m_groups):
                    if (sl.expert, fg + lg) not in ctx.dispatch_ready_event:
                        missing.append((sl.expert, fg + lg))
        if missing:
            raise ValueError(f"missing DispatchReady joins for groups: {missing[:12]}")

        return self.events, self.cursor_trace
