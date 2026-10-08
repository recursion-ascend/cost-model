"""第 4 层: MTE 路径建图器 — dispatch 前瞻 + 波主循环 + GMM2 滞后补跑.

MTE 路径 (非 Layered).
各 stage 的建图函数在同包: gmm1.py / activation.py / gmm2.py; dispatch 与 combine 在
comm/peerwrite.py (没有 dispatch.py / combine.py 这两个文件)。本类只负责编排与共享状态 BuildContext.
"""
from __future__ import annotations

from typing import List, Tuple

from ..config.hardware import KernelConfig
from ..config.policy import InstancePolicy
from ..scheduler.events import Event
from ..shape import BlockCursor, CursorTrace, MegaMoeShape
from ..planning.waves import Wave
from .base import EventBuilderBase
from .comm import PeerWriteCombine, PeerWriteDispatch
from .context import BuildContext
from .gmm1 import add_gmm1_wave
from .gmm2 import add_gmm2_wave


class MteEventBuilder(EventBuilderBase):
    def build(self, shape: MegaMoeShape,
              waves: List[Wave]) -> Tuple[List[Event], List[CursorTrace]]:
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
        dispatch_backend = PeerWriteDispatch()
        self.combine_backend = PeerWriteCombine()

        shared_gates = self._build_shared_expert(shape, km, ACT_HALF, TILE_M, TILE_N, p, c)

        # ---- stage 波偏移 (InstancePolicy.effective_wave_offsets) ----
        # 以 GMM1 的波为锚: dispatch 超前 offs.dispatch 个波, GMM2 滞后
        # |offs.gmm2| 个波. 缺省由 la/lag 推导.
        offs = policy.effective_wave_offsets(shape.token_num)
        n_waves = len(waves)
        dispatched: set = set()
        gmm2_done: set = set()

        def _dispatch(di: int) -> None:
            if 0 <= di < n_waves and di not in dispatched:
                dispatch_backend.add_wave(self, ctx, waves[di], shape,
                                          km, p, c, policy, shared_gates)
                dispatched.add(di)

        # 初始预取: W0..W(offs.dispatch)
        for di in range(offs.dispatch + 1):
            _dispatch(di)

        # ---- 波主循环: GMM1 锚定迭代号 ----
        for iteration, w in enumerate(waves):
            if iteration > 0:
                _dispatch(iteration + offs.dispatch)

            cursor_before = ctx.cursor.start
            add_gmm1_wave(self, ctx, w, shape, km, p, c, core_assign, policy,
                          TILE_M, TILE_N, ACT_HALF)
            cursor_after_gmm1 = ctx.cursor.start

            gmm2_wave_idx = iteration + offs.gmm2
            if 0 <= gmm2_wave_idx < n_waves:
                add_gmm2_wave(self, ctx, waves[gmm2_wave_idx], shape, km, p, c,
                              core_assign, policy, TILE_M, TILE_N, ACT_HALF,
                              call_iteration=iteration)
                # 该波 GMM2 全部建完 -> 通知 combine 后端收口 (per_expert 粒度在这里
                # 发事件; per_tile 粒度已在 on_gmm2_tile 里逐 tile 发完, 这里是空操作)
                self.combine_backend.flush_wave(
                    self, ctx, waves[gmm2_wave_idx], shape, km, p)
                gmm2_done.add(gmm2_wave_idx)
            cursor_after_gmm2 = ctx.cursor.start

            has_next = iteration + 1 < len(waves)
            resonance = (has_next and cursor_after_gmm2 == cursor_before
                         and cursor_after_gmm1 != cursor_before)
            if resonance and policy.cursor_resonance_fix:
                ctx.cursor.set(cursor_after_gmm1)

            self.cursor_trace.append(CursorTrace(
                iteration=iteration, gmm1_wave=w.index,
                cursor_before_gmm1=cursor_before,
                cursor_after_gmm1=cursor_after_gmm1,
                gmm2_wave=(gmm2_wave_idx if 0 <= gmm2_wave_idx < n_waves else None),
                cursor_after_gmm2=cursor_after_gmm2,
                resonance_fix_applied=resonance,
                cursor_after_fix=ctx.cursor.start))

        # ---- dispatch 补齐 (防御性 sweep; 正常偏移下主循环已覆盖) ----
        for di in range(n_waves):
            _dispatch(di)

        # ---- GMM2 补跑: 主循环内没轮到的波, 按波序补齐 ----
        remaining = [k for k in range(n_waves) if k not in gmm2_done]
        for j, k in enumerate(remaining):
            add_gmm2_wave(self, ctx, waves[k], shape, km, p, c, core_assign,
                          policy, TILE_M, TILE_N, ACT_HALF,
                          call_iteration=n_waves + j)
            self.combine_backend.flush_wave(self, ctx, waves[k], shape, km, p)
            self.cursor_trace.append(CursorTrace(
                iteration=n_waves + j,
                gmm1_wave=None,
                cursor_before_gmm1=ctx.cursor.start,
                cursor_after_gmm1=ctx.cursor.start,
                gmm2_wave=waves[k].index,
                cursor_after_gmm2=ctx.cursor.start,
                resonance_fix_applied=False,
                cursor_after_fix=ctx.cursor.start))

        # ---- 完成事件 (每核三引擎排空) ----
        # 须在全部 stage 事件生成之后, 且先于尾段链: 尾段的门就是这些排空节点
        drains = self._add_completion(p)

        # ---- 尾段链 ----
        if waves:
            self._add_epilogue(shape, km, ACT_HALF, p, c, drains)

        # ---- 守恒校验 ----
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
