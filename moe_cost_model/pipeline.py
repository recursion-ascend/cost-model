"""Pipeline constraints: L0 同步/槽位, L1 队列深度, L2 信道, 相位速率.

设计原则:
  - 全部默认值 = 现行为 (回归安全): 同步延迟 0, 队列深度 1, 信道空, 速率 None
  - 数值必须带出处: 结构常数源自 kernel 源码/tiling, 硬件参数源自单点实测
  - from_tiling 从 tiling_rank*.bin 读 kernel 真实槽位/深度
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from .dag import Channel

# ---------------------------------------------------------------------------
# tiling_rank*.bin 解析 (MegaMoeTilingData, mega_moe_tiling.h:162)
# 字段偏移随 kernel struct 变化必须同步.
# ---------------------------------------------------------------------------


def parse_tiling(path) -> Dict[str, int]:
    """解析 tiling_rank*.bin -> kernel 实际参数与缓冲槽位配置."""
    raw = Path(path).read_bytes()
    (moe_epr, bs, h, hidden, ep, _bpep, _mo, topk, aic, aiv) = struct.unpack_from("<10I", raw, 0)
    shared, = struct.unpack_from("<I", raw, 64)
    # dispatchBufferConfig @80: routeItemsPerBatch, routeBatchCount, bufferCount, copyBufferBytes
    disp_items, disp_batches, disp_bufs, _disp_bytes = struct.unpack_from("<4i", raw, 80)
    # sendMaskConfigWithExtraExpert @96, WithoutExtra @112
    sm_e_items, sm_e_batches, sm_e_bufs, _ = struct.unpack_from("<4i", raw, 96)
    sm_n_items, sm_n_batches, sm_n_bufs, _ = struct.unpack_from("<4i", raw, 112)
    # unpermuteConfigForFullTokenChunk @132: tokensPerBatch, inputBufferCount, ...
    up_tokens, up_bufs = struct.unpack_from("<2i", raw, 132)
    mgw, = struct.unpack_from("<I", raw, 204)
    combine_slots, = struct.unpack_from("<Q", raw, 72)   # layered 内核专用, wave 路径恒 0
    gmm_mode = raw[52]   # groupedMatmulMode: 0=A8W8-Z, 2=A8W8-NZ, 1/3/4=W4 系
    return {
        "combineSyncSlotCountPerExpert": combine_slots,
        "groupedMatmulMode": gmm_mode,
        "moeEpr": moe_epr, "bs": bs, "h": h, "hidden": hidden, "ep": ep,
        "topk": topk, "aic": aic, "aiv": aiv, "shared": shared,
        "mGroupsPerWave": mgw,
        "dispatchRouteItemsPerBatch": disp_items, "dispatchRouteBatches": disp_batches,
        "dispatchBufferCount": disp_bufs,
        "sendMaskBufferCountWithExtra": sm_e_bufs,
        "sendMaskBufferCountWithoutExtra": sm_n_bufs,
        "unpermuteTokensPerBatch": up_tokens, "unpermuteInputBufferCount": up_bufs,
    }


# ---------------------------------------------------------------------------
# L0: 同步延迟 (逐边, 默认 0 = 依赖边零成本的现行为)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncLatency:
    gmm1_act_handshake_us: float = 0.0   # AIC→AIV0 WaitForVector RTT (WAIT_GMM1_BUFFER median)
    act_gmm2_ready_us: float = 0.0       # ACT→GMM2 activationToGmm2Flag
    gmm2_combine_ack_us: float = 0.0     # GMM2→COMBINE slot 归还


# ---------------------------------------------------------------------------
# L1: 缓冲槽位容量 (令牌数; 0 = 不启用该令牌, 走 ModelOptions 距离依赖)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferSlots:
    """gmm1→act 深度不用槽位字段: 距离依赖由 ModelOptions.gmm1_activation_depth
    由 ModelOptions.gmm1_activation_depth 参数化 (距离依赖保序无环).

    以下为 tiling 真值 (from_tiling 填充); wave 路径中 gmm2/combine 同步 slot
    恒 0 (combineSyncSlotCountPerExpert 是 layered 内核专用字段).
    """
    gmm2_combine: Optional[int] = None  # GMM2→Combine 同步 slot (wave 路径无)
    # 已解析暂无消费者 (dispatch 机制模型/UNPERMUTE 未接入, 防真值丢失):
    dispatch_window: int = 0            # tiling dispatchBufferConfig.bufferCount
    send_mask_with_extra: int = 0       # tiling sendMaskConfig.bufferCount (多专家核)
    send_mask_without_extra: int = 0    # (少专家核)
    unpermute_in: int = 0               # tiling unpermuteConfig.inputBufferCount


@dataclass(frozen=True)
class QueueDepths:
    """核内引擎队列深度 (在飞上限). 深度 1 = 串行 = 现行为."""
    mte_aic: int = 1    # AIC MTE1/MTE2: GM→L1 载入在飞
    cube: int = 1       # Cube MMAD 在飞
    fix: int = 1        # FixPipe (L0C→UB) 在飞
    vec: int = 1        # AIV Vector 在飞
    mte_aiv: int = 1    # AIV MTE2/MTE3: GM↔UB 在飞


@dataclass(frozen=True)
class PhaseRates:
    """可选相位速率. None = 该相位时长折入承载闭式时长的相位 (BW 主导假设).

    提供速率后相位展开会把 load/cube/fix 拆开, 支持跨 tile 流水重叠.
    """
    cube_mac_per_us: Optional[float] = None       # AIC 立方计算速率 (MAC/µs)
    fix_bw_bytes_per_us: Optional[float] = None   # FixPipe 带宽 (B/µs)
    act_load_bw_bytes_per_us: Optional[float] = None  # ACT GM→UB 读带宽
    combine_load_bw_bytes_per_us: Optional[float] = None  # COMBINE GM 读带宽


# ---------------------------------------------------------------------------
# 总装
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConstraints:
    sync: SyncLatency = field(default_factory=SyncLatency)
    buffers: BufferSlots = field(default_factory=BufferSlots)
    queues: QueueDepths = field(default_factory=QueueDepths)
    phases: PhaseRates = field(default_factory=PhaseRates)
    channels: Tuple[Channel, ...] = ()   # 空 = L2 关闭

    def __post_init__(self) -> None:
        q = self.queues
        if q.mte_aic < 1 or q.cube < 1 or q.fix < 1 or q.vec < 1 or q.mte_aiv < 1:
            raise ValueError("queue depths must be >= 1")
        b = self.buffers
        if b.gmm2_combine is not None and b.gmm2_combine < 0:
            raise ValueError("gmm2_combine must be >= 0 when set")
        if b.dispatch_window < 0 or b.send_mask_with_extra < 0 \
                or b.send_mask_without_extra < 0 or b.unpermute_in < 0:
            raise ValueError("tiling buffer counts must be >= 0")

    @classmethod
    def from_tiling(
        cls,
        tiling: Dict[str, int],
        *,
        sync: Optional[SyncLatency] = None,
        queues: Optional[QueueDepths] = None,
        phases: Optional[PhaseRates] = None,
        channels: Tuple[Channel, ...] = (),
    ) -> "PipelineConstraints":
        """从 tiling 真值构建.

        gmm1→act 深度不在 tiling 中 (kernel 结构常数), 走
        ModelOptions.gmm1_activation_depth 默认; gmm2/combine 同步 slot 在
        wave 路径恒 0 (layered 专用), 不强行接线.
        """
        return cls(
            sync=sync or SyncLatency(),
            buffers=BufferSlots(
                gmm2_combine=None,
                dispatch_window=tiling.get("dispatchBufferCount", 0),
                send_mask_with_extra=tiling.get("sendMaskBufferCountWithExtra", 0),
                send_mask_without_extra=tiling.get("sendMaskBufferCountWithoutExtra", 0),
                unpermute_in=tiling.get("unpermuteInputBufferCount", 0),
            ),
            queues=queues or QueueDepths(),
            phases=phases or PhaseRates(),
            channels=channels,
        )


def default_channels(
    aic_num: int,
    *,
    bw_l1_gm: float,
    bw_scatter: float,
) -> Tuple[Channel, ...]:
    """L2 默认信道: 每核应得速率 = 闭式公式所用带宽, 聚合 = ×核数.

    gm_to_l1:  GMM1/GMM2 权重+激活 GM→L1 流量
    hbm_write: COMBINE 散射写
    无争用时事件速率 = 应得速率, 与闭式公式逐字节一致;
    争用 (并发超过聚合允许) 时降速/排队.
    """
    return (
        Channel("gm_to_l1", bw_total=bw_l1_gm * aic_num, ports=0, max_rate_per_event=bw_l1_gm),
        Channel("hbm_write", bw_total=bw_scatter * aic_num, ports=0, max_rate_per_event=bw_scatter),
    )
