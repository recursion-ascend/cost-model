"""Pipeline constraints: L0 同步/槽位, L1 队列深度, L2 信道, 相位速率.

设计原则:
  - 全部默认值 = 现行为 (回归安全): 同步延迟 0, 队列深度 1, 信道空, 速率 None
  - 数值必须带出处: 结构常数源自 kernel 工程, 硬件参数源自单点实测
  - from_tiling 从 tiling_rank*.bin 读 kernel 真实槽位/深度; 打点 bin 体积大不入库,
    故 parse_tiling 同时接受 tools/export_tiling.py 导出的 tiling_rank*.json 旁置文件
    (同一批整数, 几百字节, 可入库), 并在 .bin 缺失时自动回落到同名 .json
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# tiling_rank*.bin 解析 (MegaMoeTilingData)
# 字段偏移随 kernel struct 变化必须同步.
# ---------------------------------------------------------------------------


#: parse_tiling 返回的字段名; JSON 旁置文件必须恰好含这些键.
TILING_FIELDS = (
    "combineSyncSlotCountPerExpert", "groupedMatmulMode",
    "moeEpr", "bs", "h", "hidden", "ep", "topk", "aic", "aiv", "shared",
    "mGroupsPerWave",
    "dispatchRouteItemsPerBatch", "dispatchRouteBatches", "dispatchBufferCount",
    "sendMaskBufferCountWithExtra", "sendMaskBufferCountWithoutExtra",
    "unpermuteTokensPerBatch", "unpermuteInputBufferCount",
)


def resolve_tiling_path(path) -> Path:
    """tiling 真值的实际文件.

    打点工件 (raw/tiling_rank0.bin) 体积大、不入库, 所以场景文件照旧指向 .bin,
    而本函数在 .bin 不存在时回落到同目录同名的 .json 旁置文件 —— 后者由
    tools/export_tiling.py 在采集机上导出一次并入库, 之后 CI 与干净克隆都能
    跑 tiling 真值护栏。两者都没有才报错, 且提示怎么生成。
    """
    p = Path(path)
    if p.exists():
        return p
    # 候选旁置文件: 同目录同名; 以及 bin 在 raw/ 下时的上一级 —— /data/*/raw/ 整个
    # 被 gitignore, 所以入库的旁置文件落在 run 根目录, 场景文件无须改 path。
    cands = [p.with_suffix(".json")]
    if p.parent.name == "raw":
        cands.append(p.parent.parent / p.with_suffix(".json").name)
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"tiling 真值不存在: {p}\n"
        f"也没有旁置文件 ({', '.join(str(c) for c in cands)})。\n"
        f"打点 bin 不入库 (.gitignore: /data/*/raw/); 在有 bin 的采集机上跑一次\n"
        f"  python tools/export_tiling.py --all\n"
        f"把 tiling_rank0.json 导出并入库即可。")


def parse_tiling(path) -> Dict[str, int]:
    """解析 tiling 真值 -> kernel 实际参数与缓冲槽位配置.

    path 可以是 tiling_rank*.bin (打点工件) 或 tiling_rank*.json (入库旁置文件);
    .bin 缺失时自动回落到同名 .json, 见 resolve_tiling_path.
    """
    path = resolve_tiling_path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        missing = [k for k in TILING_FIELDS if k not in data]
        if missing:
            raise ValueError(f"{path}: tiling 旁置文件缺字段 {missing}; "
                             f"用 tools/export_tiling.py 重新导出")
        return {k: int(data[k]) for k in TILING_FIELDS}
    raw = path.read_bytes()
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
# L1: 缓冲槽位容量 (计数信号量; 0 = 不启用, 走 ModelOptions 距离依赖)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferSlots:
    """gmm1→act 深度不用槽位字段: 它是 ModelOptions.links 里 gmm1->activation
    那条边的 depth (计数信号量, 见 config/links.py).

    以下为 tiling 真值 (from_tiling 填充); wave 路径中 gmm2/combine 同步 slot
    恒 0 (combineSyncSlotCountPerExpert 是 layered 内核专用字段).
    """
    gmm2_combine: Optional[int] = None  # GMM2→Combine 同步 slot (wave 路径无)
    # dispatch_window 的消费者: Scenario.build_costs 把它填进
    # DispatchMechanisticLatency.buffer_count (行级软流水槽数)
    dispatch_window: int = 0            # tiling dispatchBufferConfig.bufferCount
    # 以下已解析暂无消费者 (UNPERMUTE/sendMask 未接入, 防真值丢失):
    send_mask_with_extra: int = 0       # tiling sendMaskConfig.bufferCount (多专家核)
    send_mask_without_extra: int = 0    # (少专家核)
    unpermute_in: int = 0               # tiling unpermuteConfig.inputBufferCount


@dataclass(frozen=True)
class QueueDepths:
    """核内引擎队列深度 = **在飞上限 / 缓冲槽数**: 能攒多少笔待处理. 深度 1 = 串行.

    **这不是"同时能跑几笔"。** 两件事必须分开, 混在一起是 2026-10-05 那个 bug 的根源:

      队列深度 (本类)   能提前多少发起 —— 由缓冲槽数决定 (L1 槽、UB 槽), 是编排/
                        编译期选择, 所以它是旋钮。
      执行单元数        同时能跑几笔 —— **硬件事实**, 每核每种单元恒为 1
                        (一个 AIC 一条 MTE2、一条 Cube、一条 FixPipe;
                         一个 AIV 一条 MTE、一条 Vector)。不是旋钮。

    深度恒为 1 时两者重合, 所以缺省下看不出区别。深度 >1 时, 只有队列深度、没有执行
    单元约束, 等于给每个核凭空多出几条管道 —— 实测后果: 载入可无限并行, 墙钟低于
    带宽下界 26.6% (见 docs/design_space_gaps.md 的"下界与漏账")。

    执行单元在 builders/pipeline_expand.py 里按容量 1 的计数信号量给出:

      mte_aic  -> MTE2:c{core}            (.ld 相位; GM→L1)
      fix      -> FIXPIPE:c{core}         (fix 相位; L0C→UB/GM)
      mte_aiv  -> MTE_AIV:{eng}:c{core}   (AIV 侧 .ld; GM↔UB)
      cube     -> 无需另给: .cb 相位本身独占 AIC 核资源
      vec      -> 无需另给: ACT/COMBINE 主事件本身独占 AIV 核资源
    """
    mte_aic: int = 1    # AIC MTE1/MTE2 的 L1 缓冲槽数 (GM→L1 载入能攒几笔)
    cube: int = 1       # Cube MMAD 能攒几笔
    fix: int = 1        # FixPipe (L0C→UB) 能攒几笔
    vec: int = 1        # AIV Vector 能攒几笔
    mte_aiv: int = 1    # AIV MTE2/MTE3 (GM↔UB) 能攒几笔


@dataclass(frozen=True)
class PhaseRates:
    """可选相位速率. None = 该相位不单独计时.

    GMM1 的 load / cube 相位时长不在这里给: 取自 GMM 公式的 A 流与计算分解
    (载入带宽与 Cube 速率以公式为唯一来源), 与闭式时长同口径.
    """
    # FixPipe 带宽 (B/µs)。**当前不影响时长**: 结果写出 (数据释放事件) 按口径忽略
    # 不计, 见 builders/pipeline_expand.py 的 fix 相位。字段保留以备改口径。
    fix_bw_bytes_per_us: Optional[float] = None
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
    ) -> "PipelineConstraints":
        """从 tiling 真值构建.

        gmm1→act 深度不在 tiling 中 (kernel 结构常数), 走
        gmm1->activation 那条边的 depth 默认; gmm2/combine 同步 slot 在
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
        )
