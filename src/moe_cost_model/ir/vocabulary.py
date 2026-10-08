"""硬件事件图的**类型词表**: 资源 / 内存空间 / 通路 / 令牌 / 边.

为什么要这一层 ——
`scheduler.Event` 已经能表达依赖、资源、计数信号量与字节申报, 但表达方式是**字符串约定**:

  "MTE2:c7"            一条 MTE2 搬运管道 (执行单元, 容量恒 1)
  "QUEUE:mte_aic:c7"   一个 L1 缓冲槽 (load 相位取、fix 相位还, 跨事件持有)
  "UB:gmm1act:c7"      一个 UB 槽 (GMM1 取、配对 ACT 还)
  "Q:aic:c7"           每核引擎队列 (独占核时是空约束, 容量恒 1)
  "gm_to_l1"           一条访存通路的名字, 方向与两端内存**藏在名字里**
  deps                 一条边; **数据依赖与程序序依赖无从区分**

约定能跑, 但它不可查询: 调度器要靠 `rsplit(":", 1)` 解核号 (engine.py 的 _core_of),
相位拆分要靠 `(同 stage, 同 core)` **猜**哪条边是程序序 (pipeline_expand 的
_drop_program_order), 访存方向只能靠 `k.endswith("gm_to_l1")` 这种判断
(analysis/design_space.py)。换一份 kernel 时, 这些约定没有一处会报错 —— 它们只会悄悄
对不上。

本模块把约定**提升为类型**, 并且只做这一件事:

  * 它是**只读视图**: 从现有 Event 的字段解析出类型化记录, 不改 Event, 不改调度。
    所以它对时长、事件名、Event.order 零影响 (判据: golden 39 个指纹不变)。
  * 它把**不可表达的东西写成明文**: 例如异步发射 (issue) 与执行 (execution) 现在只有
    一个 duration_us, 模型用"把事件拆成相位"来近似; 这不是能靠加字段解决的, 见
    UNREPRESENTABLE。写出来比沉默好 —— 沉默会被当成"已经建模了"。

搬运语义的真实来源 (仓内 kernel):
  MTE2   GM -> L1/UB 载入     stage/mega_moe_gmm1_activation.h 的 CopyIn 路径
  MTE3   UB -> GM 写出        stage/mega_moe_gmm2_combine.h 的 DataCopyPad
  FIXPIPE L0C -> UB/GM        blaze/epilogue/block_epilogue_activation_mx_quant.h
  跨卡    窗口读 / PUT        common/mega_moe_peermem.h, stage/mega_moe_layered_dispatch.h
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Engine(str, Enum):
    """执行角色. 一个 AI Core = 1 个 AIC (Cube) + 2 个 AIV (向量), 平台结构常数."""

    AIC = "AIC"
    AIV0 = "AIV0"
    AIV1 = "AIV1"


class Pipe(str, Enum):
    """核内的搬运/计算管道. 每核每种一条, 容量恒 1 —— 这是硬件事实, 不是参数.

    与 Engine 的区别: Engine 是"谁在算", Pipe 是"哪条通路在搬"。一个事件可以占着
    AIC 同时让 MTE2 搬下一块 (相位拆分表达的正是这个重叠)。
    """

    MTE2 = "MTE2"          # GM -> L1/UB
    MTE3 = "MTE3"          # UB -> GM
    FIXPIPE = "FIXPIPE"    # L0C -> UB/GM
    CUBE = "CUBE"          # MMAD
    VEC = "VEC"            # 向量


class MemorySpace(str, Enum):
    """内存空间. 容量常数见 config/hardware (TOTAL_L1_SIZE / TOTAL_UB_SIZE / ...)."""

    GM = "GM"              # 片外 HBM
    L1 = "L1"
    L0C = "L0C"
    UB = "UB"
    PEER_GM = "PEER_GM"    # 对端卡的 GM (跨卡窗口)


class TokenKind(str, Enum):
    """计数令牌的两类语义. 两类都用同一个 acquires/releases 机制, 所以必须分型.

    EXECUTION_UNIT  执行单元: 容量恒 1 的硬件事实 (MTE2 / FIXPIPE / MTE_AIV)。
    BUFFER_SLOT     缓冲槽: 容量是**编排选择** (L1 缓冲块数 / UB 槽数)。
    ENGINE_QUEUE    每核引擎队列: 在本模型里恒为空约束 (持核事件独占该核),
                    容量写死 1 且不设参数 —— 见 model.py 声明容量处的说明。
    """

    EXECUTION_UNIT = "execution_unit"
    BUFFER_SLOT = "buffer_slot"
    ENGINE_QUEUE = "engine_queue"


class DependencyKind(str, Enum):
    """边的种类. **当前 Event 不带这个字段**, 所以这里是"能从结构推出来的那部分".

    DATA         生产者产出被消费者读 (GMM1 -> ACT -> GMM2 -> COMBINE)
    READINESS    就绪标记 (dispatch_ready 这类零时长汇聚点) 带来的边
    PROGRAM_ORDER 同一个角色上的程序序 (AIV1 的 last-chain), 不是数据依赖
    PHASE        同一个 tile 拆相位后相位之间的边 (lg -> ld -> cb -> fix)
    BARRIER      栅栏 / 排空节点带来的边
    CREDIT       槽位归还形成的边 (gmm2_combine_credit)
    UNKNOWN      推不出来 —— 不猜。pipeline_expand 的 _drop_program_order 是**猜**的
                 (同 stage + 同 core), 本模块不复制那个猜测, 见 classify_dependency
    """

    DATA = "data"
    READINESS = "readiness"
    PROGRAM_ORDER = "program_order"
    PHASE = "phase"
    BARRIER = "barrier"
    CREDIT = "credit"
    UNKNOWN = "unknown"


class TransferDirection(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


#: 访存通路 -> (源, 目的, 方向, 协议)。方向与两端内存原先只藏在通路名里。
#: 协议: "mte" = 核内搬运指令; "window" = 跨卡窗口读写 (common/mega_moe_peermem.h);
#:       "urma" = Layered 的 Hcomm GET/PUT (stage/mega_moe_layered_dispatch.h)。
CHANNEL_SEMANTICS = {
    "gm_to_l1": (MemorySpace.GM, MemorySpace.L1, TransferDirection.READ, "mte"),
    "hbm_write": (MemorySpace.UB, MemorySpace.GM, TransferDirection.WRITE, "mte"),
    "combine_read": (MemorySpace.GM, MemorySpace.UB, TransferDirection.READ, "mte"),
    "dispatch_read": (MemorySpace.PEER_GM, MemorySpace.UB, TransferDirection.READ,
                      "window"),
    "dispatch_write": (MemorySpace.UB, MemorySpace.GM, TransferDirection.WRITE, "mte"),
}

#: 跨卡通路: fab_src:{rank} = 流量离开该卡, fab_dst:{rank} = 到达该卡。
#: 两条记的是**同一批字节**的两端 (builders/comm/peerwrite.py 同时申报), 所以汇总时不能相加。
_FAB = re.compile(r"^fab_(src|dst):(\d+)$")

#: 资源名 -> 引擎 (核号在后缀)。"AIC:c7" / "AIC:*" (晚绑定占位) 两种都要认。
_RES = re.compile(r"^(AIC|AIV0|AIV1):(c?\*|c?\d+)$")

#: 令牌名 -> (类型, 管道或缓冲名)。出处见本模块 docstring 的字符串约定清单。
_TOKENS = (
    (re.compile(r"^MTE2:c"), TokenKind.EXECUTION_UNIT, Pipe.MTE2, None),
    (re.compile(r"^FIXPIPE:c"), TokenKind.EXECUTION_UNIT, Pipe.FIXPIPE, None),
    (re.compile(r"^MTE_AIV:"), TokenKind.EXECUTION_UNIT, Pipe.MTE3, None),
    (re.compile(r"^QUEUE:mte_aic:"), TokenKind.BUFFER_SLOT, Pipe.MTE2, MemorySpace.L1),
    (re.compile(r"^QUEUE:mte_aiv:"), TokenKind.BUFFER_SLOT, Pipe.MTE3, MemorySpace.UB),
    (re.compile(r"^QUEUE:cube:"), TokenKind.BUFFER_SLOT, Pipe.CUBE, MemorySpace.L0C),
    (re.compile(r"^QUEUE:fix:"), TokenKind.BUFFER_SLOT, Pipe.FIXPIPE, MemorySpace.UB),
    (re.compile(r"^QUEUE:vec:"), TokenKind.BUFFER_SLOT, Pipe.VEC, MemorySpace.UB),
    (re.compile(r"^UB:"), TokenKind.BUFFER_SLOT, None, MemorySpace.UB),
    (re.compile(r"^Q:(aic|vec0|aiv1):"), TokenKind.ENGINE_QUEUE, None, None),
)

#: **这个事件代数表达不了的东西**。写成明文, 因为沉默会被当成"已经建模"。
UNREPRESENTABLE = {
    "issue_vs_execution":
        "异步发射与完成只有一个 duration_us (events.py), end = start + dur "
        "(engine.py)。模型用把一个 tile 拆成 lg/ld/cb/fix 四个相位事件来近似重叠 "
        "(builders/pipeline_expand.py); 真正的 issue_duration 需要发射开销这个物理量, "
        "模型没有标定它, 不能凭空给。",
    "flag_identity":
        "硬件 flag (SetFlag/WaitFlag/CrossCoreSetFlag) 没有对象: 三个 SyncLatency 字段 "
        "(config/pipeline.py, 缺省全 0) 把 flag 表达成**某条边上的延迟**, 既没有 flag "
        "身份也没有 set/wait 配对。kernel 侧有 20 多个 flag, 哪三个对应这三个字段并无记载 "
        "—— 这是一条真缺口, 不是命名问题。",
    "bandwidth_contention":
        "带宽域没有争用: channel_bytes 只做字节汇总, 不参与准入 (2026-10-03 停用速率"
        "服务器, 原因是两个 fab 常数尺度不同, 叠加会把争用计两遍)。所以 Transfer 有 "
        "bandwidth_domain 这个**标签**, 但没有共享速率的后果。",
    "cross_core_flag_wait":
        "跨核等待只以依赖边出现, 没有「等谁的哪个 flag」。idle.py 的 avoidable_idle_us "
        "因此只是上界。",
}


@dataclass(frozen=True)
class ResourceRef:
    """一个具体资源: 引擎 + 核号. 核号为 None = 晚绑定占位 (派发时才定)."""

    engine: Engine
    core: Optional[int]
    raw: str

    @property
    def late_bound(self) -> bool:
        return self.core is None


@dataclass(frozen=True)
class TokenRef:
    """一个计数令牌: 类型 + 它守的是哪条管道/哪块内存 + 核号."""

    kind: TokenKind
    raw: str
    pipe: Optional[Pipe] = None
    space: Optional[MemorySpace] = None
    core: Optional[int] = None
    count: int = 1

    @property
    def is_hardware_fact(self) -> bool:
        """执行单元的容量是硬件事实 (恒 1); 缓冲槽的容量是编排选择."""
        return self.kind is TokenKind.EXECUTION_UNIT


@dataclass(frozen=True)
class TransferRef:
    """一次搬运: 两端内存 + 方向 + 字节 + 协议 + 带宽域."""

    channel: str
    bytes: float
    src: Optional[MemorySpace] = None
    dst: Optional[MemorySpace] = None
    direction: Optional[TransferDirection] = None
    protocol: str = ""
    bandwidth_domain: str = ""
    #: 跨卡通路的那一端 (fab_src / fab_dst 的 rank)
    peer_rank: Optional[int] = None


def _core_of(text: str) -> Optional[int]:
    tail = text.rsplit(":", 1)[-1]
    if tail in ("*", "c*"):
        return None
    tail = tail[1:] if tail.startswith("c") else tail
    return int(tail) if tail.isdigit() else None


def classify_resource(raw: str) -> Optional[ResourceRef]:
    """资源名 -> ResourceRef; 认不出来返回 None (例如全局的 DISPATCH_COMM).

    返回 None 而不是猜: 认不出的资源交给调用方决定怎么办, 本模块不编造语义。
    """
    body = raw.split(".", 1)[-1]            # 去掉 rank 前缀 R0.
    mo = _RES.match(body)
    if not mo:
        return None
    return ResourceRef(engine=Engine(mo.group(1)), core=_core_of(body), raw=raw)


def classify_token(raw: str, count: int = 1) -> Optional[TokenRef]:
    """令牌名 -> TokenRef; 认不出来返回 None."""
    body = raw.split(".", 1)[-1]
    for pattern, kind, pipe, space in _TOKENS:
        if pattern.match(body):
            return TokenRef(kind=kind, raw=raw, pipe=pipe, space=space,
                            core=_core_of(body), count=count)
    return None


def classify_transfer(channel: str, nbytes: float) -> TransferRef:
    """通路名 + 字节 -> TransferRef. 未知通路保留名字, 两端留 None (不猜)."""
    body = channel.split(".", 1)[-1]
    mo = _FAB.match(body)
    if mo:
        side, rank = mo.group(1), int(mo.group(2))
        return TransferRef(
            channel=channel, bytes=float(nbytes),
            src=MemorySpace.UB if side == "src" else MemorySpace.PEER_GM,
            dst=MemorySpace.PEER_GM if side == "src" else MemorySpace.GM,
            direction=TransferDirection.WRITE if side == "src" else TransferDirection.READ,
            protocol="fabric", bandwidth_domain="fabric", peer_rank=rank)
    known = CHANNEL_SEMANTICS.get(body)
    if known is None:
        return TransferRef(channel=channel, bytes=float(nbytes))
    src, dst, direction, protocol = known
    domain = "hbm" if MemorySpace.GM in (src, dst) else "onchip"
    return TransferRef(channel=channel, bytes=float(nbytes), src=src, dst=dst,
                       direction=direction, protocol=protocol, bandwidth_domain=domain)
