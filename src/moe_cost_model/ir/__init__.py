"""硬件事件图的类型化视图 (只读).

vocabulary  类型词表: Engine / Pipe / MemorySpace / TokenKind / DependencyKind /
            TransferDirection, 以及把现有字符串约定解析成类型的 classify_*。
            UNREPRESENTABLE 列出这个事件代数**表达不了**的东西 (异步发射时长、flag 身份、
            带宽争用、跨核 flag 等待), 写成明文而不是留白。
graph       EventGraphView: Event 列表 -> TaskNode / DependencyEdge / BufferLifetime,
            带 summary() (含"推不出来"的计数)。
"""
from .graph import (BufferLifetime, DependencyEdge, EventGraphView, TaskNode)
from .vocabulary import (CHANNEL_SEMANTICS, UNREPRESENTABLE, DependencyKind, Engine,
                         MemorySpace, Pipe, ResourceRef, TokenKind, TokenRef,
                         TransferDirection, TransferRef, classify_resource,
                         classify_token, classify_transfer)

__all__ = [
    "BufferLifetime", "CHANNEL_SEMANTICS", "DependencyEdge", "DependencyKind",
    "Engine", "EventGraphView", "MemorySpace", "Pipe", "ResourceRef", "TaskNode",
    "TokenKind", "TokenRef", "TransferDirection", "TransferRef", "UNREPRESENTABLE",
    "classify_resource", "classify_token", "classify_transfer",
]
