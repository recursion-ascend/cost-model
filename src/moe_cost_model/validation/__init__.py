"""校验层: 结构不变量 + 与实测 trace 的结构比对.

invariants  check_graph / check_adapter —— 用 IR 词表写的结构不变量 (与 kernel 无关)
trace       read_trace / read_run / read_run_config —— 读实测 trace (截断可恢复且会标记)
compare     compare_run —— 预测 DAG 与实测 trace 的结构比对 (先声明能比什么)
"""
from .compare import COMPARED_STAGES, RunComparison, StageComparison, compare_run
from .invariants import Violation, check_adapter, check_graph
from .trace import (NOT_MODELLED, STAGE_MAP, TraceEvent, TraceFile, read_run,
                    read_run_config, read_trace)

__all__ = [
    "COMPARED_STAGES", "NOT_MODELLED", "RunComparison", "STAGE_MAP", "StageComparison",
    "TraceEvent", "TraceFile", "Violation", "check_adapter", "check_graph",
    "compare_run", "read_run", "read_run_config", "read_trace",
]
