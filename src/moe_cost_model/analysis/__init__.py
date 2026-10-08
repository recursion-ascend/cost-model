"""第 6 层: 决策分析 — 设计空间扫描 / 关键路径归因 / 空闲核任务转移."""
from .idle import IdleReport, IdleSegment, idle_decomposition
from .critical_path import (bottleneck_report, critical_path_breakdown,
                            extract_critical_path, next_bottleneck,
                            resource_utilization, what_if)
from .design_space import compare_variants, design_space, format_design_space
from .stealing import idle_core_stealing

__all__ = [
    "bottleneck_report",
    "compare_variants",
    "design_space",
    "format_design_space",
    "critical_path_breakdown",
    "extract_critical_path",
    "IdleReport",
    "IdleSegment",
    "idle_decomposition",
    "idle_core_stealing",
    "next_bottleneck",
    "resource_utilization",
    "what_if",
]
