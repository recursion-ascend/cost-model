"""六问决策分析层.

建模章程: 不追求 100% 复现硬件; 以可解释可验证的抽象支撑工程决策 ——
瓶颈在哪里 / 为什么形成 / 可以改什么 / 预计收益范围 / 下一个瓶颈 /
最小验证实验. 精度投入以"是否影响决策"为准绳, 证据等级与误差界显式声明.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dag import ScheduledEvent


# ---------------------------------------------------------------------------
# 证据等级表 (来自 12 数据集 × 4 rank 全量验证, 2026-09 eval)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    regime: str          # 适用域
    stage_error: str     # 各 stage 误差带
    wall_error: str      # 墙钟误差带
    level: str           # A(定量) / B(排序可信) / C(方向性) / D(失效)
    notes: str


EVIDENCE_TABLE: Tuple[Evidence, ...] = (
    Evidence("B<=128, 随机/受控路由 (random/asym/cyclic)",
             "DISP ±5%, GMM1 +4~10%(系统性高估), ACT ±1%, GMM2 ±2%",
             "-4% ~ -9% (系统性乐观)",
             "B",
             "相对排序与趋势可信; 墙钟绝对值需按域偏置校正"),
    Evidence("B<=128, hot 极端路由",
             "DISP -8~-12%, COMB -23~-32% (散射模式域外)",
             "-4% ~ -6%", "C",
             "COMB 仅方向性; 决策不依据 COMB 绝对值"),
    Evidence("B=1024, 随机路由",
             "ACT -14~15%, GMM2 -12~14%, COMB +114~246% (BW_SCATTER 单点标定域外)",
             "-17% ~ -19%", "D",
             "COMB 模型失效; 大 B 结论需先修 COMB chunk 流水机制"),
)


def evidence_for(bs: int, routing: str) -> Evidence:
    """按 (batch, 路由) 落证据等级."""
    if bs >= 1024:
        return EVIDENCE_TABLE[2]
    if routing == "hot":
        return EVIDENCE_TABLE[1]
    return EVIDENCE_TABLE[0]


# ---------------------------------------------------------------------------
# 关键路径与瓶颈归因
# ---------------------------------------------------------------------------

def extract_critical_path(scheduled: Sequence[ScheduledEvent]) -> List[ScheduledEvent]:
    """从最晚结束事件沿 critical_parent 回溯到根."""
    by_name = {e.name: e for e in scheduled}
    if not scheduled:
        return []
    leaf = max(scheduled, key=lambda e: (e.end_us, e.order, e.name))
    path: List[ScheduledEvent] = []
    cur: Optional[ScheduledEvent] = leaf
    seen = set()
    while cur is not None and cur.name not in seen:
        seen.add(cur.name)
        path.append(cur)
        cur = by_name.get(cur.critical_parent) if cur.critical_parent else None
    path.reverse()
    return path


def critical_path_breakdown(path: Sequence[ScheduledEvent]) -> Dict[str, object]:
    """关键路径按 stage 与机制归因; stage 时长含事件本体, 等待归到消费方."""
    by_stage: Dict[str, float] = {}
    by_mechanism: Dict[str, float] = {}
    wait_total = 0.0
    for ev in path:
        dur = ev.end_us - ev.start_us
        stage = str(ev.meta.get("stage", "other"))
        by_stage[stage] = by_stage.get(stage, 0.0) + dur
        if ev.critical_reason == "dependency":
            wait = max(0.0, ev.start_us - ev.dependency_ready_us)
            # dependency_wait 含程序序/数据等待, 均计入等待
            w = ev.dependency_wait_us
        elif ev.critical_reason.startswith("resource"):
            w = ev.resource_queue_us
        elif ev.critical_reason == "capacity":
            w = ev.capacity_wait_us
        elif ev.critical_reason == "channel":
            w = ev.channel_wait_us
        else:
            w = 0.0
        wait_total += w
        by_mechanism[ev.critical_reason.split(":")[0]] = \
            by_mechanism.get(ev.critical_reason.split(":")[0], 0.0) + w
    total = path[-1].end_us - path[0].start_us if path else 0.0
    work_total = total - wait_total
    return {
        "total_us": total,
        "work_us": work_total,
        "wait_us": wait_total,
        "by_stage": by_stage,
        "by_mechanism": by_mechanism,
        "top_events": sorted(
            (
                {"name": e.name, "stage": str(e.meta.get("stage", "other")),
                 "core": e.meta.get("core"), "expert": e.meta.get("expert"),
                 "span": e.end_us - e.start_us, "reason": e.critical_reason}
                for e in path
            ),
            key=lambda x: -x["span"],
        )[:8],
    }


def resource_utilization(rank_result: Dict[str, object]) -> List[Dict[str, object]]:
    """每核 busy/span 利用率, 降序 — 找负载不均."""
    busy: Dict[str, float] = rank_result.get("resource_busy", {})
    span: Dict[str, float] = rank_result.get("resource_span", {})
    rows = []
    for res, b in busy.items():
        sp = span.get(res, 0.0)
        rows.append({"resource": res, "busy_us": b, "span_us": sp,
                     "util": (b / sp) if sp > 0 else 0.0})
    rows.sort(key=lambda r: -r["busy_us"])
    return rows


def bottleneck_report(simulate_result: Dict[str, object]) -> Dict[str, object]:
    """Q1/Q2: 瓶颈在哪里 + 为什么形成."""
    ranks: Dict[int, Dict[str, object]] = simulate_result["rank_results"]
    slowest = simulate_result["slowest_rank"]
    rank = ranks[slowest]
    scheduled: List[ScheduledEvent] = rank["events"]
    path = extract_critical_path(scheduled)
    br = critical_path_breakdown(path)
    return {
        "slowest_rank": slowest,
        "rank_total_us": {r: ranks[r]["total_us"] for r in ranks},
        "critical_path": [e.name for e in path],
        "breakdown": br,
        "resources": resource_utilization(rank)[:12],
        "stage_busy": rank.get("stage_busy", {}),
    }


# ---------------------------------------------------------------------------
# 方案筛选 (Q3/Q4/Q5)
# ---------------------------------------------------------------------------

def what_if(
    simulate_fn,
    variants: Dict[str, object],
) -> Dict[str, Dict[str, float]]:
    """Q3/Q4: 逐方案重仿真.

    simulate_fn() -> baseline total_us; variants = {方案名: 零参可调用 -> total_us}.
    返回 {名: {total_us, delta_us}}.
    """
    base = simulate_fn()
    out = {"baseline(当前配置)": {"total_us": base, "delta_us": 0.0}}
    for name, fn in variants.items():
        total = fn()
        out[name] = {"total_us": total, "delta_us": total - base}
    return out


def next_bottleneck(simulate_fn, best_option) -> Dict[str, object]:
    """Q5: 应用最优方案后的新关键路径 (瓶颈迁移)."""
    sim = simulate_fn(best_option, full=True)
    return bottleneck_report(sim)
