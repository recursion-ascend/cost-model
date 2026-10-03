"""C5: 固定开销是"某一版实现的残留", 要能被算子工程师按自己的实现填.

分清两类:
  * 实现残留: 尾段五项固定耗时、每波每核的 dispatch 调用开销 —— 换一版 kernel 就变
  * 物理:     unpermute 的字节量 / 带宽, 搬运与计算公式
"""
import re

import moe_cost_model as m


def _run(local=3, hd=9216, call=0.0, **kw):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hd, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(t_call_oh_us=call)),
        p1_override=1, p2_override=1, topk=6,
        options=m.ModelOptions(**kw))["rank_results"][0]


def test_epilogue_overheads_default_to_the_measured_constants():
    assert _run()["dag_end_us"] == _run(epilogue_overheads=m.EpilogueOverheads())["dag_end_us"]


def test_epilogue_overheads_can_be_zeroed_for_a_physics_only_baseline():
    """literal=True 时五项按字面取 (含 0): 实测恰好少 7.2us = 1+2+2.2+1+1."""
    base = _run()["dag_end_us"]
    pure = _run(epilogue_overheads=m.EpilogueOverheads(literal=True))["dag_end_us"]
    assert abs((base - pure) - 7.2) < 1e-6


def test_epilogue_overheads_are_settable():
    base = _run()["dag_end_us"]
    slow = _run(epilogue_overheads=m.EpilogueOverheads(core_sync_us=10.0))["dag_end_us"]
    assert abs((slow - base) - 8.0) < 1e-6      # 2.0 -> 10.0


def test_call_overhead_is_once_per_wave_and_core_in_rows_mode():
    """"rows" 切法没有按核的调用结构 -> 不发 dispatch_call 事件, 开销走 once_per_core.

    并且这时**一个带核号的事件名都不剩** —— C1 的目标在这条路径上完全达成。
    """
    rr = _run(call=1.006, dispatch_partition="rows", dispatch_rows_per_item=16)
    evs = rr["events"]
    assert not [e for e in evs if e.meta.get("stage") == "dispatch_call"]
    assert not [e for e in evs if re.search(r"\.c\d+$", e.name)]
    charged = [(e.meta["wave"], e.resources[0]) for e in evs
               if e.meta.get("once_per_core_us")]
    assert charged and len(charged) == len(set(charged))     # 每 (波, 核) 只算一次
    worked = {(e.meta["wave"], e.resources[0]) for e in evs
              if e.meta.get("stage") == "dispatch"}
    assert set(charged) == worked                            # 搬过数据的都付过


def test_kernel_mode_keeps_the_call_event_for_trace_alignment():
    """"kernel" 切法保留 dispatch_call: 实测 trace 有 DISPATCH_SCHEDULE 包络,
    tools/compare_measured.py 按它对齐。"""
    evs = _run(call=1.006)["events"]
    assert len([e for e in evs if e.meta.get("stage") == "dispatch_call"]) == 56


def test_call_overhead_costs_the_same_either_way():
    """两种切法下调用开销对墙钟的影响一致 (本形状 +1.0us)."""
    for mode, rpi in (("kernel", 0), ("rows", 16)):
        a = _run(call=0.0, dispatch_partition=mode, dispatch_rows_per_item=rpi)
        b = _run(call=1.006, dispatch_partition=mode, dispatch_rows_per_item=rpi)
        assert 0.9 < b["dag_end_us"] - a["dag_end_us"] < 1.1, mode
