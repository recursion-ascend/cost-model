#!/usr/bin/env python3
"""把模型结果与一次实测 run 逐 stage 对比.

用法:
  python tools/compare_measured.py <run_dir> <scenario.toml> [--rank 0]

run_dir 需含 raw/tiling_rank*.bin 与 *_trace_rank*.json (Chrome trace)。
trace 里每个事件出现两次 (同 tid/ts/dur), 先去重。

对比口径: 时间原点取各自的首个 dispatch 事件; 执行时间记到最后一个 COMBINE 结束。
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from moe_cost_model import load_scenario, parse_tiling, simulate  # noqa: E402

# 模型 stage → 实测 trace 事件名前缀 (trace 按波加 ·w0/·w1 后缀)
# DISPATCH_SCHEDULE 是每 (核, 波) 的包络, 内含该核该波全部 XFER/LOCAL —— 单列一节比,
# 不能和模型的 dispatch_call 直接比。
STAGE_MAP = [
    ("dispatch", ("DISPATCH_XFER", "DISPATCH_LOCAL")),
    ("gmm1", ("GMM1",)),
    ("activation", ("ACT_QUANT",)),
    ("gmm2", ("GMM2",)),
    ("combine", ("COMBINE",)),
]


def load_trace(run: Path, rank: int):
    files = sorted(run.glob(f"*_trace_rank{rank}.json"))
    if not files:
        raise SystemExit(f"{run} 下没有 *_trace_rank{rank}.json")
    d = json.loads(files[0].read_text(encoding="utf-8"))
    tname = {e["tid"]: e["args"]["name"]
             for e in d["traceEvents"] if e.get("name") == "thread_name"}
    seen, out = set(), []
    for e in d["traceEvents"]:
        if e.get("ph") != "X":
            continue
        key = (e["tid"], e["ts"], e["dur"], e["name"])
        if key in seen:          # trace 每个事件写两遍
            continue
        seen.add(key)
        e["_core"] = int(tname[e["tid"]].split("-")[1])
        out.append(e)
    return out


def measured_stage(events, prefixes):
    sel = [e for e in events if any(e["name"].startswith(p) for p in prefixes)]
    if not sel:
        return None
    return {
        "a": min(e["ts"] for e in sel),
        "b": max(e["ts"] + e["dur"] for e in sel),
        "n": len(sel),
        "dur": [e["dur"] for e in sel],
        "cores": len({e["_core"] for e in sel}),
    }


def model_stage(rank_result, stage):
    sel = [e for e in rank_result["events"] if e.meta.get("stage") == stage]
    if not sel:
        return None
    return {
        "a": min(e.start_us for e in sel),
        "b": max(e.end_us for e in sel),
        "n": len(sel),
        "dur": [e.end_us - e.start_us for e in sel],
        "cores": len({e.resources[0] for e in sel if e.resources}),
    }


def merge_gmm2_parts(rank_result):
    """模型把 GMM2 拆成 head/tail 两个事件; 合成每 tile 的总时长以便与实测比."""
    by_tile = collections.defaultdict(float)
    for e in rank_result["events"]:
        if e.meta.get("stage") != "gmm2":
            continue
        k = (e.meta["expert"], e.meta["mgroup"], e.meta["ntile"], e.meta["core"])
        by_tile[k] += e.end_us - e.start_us
    return sorted(by_tile.values())


def pct(model, meas):
    return f"{(model - meas) / meas * 100:+6.1f}%" if meas else "     —"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ev = load_trace(args.run, args.rank)
    til = parse_tiling(args.run / "raw" / f"tiling_rank{args.rank}.bin")
    sc = load_scenario(args.scenario)
    res = simulate(sc)
    rr = res["rank_results"][args.rank]

    print("== 输入核对 (场景 vs tiling 真值) ==")
    for name, got, want in (("bs", sc.workload.tokens, til["bs"]),
                            ("h", sc.h, til["h"]),
                            ("hiddenDim", sc.hidden_dim, til["hidden"]),
                            ("topk", sc.workload.topk, til["topk"]),
                            ("每卡专家", sc.workload.local_experts, til["moeEpr"]),
                            ("卡数", sc.workload.world, til["ep"]),
                            ("aic", sc.aic_num, til["aic"]),
                            ("共享专家", sc.workload.shared_expert_num, til["shared"]),
                            ("mGroupsPerWave", rr["m_groups_per_wave"], til["mGroupsPerWave"])):
        print(f"  {name:16s} 场景 {str(got):>8s}   tiling {str(want):>8s}   "
              + ("一致" if str(got) == str(want) else "★ 不一致 ★"))

    m_origin = min(e["ts"] for e in ev if e["name"].startswith("DISPATCH"))
    m_end = max(e["ts"] + e["dur"] for e in ev if e["name"].startswith("COMBINE"))
    print("\n== 执行时间 (首个 dispatch → 末个 COMBINE) ==")
    print(f"  模型 {res['kernel_total_us']:9.3f} us   实测 {m_end - m_origin:9.3f} us   "
          f"{pct(res['kernel_total_us'], m_end - m_origin)}")

    # dispatch 包络: 每 (核, 波) 从 dispatch_call 起到该核该波最后一个 dispatch 止
    env = collections.defaultdict(lambda: [float("inf"), float("-inf")])
    for e in rr["events"]:
        if e.meta.get("stage") in ("dispatch_call", "dispatch"):
            k = (e.meta["core"], e.meta["wave"])
            env[k][0] = min(env[k][0], e.start_us)
            env[k][1] = max(env[k][1], e.end_us)
    m_env = statistics.median([v[1] - v[0] for v in env.values()]) if env else 0.0
    x_sch = [e["dur"] for e in ev if e["name"].startswith("DISPATCH_SCHEDULE")]
    if x_sch:
        print("\n== dispatch 包络 (每核每波; 实测 DISPATCH_SCHEDULE 内含 XFER/LOCAL) ==")
        print(f"  模型 {len(env):3d} 个 中位 {m_env:7.3f} us   实测 {len(x_sch):3d} 个 中位 "
              f"{statistics.median(x_sch):7.3f} us   {pct(m_env, statistics.median(x_sch))}")
    print("\n== 逐 stage (时刻相对各自的首个 dispatch) ==")
    print(f"  {'stage':14s} {'模型 起→止':>20s} {'实测 起→止':>20s} "
          f"{'模型 n/核':>10s} {'实测 n/核':>10s} {'单事件中位':>18s}")
    for stage, prefixes in STAGE_MAP:
        ms, xs = model_stage(rr, stage), measured_stage(ev, prefixes)
        if not ms or not xs:
            continue
        md = merge_gmm2_parts(rr) if stage == "gmm2" else ms["dur"]
        mm, xm = statistics.median(md), statistics.median(xs["dur"])
        print(f"  {stage:14s} {ms['a']:8.1f}→{ms['b']:8.1f}  "
              f"{xs['a']-m_origin:8.1f}→{xs['b']-m_origin:8.1f}  "
              f"{ms['n']:5d}/{ms['cores']:<4d} {xs['n']:5d}/{xs['cores']:<4d} "
              f"{mm:7.3f} / {xm:7.3f} {pct(mm, xm)}")
    print("\n  注: 模型的 GMM2 拆成 head/tail 两个事件, 表里已按 tile 合并;")
    print("      实测的 n 是去重后的事件数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
