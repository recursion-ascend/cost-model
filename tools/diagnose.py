#!/usr/bin/env python3
"""六问诊断报告: 瓶颈/归因/改什么/收益/下一瓶颈/最小验证实验.

用法: python tools/diagnose.py <run_dir> [--compare <run_dir2>]
一切结论附带证据等级与误差界 (analysis.EVIDENCE_TABLE).
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
REPO = PROJ.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(Path(__file__).parent))

from moe_cost_model import (
    ModelOptions, PipelineConstraints, PrimitiveCosts, QueueDepths,
    AnalyticalActCosts, AnalyticalCombineCosts, AnalyticalGmmCosts,
    DispatchMechanisticLatency, simulate_routing_counts,
    calc_m_groups_per_wave, parse_tiling, BW_L1_GM, T_COUNT_GATE,
)
from moe_cost_model.analysis import bottleneck_report, evidence_for, what_if, extract_critical_path, critical_path_breakdown
from routing import make_routing
from artifacts import read_prof_bin

import csv


def auto_config(run: Path):
    t = parse_tiling(run / "raw/tiling_rank0.bin")
    cfg = run / "config.json5"
    seed, routing_mode = 0, "random"
    if cfg.exists():
        s = cfg.read_text()
        m = re.search(r'"seed":\s*(\d+)', s)
        if m:
            seed = int(m.group(1))
        m = re.search(r'"routing":\s*"(\w+)"', s)
        if m:
            routing_mode = m.group(1)
    world, local = t["ep"], t["moeEpr"]
    case = dict(tokens=t["bs"], experts=local * world, topk=t["topk"],
                ep=world, seed=seed, routing=routing_mode)
    # C[dst][expert][src] — 与 simulate_routing_counts 的 routing_counts 同构
    C = [[[0] * world for _ in range(local)] for _ in range(world)]
    for s in range(world):
        for gid in make_routing(case, s).reshape(-1).tolist():
            C[gid // local][gid % local][s] += 1
    return t, C, routing_mode


def measured_wall(run: Path, rank: int = 0) -> float:
    reps = sorted(run.glob(f"raw/prof_rank{rank}_rep*.bin"))
    vals = []
    for p in reps or [run / f"raw/prof_rank{rank}.bin"]:
        if not p.exists():
            continue
        kern = []
        q = {}
        for cid, ev in read_prof_bin(p).items():
            for cyc, eid, pay in ev:
                if eid == 0x0001:
                    q.setdefault(cid, []).append(cyc)
                elif eid == 0x00FF and cid in q and q[cid]:
                    b = q[cid].pop(0)
                    kern.append((b, cyc))
        if kern:
            vals.append((max(e for _, e in kern) - min(b for b, _ in kern)) / 1000.0)
    return statistics.mean(vals) if vals else 0.0


def make_simulator(t, C):
    world = t["ep"]
    rc = tuple(tuple(tuple(row) for row in C_dst) for C_dst in C)

    def p1p2():
        mgw = t["mGroupsPerWave"]
        for p2 in (1, 2, 3):
            for p1 in (2, 3, 4, 6, 8):
                if calc_m_groups_per_wave(hidden_dim=t["hidden"], h=t["h"], aic_num=t["aic"], p1=p1, p2=p2) == mgw:
                    return p1, p2
        return 0, 0

    p1, p2 = p1p2()

    def sim(options=None, full=False, p1_ovr=None, p2_ovr=None):
        costs = PrimitiveCosts(
            dispatch_mechanistic=DispatchMechanisticLatency(
                begin_offset_us=tuple([0.0] * t["aic"])),
            gmm1_tile=AnalyticalGmmCosts().gmm1_tile,
            gmm2_tile=AnalyticalGmmCosts().gmm2_tile,
            activation_tile=AnalyticalActCosts().tile,
            combine_tile=AnalyticalCombineCosts(h=t["h"]).tile,
            count_table_prepare_us=T_COUNT_GATE)
        res = simulate_routing_counts(
            routing_counts=rc, token_num_per_rank=t["bs"], h=t["h"],
            hidden_dim=t["hidden"], aic_num=t["aic"], costs=costs,
            topk=t["topk"], shared_expert_num=t["shared"],
            options=options or ModelOptions(),
            p1_override=p1_ovr if p1_ovr is not None else p1,
            p2_override=p2_ovr if p2_ovr is not None else p2)
        return res if full else res["kernel_total_us"]

    return sim, (p1, p2)


def fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"


def report(run: Path, compare: Path = None):
    t, C, routing = auto_config(run)
    ev = evidence_for(t["bs"], routing)
    sim, (p1, p2) = make_simulator(t, C)
    base_res = sim(None, full=True)
    rep = bottleneck_report(base_res)
    wall_model = rep["rank_total_us"][rep["slowest_rank"]]
    bd = rep["breakdown"]

    print(f"# 六问诊断: {run.parent.name}/{run.name}")
    print(f"工况: bs={t['bs']} h={t['h']} topk={t['topk']} routing={routing} "
          f"mgw={t['mGroupsPerWave']} (p1={p1},p2={p2})\n")

    # 证据声明
    print("## 证据等级 (适用域声明)")
    print(f"- 域: {ev.regime}")
    print(f"- Stage 误差: {ev.stage_error}")
    print(f"- 墙钟误差: {ev.wall_error}")
    print(f"- 等级: {ev.level} — {ev.notes}\n")

    # Q1 瓶颈在哪里
    print("## Q1 瓶颈在哪里")
    print(f"- 最慢 rank: {rep['slowest_rank']} (rank 墙钟: "
          + ", ".join(f"r{r}={v:.0f}µs" for r, v in sorted(rep['rank_total_us'].items())) + ")")
    print(f"- 关键路径 {bd['total_us']:.0f}µs = 计算 {bd['work_us']:.0f} + 等待 {bd['wait_us']:.0f}")
    stage_items = sorted(bd["by_stage"].items(), key=lambda x: -x[1])
    print("- 路径构成: " + ", ".join(f"{k}={v:.0f}µs({100*v/bd['total_us']:.0f}%)" for k, v in stage_items))
    mech_items = sorted(bd["by_mechanism"].items(), key=lambda x: -x[1])
    if mech_items:
        print("- 等待机制: " + ", ".join(f"{k}={v:.0f}µs" for k, v in mech_items))
    res = rep["resources"]
    if res:
        aic = [r for r in res if r["resource"].startswith("AIC:")]
        if aic:
            hot, cold = aic[0], aic[-1]
            print(f"- 核负载: 最热 {hot['resource']} busy={hot['busy_us']:.0f}µs "
                  f"vs 最冷 {cold['resource']} busy={cold['busy_us']:.0f}µs "
                  f"(不均衡 {100*(1-cold['busy_us']/max(hot['busy_us'],1e-9)):.0f}%)")
    print()

    # Q2 为什么形成
    print("## Q2 为什么形成")
    for e in bd["top_events"][:5]:
        print(f"- {e['name'][:64]}  span={e['span']:.1f}µs  机制={e['reason']}")
    print()
    # Q3/Q4 改什么 + 收益范围
    print("## Q3/Q4 可以改什么 / 预计收益范围")
    variant_kwargs = {
        "mgw 扫描 p1=2": dict(p1_ovr=2, p2_ovr=1),
        "mgw 扫描 p1=4": dict(p1_ovr=4, p2_ovr=1),
        "mgw 扫描 p1=8": dict(p1_ovr=8, p2_ovr=1),
        "gmm1→act 深度=2": dict(options=ModelOptions(gmm1_activation_depth=2)),
        "kL1=256 显式": dict(options=ModelOptions(gmm2_kl1=256)),
    }
    if t["bs"] >= 1024:
        variant_kwargs["pipeline+信道"] = dict(options=ModelOptions(
            pipeline=PipelineConstraints(queues=QueueDepths(mte_aic=2, cube=2, fix=2, vec=2, mte_aiv=2))))
    variants = {name: (lambda kw=kw: sim(**kw)) for name, kw in variant_kwargs.items()}
    wi = what_if(lambda: sim(), variants)
    # 误差界 (来自证据表): 相对比较时系统性偏置抵消, 残差取 stage 误差量级
    err_band = 0.05 if ev.level in ("A", "B") else 0.15
    for name, r in sorted(wi.items(), key=lambda x: x[1]["delta_us"]):
        d = 100 * r["delta_us"] / wall_model
        lo, hi = d - 100 * err_band, d + 100 * err_band
        print(f"- {name:<22} 墙钟 {r['total_us']:.0f}µs  Δ={fmt_pct(d)} (收益范围 {lo:+.1f}%~{hi:+.1f}%)")
    print()

    # Q5 下一个瓶颈
    best_name = min((n for n in wi if n != "baseline(当前配置)"),
                    key=lambda n: wi[n]["delta_us"], default=None)
    if best_name and wi[best_name]["delta_us"] < 0:
        nxt = bottleneck_report(sim(full=True, **variant_kwargs[best_name]))
        nbd = nxt["breakdown"]
        stage_items = sorted(nbd["by_stage"].items(), key=lambda x: -x[1])
        print("## Q5 下一个瓶颈")
        print(f"- 应用 [{best_name}] 后关键路径 {nbd['total_us']:.0f}µs, 构成: "
              + ", ".join(f"{k}={v:.0f}µs({100*v/nbd['total_us']:.0f}%)" for k, v in stage_items[:4]))
        print()


    # Q6 最小验证实验
    print("## Q6 最小验证实验")
    if best_name and wi[best_name]["delta_us"] < 0:
        d = 100 * wi[best_name]["delta_us"] / wall_model
        lo, hi = d - 100 * err_band, d + 100 * err_band
        print(f"- 实验: 同工况重跑, 修改 {best_name} 对应的运行参数")
        print(f"- 预期: 墙钟变化落在 [{lo:+.1f}%, {hi:+.1f}%] 内 → 模型在该决策点可信")
        print(f"- 判据: 实测变化出界 → 用 tools/validate_waits.py 复核该 stage 的等待归因")
    else:
        print("- 当前配置已在扫描空间内最优; 建议扩大扫描维度 (路由均衡/共享专家路径)")
    if compare:
        w2 = measured_wall(compare)
        w1 = measured_wall(run)
        if w1 and w2:
            print(f"\n## 实测对照 ({compare.name})")
            print(f"- 实测墙钟: {run.name}={w1:.0f}µs vs {compare.name}={w2:.0f}µs "
                  f"→ 实测 Δ={fmt_pct(100*(w2-w1)/w1)}")
            print(f"- (模型侧同工况对比请用 --compare 的模型报告交叉验证)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--compare", type=Path, default=None)
    args = ap.parse_args()
    report(REPO / "prof_runs" / args.run if not args.run.is_absolute() else args.run,
           (REPO / "prof_runs" / args.compare if args.compare and not args.compare.is_absolute() else args.compare))
