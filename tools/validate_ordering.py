#!/usr/bin/env python3
"""编译器参数组合序关系一致性验证.

相同模型参数 (唯一变量 = p1/p2 编译器超参), 模型排序 vs 实测排序:
Kendall tau + 不一致对分析 (区分 '模型错' 与 '实测噪声内').

数据: wave_policy_reps_20260920 (B=64, 37 组合, 10 reps, 4 ranks).
"""
import re
import statistics
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
REPO = PROJ.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(Path(__file__).parent))

from artifacts import read_prof_bin
from moe_cost_model import (
    AnalyticalActCosts, AnalyticalCombineCosts, AnalyticalGmmCosts,
    DispatchMechanisticLatency, KernelConfig, PrimitiveCosts, T_COUNT_GATE,
    simulate_routing_counts, parse_tiling,
)
from routing import make_routing

SWEEP = REPO / "prof_runs/wave_policy_reps_20260920"


def parse_p1p2(name, tiling):
    m = re.match(r"p1_(\d+)_p2_(\d+)$", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"mgw\d+_p1_(\d+)_p2_(\d+)$", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 2, 1  # default: 自动档 (tokens<2048 → p1=2, p2=1)


def measured_kernel_wall(run):
    """每 rep 的 kernel 墙钟 = max over ranks; 返回 (mean, stdev, n)."""
    reps = sorted(run.glob("raw/prof_rank0_rep*.bin"))
    if not reps:
        return 0.0, 0.0, 0
    n_rep = len(reps)
    walls = []
    for rep in range(n_rep):
        rank_walls = []
        for rank in range(4):
            p = run / f"raw/prof_rank{rank}_rep{rep}.bin"
            if not p.exists():
                continue
            kern = []
            for cid, ev in read_prof_bin(p).items():
                for cyc, eid, pay in ev:
                    if eid in (0x0001, 0x00FF):
                        kern.append(cyc)
            if len(kern) >= 2:
                rank_walls.append((max(kern) - min(kern)) / 1000.0)
        if rank_walls:
            walls.append(max(rank_walls))
    if not walls:
        return 0.0, 0.0, 0
    return statistics.mean(walls), statistics.stdev(walls) if len(walls) > 1 else 0.0, len(walls)


def build_C(t, seed, routing):
    world, local = t["ep"], t["moeEpr"]
    case = dict(tokens=t["bs"], experts=local * world, topk=t["topk"],
                ep=world, seed=seed, routing=routing)
    C = [[[0] * world for _ in range(local)] for _ in range(world)]
    for s in range(world):
        for gid in make_routing(case, s).reshape(-1).tolist():
            C[gid // local][gid % local][s] += 1
    return C


def main():
    runs = sorted(d for d in SWEEP.iterdir()
                  if (d / "raw/tiling_rank0.bin").exists() and d.is_dir()
                  and not d.name.endswith((".log",))
                  and d.name not in ("build_shared",))
    # 路由固定 (同 seed), C 只算一次
    t0 = parse_tiling(runs[0] / "raw/tiling_rank0.bin")
    cfg = runs[0] / "config.json5"
    seed = int(re.search(r'"seed":\s*(\d+)', cfg.read_text()).group(1))
    routing = re.search(r'"routing":\s*"(\w+)"', cfg.read_text()).group(1)
    C = build_C(t0, seed, routing)
    rc = tuple(tuple(tuple(r) for r in cd) for cd in C)

    rows = []
    for run in runs:
        t = parse_tiling(run / "raw/tiling_rank0.bin")
        p1, p2 = parse_p1p2(run.name, t)
        mw, msd, n = measured_kernel_wall(run)
        if n == 0:
            continue
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
            p1_override=p1, p2_override=p2, topk=t["topk"],
            shared_expert_num=t["shared"])
        model_wall = max(res["rank_results"][r]["total_us"] for r in range(t["ep"]))
        rows.append({"name": run.name, "p1": p1, "p2": p2, "mgw": t["mGroupsPerWave"],
                     "meas": mw, "meas_sd": msd, "model": model_wall, "n": n})

    # ---- 序关系分析 ----
    rows.sort(key=lambda r: r["meas"])
    n = len(rows)
    concord = discord = 0
    discordant = []
    for i in range(n):
        for j in range(i + 1, n):
            dm = rows[j]["meas"] - rows[i]["meas"]       # 实测: j 更慢
            dmod = rows[j]["model"] - rows[i]["model"]   # 模型差
            noise = 2 * (rows[i]["meas_sd"] + rows[j]["meas_sd"])
            if dm <= noise:            # 实测差在噪声内: 不计入判分
                continue
            if (dm > 0) == (dmod > 0) or abs(dmod) < 1e-9 and abs(dm) < 1e-9:
                concord += 1
            else:
                discord += 1
                discordant.append((rows[i]["name"], rows[j]["name"], dm, dmod, noise))

    tau = (concord - discord) / (concord + discord) if (concord + discord) else 1.0
    print(f"组合数: {n} (有效比较对: {concord + discord})")
    print(f"一致对: {concord}  不一致对: {discord}  Kendall tau = {tau:+.3f}")
    print()
    print(f"{'组合':<20} {'mgw':>4} {'实测墙钟':>9} {'±sd':>6} {'模型墙钟':>9} {'误差':>7}")
    for r in rows:
        err = 100 * (r["model"] / r["meas"] - 1)
        print(f"  {r['name']:<20} {r['mgw']:>3} {r['meas']:9.1f} {r['meas_sd']:6.1f} "
              f"{r['model']:9.1f} {err:+6.1f}%")
    if discordant:
        print(f"\n不一致对 (实测差 > 噪声 2σ):")
        for a, b, dm, dmod, noise in discordant:
            print(f"  {a} vs {b}: 实测Δ={dm:+.1f}µs 模型Δ={dmod:+.1f}µs (噪声阈 {noise:.1f}µs)")


if __name__ == "__main__":
    main()
