#!/usr/bin/env python3
"""去硬编码版全面对比: 参数全部从 run 工件自动推导.

真值来源:
  - tiling_rank0.bin  → bs, h, hiddenDim, topk, aic, shared, mGroupsPerWave (kernel 实际)
  - config.json5      → seed, routing mode
  - routing CSV       → 仅当与 tiling bs 一致时使用; 否则用 make_routing(bs) 重建
  - P1/P2 policy      → 从 tiling mGroupsPerWave 反解 (不再用默认值)
"""
import sys, struct, statistics, csv, re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
REPO = PROJ.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(Path(__file__).parent))
from moe_cost_model import (
    PrimitiveCosts, simulate_routing_counts, calc_m_groups_per_wave,
    AnalyticalGmmCosts, AnalyticalActCosts, AnalyticalCombineCosts,
    DispatchMechanisticLatency, BW_L1_GM, T_COUNT_GATE, parse_tiling, KernelConfig,
)
from routing import make_routing

ROOT = REPO / "prof_runs"
N_CORES, RING_ALIGN, REC = 84, 64, 16
STAGES = {0x6011: "DISPATCH", 0x6021: "GMM1", 0x6031: "ACT", 0x6041: "GMM2", 0x6051: "COMBINE"}


def read_bin(path):
    raw = Path(path).read_bytes()
    ring = N_CORES * RING_ALIGN
    slots = (len(raw) - ring) // (N_CORES * REC)
    per = {}
    for cid in range(N_CORES):
        cnt = struct.unpack_from("<I", raw, cid * RING_ALIGN)[0]
        base = ring + cid * slots * REC
        ev = []
        for i in range(cnt):
            o = base + i * REC
            eid, pay = struct.unpack_from("<II", raw, o)
            cyc, = struct.unpack_from("<Q", raw, o + 8)
            ev.append((cyc, eid & 0xFFFF, pay))
        per[cid] = sorted(ev)
    return per


def meas_rank(run, rank, reps):
    busy = {}
    walls = []
    n = 0
    for rep in range(reps):
        for pat in (f"raw/prof_rank{rank}_rep{rep}.bin", f"raw/prof_rank{rank}.bin"):
            p = run / pat
            if p.exists():
                break
        else:
            continue
        n += 1
        b = {}
        kern = []
        for cid, ev in read_bin(p).items():
            q = {}
            for cyc, eid, pay in ev:
                if eid == 0x0001 or eid in STAGES:
                    q.setdefault((eid, (pay >> 31) if eid in STAGES else 0), []).append(cyc)
                elif eid == 0x00FF:
                    if q.get((0x0001, 0)):
                        kern.append((q[(0x0001, 0)].pop(0), cyc))
                elif (eid - 1) in STAGES:
                    for fl in (0, 1):
                        k = (eid - 1, fl)
                        if q.get(k):
                            bb = q[k].pop(0)
                            if fl == 0:
                                b[STAGES[eid - 1]] = b.get(STAGES[eid - 1], 0) + (cyc - bb) / 1000.0
                            break
        for k, v in b.items():
            busy[k] = busy.get(k, 0) + v
        if kern:
            walls.append((max(e for _, e in kern) - min(b2 for b2, _ in kern)) / 1000.0)
    if n > 0:
        for k in busy:
            busy[k] /= n
    return busy, statistics.mean(walls) if walls else 0


def derive_p1_p2(mgw, hidden, h, aic):
    """从 tiling 的 mGroupsPerWave 反解 (p1, p2), 与 calc_m_groups_per_wave 对齐."""
    for p2 in (1, 2, 3, 8):
        for p1 in (2, 3, 4, 6, 8, 12, 16):
            if calc_m_groups_per_wave(hidden_dim=hidden, h=h, aic_num=aic, p1=p1, p2=p2) == mgw:
                return p1, p2
    return 0, 0  # 0 = 无 override (用默认 tier)


def begin_offset():
    SW = ROOT / "asym_bw_20260921"
    acc = {}
    for rep in range(6):
        per = read_bin(SW / "asym_single" / f"raw/prof_rank0_rep{rep}.bin")
        t0 = min(c for ev in per.values() for c, _, _ in ev)
        b = {}
        for cid, ev in per.items():
            for cyc, eid, pay in ev:
                if eid == 0x6011 and ((pay >> 16) & 0xFF) == 0:
                    k = (cid - 1) // 2
                    if 0 <= k < 28:
                        b[k] = (cyc - t0) / 1000.0
        med = statistics.median(b.values())
        for k, v in b.items():
            acc.setdefault(k, []).append(v - med)
    return tuple(round(statistics.mean(v), 3) for k, v in sorted(acc.items()))


BO = begin_offset()
agc = AnalyticalGmmCosts()
act = AnalyticalActCosts()


def mk_costs(h):
    comb = AnalyticalCombineCosts()
    return PrimitiveCosts(
        dispatch_mechanistic=DispatchMechanisticLatency(begin_offset_us=BO),
        gmm1_tile=agc.gmm1_tile,
        gmm2_tile=agc.gmm2_tile,
        activation_tile=act.tile,
        combine_tile=comb.tile,
        count_table_prepare_us=T_COUNT_GATE,
    )


def auto_config(run):
    """从工件自动推导全部模型输入, 返回 None 表示工件不全."""
    tiling_p = run / "raw/tiling_rank0.bin"
    cfg_p = run / "config.json5"
    if not tiling_p.exists():
        return None
    t = parse_tiling(tiling_p)
    seed, routing_mode = 0, "random"
    if cfg_p.exists():
        cfg = cfg_p.read_text()
        m = re.search(r'"seed":\s*(\d+)', cfg)
        if m:
            seed = int(m.group(1))
        m = re.search(r'"routing":\s*"(\w+)"', cfg)
        if m:
            routing_mode = m.group(1)
    world = t["ep"]
    local = t["moeEpr"]

    # C 矩阵: CSV 优先, 但必须与 tiling bs 一致 
    csv_total_ok = all((run / f"raw/routing_rank{s}.csv").exists() for s in range(world))
    C = None
    if csv_total_ok:
        C = [[[0] * local for _ in range(world)] for _ in range(world)]
        for s in range(world):
            for row in csv.DictReader(open(run / f"raw/routing_rank{s}.csv")):
                C[int(row["dst_rank"])][s][int(row["local_expert_id"])] += int(row["token_count"])
        # 校验: 每 src 总行数应 = bs × topk
        expect = t["bs"] * t["topk"]
        got = sum(sum(x) for x in C[0])  # dst0 收到的来自 src0 的行数 = bs×topk/world×... 不对
        # 每 src 发出的总 slot 数 = sum over dst,e of C[dst][s][e] = bs × topk
        got_src0 = sum(C[d][0][e] for d in range(world) for e in range(local))
        if got_src0 != expect:
            C = None  # CSV 与 kernel bs 不一致 → 重建
    if C is None:
        case = dict(tokens=t["bs"], experts=local * world, topk=t["topk"],
                    ep=world, seed=seed, routing=routing_mode)
        C = [[[0] * local for _ in range(world)] for _ in range(world)]
        for s in range(world):
            for gid in make_routing(case, s).reshape(-1).tolist():
                C[gid // local][s][gid % local] += 1

    reps = len(list(run.glob("raw/prof_rank0_rep*.bin"))) or 1
    p1, p2 = derive_p1_p2(t["mGroupsPerWave"], t["hidden"], t["h"], t["aic"])
    return t, C, reps, p1, p2, routing_mode


CURATED = [
    ROOT / "wave_policy_reps_20260920/default",
    ROOT / "wave_policy_reps_20260920/mgw7_p1_4_p2_1",
    ROOT / "wave_policy_reps_20260920/mgw14_p1_4_p2_12",
    ROOT / "asym_bw_20260921/random7777",
    ROOT / "asym_bw_20260921/cyclic",
    ROOT / "asym_bw_20260921/asym_single",
    ROOT / "asym_bw_20260921/asym_puller",
    ROOT / "asym_bw_20260921/asym_server",
    ROOT / "hot_b128/hot_m4",
    ROOT / "hverify_h6144/h6144_m7",
    ROOT / "hverify_h8192/h8192_m7",
    ROOT / "bs_sweep_v2/bs128",
]
RUN_DIRS = [p for p in CURATED if (p / "raw/tiling_rank0.bin").exists()]

print(f"{'数据集':<34} {'R':>2} {'DISP':>7} {'GMM1':>7} {'ACT':>7} {'GMM2':>7} {'COMB':>7} {'墙钟':>7}")
print("=" * 84)

for run in RUN_DIRS:
    ac = auto_config(run)
    if ac is None:
        continue
    t, C, reps, p1, p2, routing_mode = ac
    world = t["ep"]
    rc = tuple(tuple(tuple(C[d][s][e] for s in range(world)) for e in range(len(C[0][0]))) for d in range(world))
    costs = mk_costs(t["h"])
    res = simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=t["bs"], h=t["h"], hidden_dim=t["hidden"],
        aic_num=t["aic"], costs=costs, p1_override=p1, p2_override=p2,
        topk=t["topk"], shared_expert_num=t["shared"],
        kernel=KernelConfig(weight_nz=(t.get("groupedMatmulMode") == 2)),
        # 注: NZ run 需先标定 PrimitiveCosts.gmm1_bw_b_nz_bytes_per_us
    )
    label = f"{run.parent.name}/{run.name}"[:32]
    extra = f"[p1={p1},p2={p2}]" if p1 or p2 else ""
    for rank in range(world):
        meas_busy, meas_wall = meas_rank(run, rank, reps)
        if meas_wall == 0:
            continue
        sp = {}
        for ev in res["rank_results"][rank]["events"]:
            st = str(ev.meta.get("stage", "other"))
            sp[st] = sp.get(st, 0) + (ev.end_us - ev.start_us)
        model_wall = res["rank_results"][rank]["total_us"]
        row = []
        for stage, keys in [("DISPATCH", ("dispatch", "dispatch_call")), ("GMM1", ("gmm1",)),
                            ("ACT", ("activation",)), ("GMM2", ("gmm2",)), ("COMBINE", ("combine",))]:
            mm = meas_busy.get(stage, 0)
            pm = sum(sp.get(k, 0) for k in keys)
            row.append(f"{100*(pm/mm-1):+6.1f}%" if mm > 0 else "     -")
        we = 100 * (model_wall / meas_wall - 1)
        hdr = f"{label} {extra}" if rank == 0 else ""
        print(f"{hdr:<34} r{rank:>2} {''.join(row)}{we:+6.1f}%")
    print(f"  {run.name}: bs={t['bs']} h={t['h']} topk={t['topk']} mgw={t['mGroupsPerWave']} "
          f"routing={routing_mode} reps={reps} {'CSV不一致→重建C' if False else ''}")
