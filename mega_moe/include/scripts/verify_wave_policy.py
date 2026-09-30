#!/usr/bin/env python3
"""Post-run verification of the p1/p2 wave-policy sweep.

For every run dir under the sweep root:
  - parse the host log line "MegaMoe wave policy: ..." (actual values)
  - recompute expected values from scripts/wave_policy.py using the ACTUAL
    aicNum printed by the platform query and the requested p1/p2 overrides
    recorded in config.json5 (cost_sweep section)
  - compare every field; also check the final "Tiling bytes=... mGroupsPerWave="
    line equals the formula value (i.e. no MGW stomp was active)

Any mismatch => exit 1 (performance conclusions must stop until resolved).

Usage: python3 verify_wave_policy.py <sweep_root> [--aic 28]
"""
import argparse
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wave_policy as wp  # noqa: E402
from config_io import load_config  # noqa: E402

WAVE_LINE = re.compile(
    r"MegaMoe wave policy: aicNum=(\d+) hiddenDim=(\d+) h=(\d+) "
    r"requested_p1=(-?\d+) requested_p2=(-?\d+) effective_p1=(\d+) effective_p2=(\d+) "
    r"gmm1TilesPerMGroup=(\d+) gmm2TilesPerMGroup=(\d+) "
    r"gmm1RequiredMGroups=(\d+) gmm2RequiredMGroups=(\d+) mGroupsPerWave=(\d+) dominant_side=(\w+)")
WAVE_LINE_LEGACY = re.compile(
    r"MegaMoe wave policy: aicNum=(\d+) hiddenDim=(\d+) h=(\d+) gmm1TilesPerMGroup=(\d+) "
    r"gmm2TilesPerMGroup=(\d+) p1=(\d+) p2=(\d+) p1Override=(-?\d+) p2Override=(-?\d+) "
    r"gmm1RequiredMGroups=(\d+) gmm2RequiredMGroups=(\d+) mGroupsPerWave=(\d+)")
FINAL_LINE = re.compile(r"Tiling bytes=\d+ workspace=\d+ maxOutput=\d+ mGroupsPerWave=(\d+)")


def tiling_determinism_check(run_dir, ref_dir, mgw, ref_mgw):
    """All tiling_rank*.bin of a run must differ from the reference run ONLY in
    the mGroupsPerWave u32 (policy effect) and the trailing profBufGm u64
    (per-run device address). Everything else identical => identical case,
    routing inputs and buffer geometry => identical expert_tokens."""
    problems = []
    mgw_offsets = set()
    for rank in range(4):
        path = Path(run_dir) / "raw" / f"tiling_rank{rank}.bin"
        ref_path = Path(ref_dir) / "raw" / f"tiling_rank{rank}.bin"
        if not path.exists() or not ref_path.exists():
            problems.append(f"tiling_rank{rank}.bin missing")
            continue
        data = path.read_bytes()
        ref = ref_path.read_bytes()
        if len(data) != len(ref) or len(data) == 0:
            problems.append(f"tiling_rank{rank}.bin size mismatch")
            continue
        # trailing 8 bytes = profBufGm (MEGAMOE_PROFILE_ABI), legitimately per-run
        body_diff = [i for i in range(len(data) - 8) if data[i] != ref[i]]
        if body_diff:
            off = body_diff[0] & ~3  # align down to the containing u32
            if any((i & ~3) != off for i in body_diff):
                problems.append(f"tiling_rank{rank}.bin differs outside a single u32 window: "
                                f"{len(body_diff)} bytes from offset {body_diff[0]}")
                continue
            mgw_offsets.add(off)
            got = struct.unpack_from("<I", data, off)[0]
            ref_got = struct.unpack_from("<I", ref, off)[0]
            if got != mgw or ref_got != ref_mgw:
                problems.append(f"tiling_rank{rank}.bin u32@{off} = {got} (ref {ref_got}), "
                                f"expected mgw {mgw} (ref {ref_mgw})")
    if len(mgw_offsets) > 1:
        problems.append(f"inconsistent mgw window offsets across ranks: {sorted(mgw_offsets)}")
    return problems


def verify_run(run_dir, expect_aic):
    log = Path(run_dir) / "run.log"
    if not log.exists():
        return None, "run.log missing"
    text = log.read_text(errors="ignore")
    m = WAVE_LINE.search(text)
    if m:
        f = m.groups()
        actual = {
            "aicNum": int(f[0]), "hiddenDim": int(f[1]), "h": int(f[2]),
            "p1Override": int(f[3]), "p2Override": int(f[4]),
            "p1": int(f[5]), "p2": int(f[6]),
            "t1": int(f[7]), "t2": int(f[8]),
            "g1": int(f[9]), "g2": int(f[10]), "mgw": int(f[11]),
        }
    else:
        m = WAVE_LINE_LEGACY.search(text)
        if not m:
            return None, "wave policy line missing"
        f = m.groups()
        actual = {
            "aicNum": int(f[0]), "hiddenDim": int(f[1]), "h": int(f[2]),
            "t1": int(f[3]), "t2": int(f[4]), "p1": int(f[5]), "p2": int(f[6]),
            "p1Override": int(f[7]), "p2Override": int(f[8]),
            "g1": int(f[9]), "g2": int(f[10]), "mgw": int(f[11]),
        }
    finals = FINAL_LINE.findall(text)
    if not finals:
        return actual, "final Tiling line missing"

    cfg = load_config(str(Path(run_dir) / "config.json5"))
    case = cfg["case"]
    sweep = cfg.get("cost_sweep", {})
    p1_req = int(sweep.get("p1Override", 0))
    p2_req = int(sweep.get("p2Override", 0))
    bs = case["tokens"]
    hidden_dim = 2 * case["intermediate"]
    h = case["hidden"]

    problems = []
    if expect_aic is not None and actual["aicNum"] != expect_aic:
        problems.append(f"aicNum {actual['aicNum']} != expected {expect_aic}")
    if actual["hiddenDim"] != hidden_dim or actual["h"] != h:
        problems.append(f"shape mismatch vs config: hiddenDim/h = {actual['hiddenDim']}/{actual['h']}")
    if actual["p1Override"] != p1_req or actual["p2Override"] != p2_req:
        problems.append(f"requested overrides {actual['p1Override']}/{actual['p2Override']} "
                        f"!= config {p1_req}/{p2_req}")

    exp = wp.expected_policy(actual["aicNum"], hidden_dim, h, bs, p1_req, p2_req)
    for key, label in [("p1", "p1"), ("p2", "p2"), ("t1", "t1"), ("t2", "t2"),
                       ("g1", "g1"), ("g2", "g2"), ("mgw", "mGroupsPerWave")]:
        if actual[key] != exp[key]:
            problems.append(f"{label}: actual {actual[key]} != expected {exp[key]}")
    final_mgw = int(finals[-1])
    if final_mgw != exp["mgw"]:
        problems.append(f"final Tiling mGroupsPerWave {final_mgw} != formula {exp['mgw']}")

    return {**actual, "expected": exp, "final_mgw": final_mgw}, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_root")
    ap.add_argument("--aic", type=int, default=28, help="expected actual AIC count (None=skip check)")
    args = ap.parse_args()

    root = Path(args.sweep_root)
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "run_one.sh").exists())
    if not run_dirs:
        print(f"no prepared runs under {root}")
        return 1

    print(f"| run | p1(req) | p2(req) | p1 | p2 | G1 | G2 | mgw | dominant | final | tiling | check |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|")
    n_bad = 0
    ref_dir = root / "default"
    for rd in run_dirs:
        actual, problems = verify_run(rd, args.aic)
        name = rd.name
        if actual is None:
            print(f"| {name} | - | - | - | - | - | - | - | - | - | - | **NO LOG: {problems}** |")
            n_bad += 1
            continue
        exp = actual["expected"]
        dom = "GMM1" if exp["g1"] > exp["g2"] else ("GMM2" if exp["g2"] > exp["g1"] else "crossover")
        # input-determinism: tiling must differ from the default run only in mgw (+profBufGm)
        if (rd / "run_one.sh").exists():
            ref_actual, _ = verify_run(ref_dir, args.aic) if rd != ref_dir else (actual, [])
            ref_mgw = ref_actual["mgw"] if ref_actual else None
            til_problems = tiling_determinism_check(rd, ref_dir, actual["mgw"], ref_mgw)
            problems = problems + [p for p in til_problems if p not in problems]
        til_status = "OK" if not til_problems else "**" + "; ".join(til_problems) + "**"
        status = "OK" if not problems else "**" + "; ".join(problems) + "**"
        n_bad += 0 if not problems else 1
        print(f"| {name} | {actual['p1Override']} | {actual['p2Override']} | {actual['p1']} | {actual['p2']} "
              f"| {actual['g1']} | {actual['g2']} | {actual['mgw']} | {dom} | {actual['final_mgw']} "
              f"| {til_status} | {status} |")

    print()
    print(f"result: {'PASS' if n_bad == 0 else 'FAIL'} ({n_bad} bad / {len(run_dirs)} runs)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
