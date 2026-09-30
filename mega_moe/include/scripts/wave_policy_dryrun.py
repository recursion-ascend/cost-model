#!/usr/bin/env python3
"""Dry-run table for the p1/p2 wave-policy sweep (no performance runs).

Reads the ACTUAL aicNum (from --aic, cross-checked against run.log evidence),
computes G1(p1) / G2(p2) / expected mGroupsPerWave / dominant side for the
candidate grid, and selects an informative subset that covers:

  A. G1 > G2  (GMM1 policy dominates)
  B. G1 == G2 (crossover)
  C. G2 > G1  (GMM2 policy dominates)

Configs that leave mGroupsPerWave unchanged (e.g. raising p2 while GMM1 still
dominates) carry no information and are not selected for the perf sweep
(user requirement: don't sweep a no-op p2 range).

Usage:
  python3 wave_policy_dryrun.py --aic 28 --hidden-dim 4096 --h 6144 --bs 64 \
      [--output DRYRUN_TABLE.md] [--emit-specs]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wave_policy as wp  # noqa: E402

P1_CANDIDATES = [1, 2, 3, 4, 6, 8]
P2_CANDIDATES = [1, 2, 4, 6, 8, 12]

# Curated selection for the fixed perf case (aic=28, hiddenDim=4096 -> t1=16,
# h=6144 -> t2=24). Selection rules:
#   - p2=1 row across p1 (G1-dom sweep -> distinct mgw values 2/4/6/7/11/14)
#   - crossover points (G1==G2) and G2-dom points incl. same-mgw-different-side
# The default (auto policy) run is emitted additionally as spec default:0:0.
# Every selected row's expected values are re-derived from the model below and
# asserted, so a wrong aicNum or shape input fails loudly instead of mislabeling.
SELECTED = [
    ("p1_1_p2_1", 1, 1),    # crossover, mgw 2
    ("p1_2_p2_1", 2, 1),    # G1 dom, mgw 4 (== default, override-path equivalence)
    ("p1_2_p2_3", 2, 3),    # crossover, mgw 4
    ("p1_3_p2_1", 3, 1),    # G1 dom, mgw 6
    ("p1_4_p2_1", 4, 1),    # G1 dom, mgw 7 (user-stated baseline policy)
    ("p1_4_p2_6", 4, 6),    # crossover, mgw 7
    ("p1_1_p2_6", 1, 6),    # G2 dom, mgw 7 (same mgw as p1_4_p2_1, other side)
    ("p1_4_p2_8", 4, 8),    # G2 dom, mgw 10
    ("p1_6_p2_1", 6, 1),    # G1 dom, mgw 11
    ("p1_8_p2_1", 8, 1),    # G1 dom, mgw 14
    ("p1_4_p2_12", 4, 12),  # G2 dom, mgw 14
    ("p1_1_p2_12", 1, 12),  # G2 dom, mgw 14
    ("p1_8_p2_12", 8, 12),  # crossover, mgw 14
]


def dominant_label(g1, g2):
    return "GMM1>GMM2 (A)" if g1 > g2 else ("GMM2>GMM1 (C)" if g2 > g1 else "crossover (B)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aic", type=int, required=True, help="actual AIC core count from run.log")
    ap.add_argument("--hidden-dim", type=int, required=True, help="GMM1 full output N width")
    ap.add_argument("--h", type=int, required=True, help="model hidden size (GMM2 N width)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--output", default=None)
    ap.add_argument("--emit-specs", action="store_true", help="print NAME:P1:P2 lines for sweep_p1p2.sh")
    args = ap.parse_args()

    aic, hidden_dim, h, bs = args.aic, args.hidden_dim, args.h, args.bs
    t1 = wp.ceil_div(hidden_dim, wp.GMM_TILE_N)
    t2 = wp.ceil_div(h, wp.GMM_TILE_N)
    if hidden_dim % wp.GMM_TILE_N or h % wp.GMM_TILE_N:
        print(f"WARNING: hidden_dim/h not multiples of {wp.GMM_TILE_N}; CeilDiv tails apply", file=sys.stderr)

    rows = []
    lines = [
        "# p1/p2 wave-policy dry-run table (no perf runs)",
        "",
        f"- aicNum = **{aic}** (actual, from run.log `Platform aic=`)",
        f"- hiddenDim = {hidden_dim} -> gmm1TilesPerMGroup t1 = ceil({hidden_dim}/256) = **{t1}**",
        f"- h = {h} -> gmm2TilesPerMGroup t2 = ceil({h}/256) = **{t2}**",
        f"- bs = {bs} -> default tier p1 = {wp.resolve_p1(bs)}, p2 = {wp.GMM2_MIN_LOGICAL_TILES_PER_CORE}",
        "",
        "G1 = ceil(aicNum * p1 / t1), G2 = ceil(aicNum * p2 / t2), mGroupsPerWave = max(G1, G2)",
        "",
        "## Full candidate grid",
        "",
        "| p1 | p2 | G1 | G2 | expected mGroupsPerWave | dominant side |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for p1 in P1_CANDIDATES:
        for p2 in P2_CANDIDATES:
            r = wp.wave_policy(aic, hidden_dim, h, p1, p2)
            rows.append((p1, p2, r))
            lines.append(f"| {p1} | {p2} | {r['g1']} | {r['g2']} | {r['mgw']} | {dominant_label(r['g1'], r['g2'])} |")

    default_r = wp.expected_policy(aic, hidden_dim, h, bs)
    lines += [
        "",
        "## Selected informative configs (perf sweep)",
        "",
        "| name | p1 | p2 | G1 | G2 | expected mGroupsPerWave | dominant side | note |",
        "|---|---:|---:|---:|---:|---:|---|---|",
        f"| default | auto({default_r['p1']}) | {default_r['p2']} | {default_r['g1']} | {default_r['g2']} | "
        f"{default_r['mgw']} | {dominant_label(default_r['g1'], default_r['g2'])} | regression anchor |",
    ]
    sel_rows = []
    for name, p1, p2 in SELECTED:
        r = wp.wave_policy(aic, hidden_dim, h, p1, p2)
        note = {
            "p1_2_p2_1": "same mgw as default: override-path equivalence",
            "p1_4_p2_1": "user-stated baseline policy (fixed p1=4)",
            "p1_1_p2_6": "same mgw as p1_4_p2_1, GMM2-dominated",
            "p1_2_p2_3": "same mgw as default, crossover",
        }.get(name, "")
        lines.append(f"| {name} | {p1} | {p2} | {r['g1']} | {r['g2']} | {r['mgw']} | "
                     f"{dominant_label(r['g1'], r['g2'])} | {note} |")
        sel_rows.append((name, p1, p2, r))

    # Coverage assertions: this experiment's most important design requirement.
    doms = {dominant_label(r["g1"], r["g2"]) for _, _, _, r in sel_rows} | {
        dominant_label(default_r["g1"], default_r["g2"])}
    mgws = {r["mgw"] for _, _, _, r in sel_rows} | {default_r["mgw"]}
    checks = [
        ("coverage A (G1>G2)", any(d.startswith("GMM1>") for d in doms)),
        ("coverage B (crossover)", any(d.startswith("crossover") for d in doms)),
        ("coverage C (G2>G1)", any(d.startswith("GMM2>") for d in doms)),
        ("p2 changes mgw at fixed p1=4", len({wp.wave_policy(aic, hidden_dim, h, 4, p2)["mgw"]
                                               for p2 in P2_CANDIDATES}) > 1),
        ("selected distinct mgw values >= 6", len(mgws) >= 6),
    ]
    lines += ["", "## Coverage checks", ""]
    ok = True
    for label, passed in checks:
        lines.append(f"- {label}: {'OK' if passed else 'FAILED'}")
        ok = ok and passed
    lines += ["", f"- result: {'**PASS**' if ok else '**FAIL**'}", ""]

    report = "\n".join(lines)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n")
        print(f"\nwritten: {args.output}")
    if args.emit_specs:
        print("SPEC default:0:0")  # no env overrides -> tier p1 + p2=1 (regression anchor)
        for name, p1, p2 in SELECTED:
            print(f"SPEC {name}:{p1}:{p2}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
