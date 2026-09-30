#!/usr/bin/env python3
"""Correctness regression for the p1/p2 wave-policy parameterization.

Checks, over a grid of (bs, h, hiddenDim, aicNum):
  1. NEW default (tiered p1, p2=1) == HEAD@ebf6f3a formula  -> behavior preserved
  2. NEW default == imported-baseline b258e58 (fixed p1=4) for the middle tier
     (2048 <= bs < 16384)                                   -> user-stated default p1=4
  3. NEW with explicit p1=4,p2=1 == baseline for ALL bs      -> override path sanity
  4. Historical run.log anchors reproduce

Usage: python3 wave_policy_regression.py [--output REGRESSION.md]
Exit code 0 iff all checks pass.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wave_policy as wp  # noqa: E402

# Grid: multiple H / hiddenDim / aicNum / bs, including tier boundaries and
# non-multiple-of-256 tails to stress CeilDiv.
BS_VALUES = [1, 64, 1024, 2047, 2048, 2049, 5000, 8192, 16383, 16384, 16385, 65536, 0]
H_VALUES = [1024, 2048, 4096, 5120, 6144, 7168, 8192, 1056]        # h (GMM2 N width)
HIDDEN_DIM_VALUES = [512, 1024, 2048, 2560, 4096, 5120, 6144, 8192, 576]  # hiddenDim (GMM1 N width)
AIC_VALUES = [1, 2, 4, 8, 16, 20, 24, 28, 32, 48, 56, 64]

# Anchors from real run.log files (pre-patch host behavior).
ANCHORS = [
    # (bs, h, hiddenDim, aic, expected_mgw, source)
    (8192, 4096, 4096, 28, 7, "prof_runs/20260914_225500_global_wave/run.log (bs=8192 H=4096)"),
    (8192, 4096, 4096, 28, 7, "prof_runs/20260914_213500_v4_noshared_wave/run.log"),
    (64, 6144, 4096, 28, 4, "fixed perf case, default tier (bs=64 -> p1=2)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None, help="write markdown report to this path")
    args = ap.parse_args()

    lines = []
    total = 0
    failures = []

    # Check 1: default (tiered p1, p2=1) == HEAD reference, full grid.
    bad1 = 0
    for bs in BS_VALUES:
        for h in H_VALUES:
            for hidden_dim in HIDDEN_DIM_VALUES:
                for aic in AIC_VALUES:
                    old = wp.old_wave_policy(aic, hidden_dim, h, bs)["mgw"]
                    new = wp.expected_policy(aic, hidden_dim, h, bs)["mgw"]
                    total += 1
                    if old != new:
                        bad1 += 1
                        failures.append(f"check1: bs={bs} h={h} hiddenDim={hidden_dim} aic={aic}: "
                                        f"old={old} new={new}")
    lines.append(f"## Check 1: default (tiered p1, p2=1) == HEAD@ebf6f3a formula\n"
                 f"- grid points: {total}, mismatches: **{bad1}**\n")

    # Check 2: middle tier == imported baseline (fixed p1=4).
    bad2 = 0
    n2 = 0
    for bs in [2048, 2049, 5000, 8192, 16383]:
        for h in H_VALUES:
            for hidden_dim in HIDDEN_DIM_VALUES:
                for aic in AIC_VALUES:
                    base = wp.baseline_wave_policy(aic, hidden_dim, h)["mgw"]
                    new = wp.expected_policy(aic, hidden_dim, h, bs)["mgw"]
                    n2 += 1
                    if base != new:
                        bad2 += 1
                        failures.append(f"check2: bs={bs} h={h} hiddenDim={hidden_dim} aic={aic}: "
                                        f"baseline={base} new={new}")
    lines.append(f"## Check 2: middle tier default == imported baseline (fixed p1=4, p2=1)\n"
                 f"- grid points: {n2}, mismatches: **{bad2}**\n")

    # Check 3: explicit p1=4, p2=1 == baseline for ALL bs (override bypasses tiers).
    bad3 = 0
    n3 = 0
    for bs in BS_VALUES:
        for h in H_VALUES:
            for hidden_dim in HIDDEN_DIM_VALUES:
                for aic in AIC_VALUES:
                    base = wp.baseline_wave_policy(aic, hidden_dim, h)["mgw"]
                    ov = wp.expected_policy(aic, hidden_dim, h, bs, p1_override=4, p2_override=1)["mgw"]
                    n3 += 1
                    if base != ov:
                        bad3 += 1
                        failures.append(f"check3: bs={bs} h={h} hiddenDim={hidden_dim} aic={aic}: "
                                        f"baseline={base} p1=4 override={ov}")
    lines.append(f"## Check 3: explicit p1=4,p2=1 == baseline for all bs (override path)\n"
                 f"- grid points: {n3}, mismatches: **{bad3}**\n")

    # Check 4: historical anchors.
    lines.append("## Check 4: historical log anchors\n")
    bad4 = 0
    for bs, h, hidden_dim, aic, expect, src in ANCHORS:
        got = wp.expected_policy(aic, hidden_dim, h, bs)["mgw"]
        got_old = wp.old_wave_policy(aic, hidden_dim, h, bs)["mgw"]
        ok = got == expect and got_old == expect
        bad4 += 0 if ok else 1
        lines.append(f"- bs={bs} h={h} hiddenDim={hidden_dim} aic={aic}: expected={expect} "
                     f"new={got} old={got_old} -> {'OK' if ok else 'MISMATCH'} ({src})")
    lines.append("")

    n_fail = bad1 + bad2 + bad3 + bad4
    header = [f"# Wave policy p1/p2 parameterization regression",
              "",
              f"- date: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
              f"- result: {'**PASS**' if n_fail == 0 else '**FAIL**'} ({n_fail} failures)",
              ""]
    report = "\n".join(header + lines + (["## Failures", "```"] + failures[:50] + ["```"] if failures else []))

    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n")
        print(f"\nwritten: {args.output}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
