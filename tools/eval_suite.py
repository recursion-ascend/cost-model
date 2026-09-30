"""跨 run 精度汇总: 一张表看全部实测 run 的逐 stage 误差.

compare_measured.py 给单个 run 的细节; 这个给横向对比 —— 误差随 batch / 共享专家
怎么变, 才能看出一个偏差是"形态错"还是"常数错":
  形态错 -> 误差随形状系统性漂移 (如 GMM1 的 A 流相加, m=72 时 +4.7%, m=256 时 +40.8%)
  常数错 -> 误差在各形状上大小一致 (如 GMM1 改 max 之后各 run 都 -6~-8%)

用法: python tools/eval_suite.py [--rank 0]
按 examples/*.toml 里带 [tiling] 且指向 data/ 的场景自动配对。
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import moe_cost_model as m  # noqa: E402
from compare_measured import (  # noqa: E402
    STAGE_MAP, load_trace, measured_stage, merge_gmm2_parts, model_stage)

STAGES = [s for s, _ in STAGE_MAP]


def pct(model: float, meas: float) -> float:
    return 100.0 * (model - meas) / meas if meas else float("nan")


def pairs():
    """(标签, 场景文件, run 目录) —— 从场景的 [tiling] path 反推 run 目录."""
    out = []
    for toml in sorted((ROOT / "examples").glob("*.toml")):
        text = toml.read_text(encoding="utf-8")
        if "[tiling]" not in text or "/data/" not in text.replace("\\", "/"):
            continue
        for line in text.splitlines():
            if line.strip().startswith("path"):
                p = (toml.parent / line.split("=", 1)[1].strip().strip('"')).resolve()
                out.append((toml.stem, toml, p.parent.parent))
                break
    return out


def evaluate(toml: Path, run: Path, rank: int):
    sc = m.load_scenario(str(toml))
    res = m.simulate(sc)
    rr = res["rank_results"][rank]
    ev = load_trace(run, rank)
    origin = min(e["ts"] for e in ev if e["name"].startswith("DISPATCH"))
    end = max(e["ts"] + e["dur"] for e in ev if e["name"].startswith("COMBINE"))
    row = {"total": pct(res["kernel_total_us"], end - origin),
           "total_model": res["kernel_total_us"], "total_meas": end - origin,
           "warnings": res.get("warnings", ())}
    for stage, prefixes in STAGE_MAP:
        ms, xs = model_stage(rr, stage), measured_stage(ev, prefixes)
        if not ms or not xs:
            row[stage] = None
            continue
        md = merge_gmm2_parts(rr) if stage == "gmm2" else ms["dur"]
        row[stage] = pct(statistics.median(md), statistics.median(xs["dur"]))
        row[stage + "_n"] = (ms["n"], xs["n"])
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    ps = pairs()
    if not ps:
        print("没找到带 [tiling] 且指向 data/ 的场景", file=sys.stderr)
        return 1

    rows = []
    for tag, toml, run in ps:
        try:
            rows.append((tag, evaluate(toml, run, args.rank)))
        except Exception as exc:                       # noqa: BLE001
            print(f"!! {tag}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not rows:
        return 1

    hdr = f"{'run':<22} {'总时长':>18} " + " ".join(f"{s:>12}" for s in STAGES)
    print(f"\n== 逐 stage 单事件中位误差 (模型 vs 实测, rank {args.rank}) ==")
    print(hdr)
    print("-" * len(hdr))
    for tag, r in rows:
        cells = []
        for s in STAGES:
            cells.append("        -   " if r.get(s) is None else f"{r[s]:+11.1f}%")
        print(f"{tag:<22} {r['total_model']:8.0f}/{r['total_meas']:<8.0f} " + " ".join(cells))
    print("-" * len(hdr))
    print(f"{'总时长误差':<22} " + " " * 18 + " ".join(f"{'':>12}" for _ in STAGES))
    for tag, r in rows:
        print(f"  {tag:<28} {r['total']:+6.1f}%")

    print("\n== 事件数核对 (模型/实测; 不一致说明 DAG 结构错) ==")
    print(f"{'run':<22} " + " ".join(f"{s:>14}" for s in STAGES))
    for tag, r in rows:
        cells = []
        for s in STAGES:
            n = r.get(s + "_n")
            if n is None:
                cells.append(f"{'-':>14}")
            else:
                mark = "" if n[0] == n[1] else " !"
                cells.append(f"{n[0]}/{n[1]}{mark}".rjust(14))
        print(f"{tag:<22} " + " ".join(cells))
    print("\n  注: GMM2 模型拆 head/tail 两事件, 数量应是实测的 2 倍 (非错)。")

    warn = [(t, r["warnings"]) for t, r in rows if r["warnings"]]
    if warn:
        print("\n== 护栏告警 ==")
        for t, ws in warn:
            for w in ws:
                print(f"  {t}: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
