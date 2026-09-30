"""把 bench_mem.csv 变成 moe_cost_model 能直接用的常数.

做两件分离, 不做曲线拟合之外的任何加工:
  1. 尺寸轴 -> 最小二乘取 (截距, 斜率): 截距 = 每次请求的固定开销, 1/斜率 = 带宽
  2. 并发轴 -> 单核独占速率 vs 整卡聚合: 并发 1 的速率是"无争用单核",
     并发 N 的速率 x N 是该并发下的聚合; 聚合随 N 饱和的那个值就是整卡带宽

关键: 逐事件速率必须取并发 1 的值, 聚合必须取饱和值。现有常数全是在 28 核并发下
反解的单核值 (已含平均争用), 再叠到速率服务器上就是双重计费 —— 这个脚本的输出把
两者明确分开。

用法: python bench/analyze.py bench_mem.csv [--cycles-per-us 循环数]
cycles_per_us 不给时按 GetSystemCycle 的常见标定 ~310-353 ticks/us 提示你确认。
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys


def fit(xs, ys):
    """最小二乘 y = a + b x; 点数 < 2 时返回 (y, 0)."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0), 0.0
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if den == 0:
        return statistics.fmean(ys), 0.0
    b = (n * sxy - sx * sy) / den
    return (sy - b * sx) / n, b


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "case": r["case"],
                "cores": int(r["active_cores"]),
                "bytes": int(r["bytes"]),
                "rows": int(r["rows"]),
                "row_bytes": int(r["row_bytes"]),
                "row_stride": int(r["row_stride"]),
                "vec": int(r["vec_elems"]),
                "rep": int(r["rep"]),
                "core": int(r["core"]),
                "cyc": int(r["cycles"]),
            })
    return rows


def med_by(rows, keyf, *, drop_first_rep=True):
    """按 keyf 聚合取中位. 缺省丢掉 rep 0 (含冷启动)."""
    g = collections.defaultdict(list)
    for r in rows:
        if drop_first_rep and r["rep"] == 0:
            continue
        g[keyf(r)].append(r["cyc"])
    return {k: statistics.median(v) for k, v in sorted(g.items()) if v}


def report_continuous(rows, case, cpus, label):
    sel = [r for r in rows if r["case"] == case]
    if not sel:
        return
    print(f"\n=== {case}  ({label}) ===")
    print(f"{'并发':>4} {'截距(us)':>10} {'带宽(B/us)':>12} {'聚合(B/us)':>12}")
    agg = {}
    for nc in sorted({r["cores"] for r in sel}):
        pts = med_by([r for r in sel if r["cores"] == nc], lambda r: r["bytes"])
        xs = list(pts)
        ys = [pts[x] / cpus for x in xs]          # cycle -> us
        a, b = fit(xs, ys)
        bw = (1.0 / b) if b > 0 else float("inf")
        agg[nc] = bw * nc
        print(f"{nc:>4} {a:>10.4f} {bw:>12.0f} {agg[nc]:>12.0f}")
    single = agg.get(1)
    sat = max(agg.values()) if agg else 0.0
    print(f"  -> 逐事件速率 (无争用单核, 并发=1) = {(single or 0)/1:.0f} B/us")
    print(f"  -> 整卡聚合 (随并发饱和的最大值)   = {sat:.0f} B/us")
    if single:
        print(f"  -> 聚合/单核 = {sat/single:.1f}  (若 << 核数, 说明单核就能吃掉很大一份,"
              f" 速率服务器的两个常数必须分别用这两个值, 不能一个顶两个)")


def report_burst(rows, case, cpus, label):
    """分离每次突发的固定开销与每字节代价: t = a + rows*(c_req + rowBytes*c_byte)."""
    sel = [r for r in rows if r["case"] == case]
    if not sel:
        return
    print(f"\n=== {case}  ({label}) ===")
    for nc in sorted({r["cores"] for r in sel}):
        sub = [r for r in sel if r["cores"] == nc]
        # 扫 rows (rowBytes 固定在最常见值)
        rb_mode = statistics.mode([r["row_bytes"] for r in sub])
        by_rows = med_by([r for r in sub if r["row_bytes"] == rb_mode],
                         lambda r: r["rows"])
        # 扫 rowBytes (rows 固定在最常见值)
        rows_mode = statistics.mode([r["rows"] for r in sub])
        by_rb = med_by([r for r in sub if r["rows"] == rows_mode],
                       lambda r: r["row_bytes"])
        if len(by_rows) < 2 or len(by_rb) < 2:
            continue
        a_r, b_r = fit(list(by_rows), [by_rows[x] / cpus for x in by_rows])
        a_b, b_b = fit(list(by_rb), [by_rb[x] / cpus for x in by_rb])
        # b_r = 每多一行的代价 (含该行的字节); b_b = 每多一字节的代价 x rows_mode
        c_byte = b_b / rows_mode if rows_mode else 0.0
        c_req = b_r - rb_mode * c_byte
        print(f"  并发 {nc:>2}: 每行固定开销 {c_req*1000:8.1f} ns, "
              f"每字节 {c_byte*1000:7.4f} ns  (=> {1/c_byte if c_byte>0 else 0:.0f} B/us), "
              f"整体截距 {a_r:.4f} us")
        print(f"            8B/行 时固定开销占 {100*c_req/(c_req+8*c_byte):.0f}%, "
              f"512B/行 时占 {100*c_req/(c_req+512*c_byte):.0f}%"
              if c_req + 8 * c_byte > 0 else "")


def report_vector(rows, cpus):
    sel = [r for r in rows if r["case"] == "vector_only"]
    if not sel:
        return
    print("\n=== vector_only  (纯 Exp/Div/Mul, 判 ACT 是带宽绑定还是 FLOP 绑定) ===")
    for nc in sorted({r["cores"] for r in sel}):
        pts = med_by([r for r in sel if r["cores"] == nc], lambda r: r["vec"])
        xs = list(pts)
        ys = [pts[x] / cpus for x in xs]
        a, b = fit(xs, ys)
        print(f"  并发 {nc:>2}: 截距 {a:.4f} us, 每元素 {b*1000:.5f} ns "
              f"(每 64 元素向量 {b*64*1000:.2f} ns)")
    print("  对照: ACT 每向量的 UB 字节是 722B; 若上面的"
          "每向量耗时 >= 722/BW_UB, 说明 ACT 是 FLOP 绑定而不是带宽绑定")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--cycles-per-us", type=float, default=0.0)
    args = ap.parse_args()
    rows = load(args.csv)
    if not rows:
        print("CSV 为空", file=sys.stderr)
        return 1
    cpus = args.cycles_per_us
    if cpus <= 0:
        print("!! 未给 --cycles-per-us: 先用一个已知时长的 case 定标, 或查你那台机器的"
              "GetSystemCycle 频率 (megamoe 的 trace 标定是 ~310-353 ticks/us)。"
              "下面按 1 cycle = 1 us 输出, 只有相对关系有意义。", file=sys.stderr)
        cpus = 1.0

    report_continuous(rows, "gm_to_l1", cpus, "GMM 载入通路 -> BW_L1_GM")
    report_continuous(rows, "gm_to_ub", cpus, "读本卡内存 -> BW_LOCAL_GM")
    report_continuous(rows, "ub_to_gm", cpus, "写本卡内存 -> hbm_write 聚合")
    report_burst(rows, "strided_store", cpus, "一条指令多次突发")
    report_burst(rows, "scatter_store", cpus,
                 "rows 条独立指令 —— COMBINE/ACT-scale 的形态")
    report_vector(rows, cpus)
    print("\n填回模型的位置:")
    print("  BW_L1_GM        <- gm_to_l1 的并发=1 带宽 (现值 51900, 出处标着待重标)")
    print("  BW_LOCAL_GM     <- gm_to_ub 的并发=1 带宽 (现值 157000)")
    print("  hbm_write 聚合  <- ub_to_gm 的整卡聚合饱和值 (不是每核值 x 核数!)")
    print("  ACT 每行写开销  <- scatter_store 的每行固定开销 (8B/行 那档)")
    print("  COMBINE 每行    <- scatter_store 的每行固定开销 + 512B x 每字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
