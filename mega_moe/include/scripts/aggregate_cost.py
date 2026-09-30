#!/usr/bin/env python3
"""Aggregate prof_paired.csv of multiple runs into a cost-model Excel workbook.

Usage:
    python aggregate_cost.py --freq 1000 --output cost_model.xlsx RUN_DIR [RUN_DIR ...]

Each RUN_DIR must contain: config.json5, raw/prof_paired.csv, run.log (for
mGroupsPerWave). Sheets: runs / stage_summary / wave_stage / scaling / derived.
Time unit: microseconds (duration_cycles / freq).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

WAVE_STAGES = ("GMM1", "ACT_QUANT", "GMM2", "COMBINE", "DISPATCH_SCHEDULE",
               "WAIT_GMM1_INPUT", "WAIT_GMM1_BUFFER", "WAIT_ACT_INPUT",
               "WAIT_GMM2_INPUT", "WAIT_COMBINE_INPUT")
KEY_STAGES = ["DISPATCH_SCHEDULE", "DISPATCH_XFER", "DISPATCH_LOCAL", "INPUT_QUANT",
              "GMM1", "ACT_QUANT", "GMM2", "COMBINE", "UNPERMUTE", "FINALIZE"]


def stage_class(name):
    if name.startswith("WAIT_"):
        return "wait"
    if name in ("DISPATCH_XFER", "DISPATCH_LOCAL", "COMBINE", "ROUTE_SEND", "SYNC_RESET"):
        return "comm"
    if name in ("INIT", "FINALIZE", "INPUT_BUFFER_INIT", "OUTPUT_BUFFER_INIT",
                "DISPATCH_BUFFER_INIT", "TOKEN_COUNT_PREPARE", "COUNTS_EXPORT"):
        return "overhead"
    return "compute"


def load_run(run_dir, freq):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from config_io import load_config
    cfg = load_config(os.path.join(run_dir, "config.json5"))
    df = pd.read_csv(os.path.join(run_dir, "raw", "prof_paired.csv"))
    df["us"] = df["duration_cycles"] / freq
    case, dev = cfg["case"], cfg["device"]
    bs = case["tokens"]

    mg = None
    log = os.path.join(run_dir, "run.log")
    if os.path.exists(log):
        with open(log, errors="ignore") as f:
            m = re.search(r"mGroupsPerWave=(\d+)", f.read())
            mg = int(m.group(1)) if m else None

    tier = "S(bs<2048)" if bs < 2048 else ("L(bs>=16384)" if bs >= 16384 else "M(default)")
    waves = sorted({(int(p) >> 16) & 0xFF for p in df["payload"]
                    if df["event"].iloc[0] in ("KERNEL", "INIT")} ) # placeholder, refined below
    wave_mask = df["event"].isin(WAVE_STAGES)
    n_waves = int(df.loc[wave_mask, "payload"].map(lambda p: (int(p) >> 16) & 0xFF).max()) + 1 \
        if wave_mask.any() else 0

    meta = {
        "run_id": os.path.basename(run_dir.rstrip("/")),
        "bs": bs, "hidden": case["hidden"], "intermediate": case["intermediate"],
        "experts": case["experts"], "topk": case["topk"], "ep": case["ep"],
        "shared_experts": case.get("shared_experts", 0), "dtype": case["dtype"],
        "routing": case.get("routing", "random"), "seed": case.get("seed", 0),
        "mgw_override": int(cfg.get("cost_sweep", {}).get("mGroupsPerWaveOverride", 0)),
        "p1_override": int(cfg.get("cost_sweep", {}).get("p1Override", 0)),
        "p2_override": int(cfg.get("cost_sweep", {}).get("p2Override", 0)),
        "mGroupsPerWave": mg, "tier": tier, "n_waves": n_waves,
        "aic_cores": dev["aic_cores"], "aiv_cores": dev["aiv_cores"],
    }
    kern = df[df["event"] == "KERNEL"]
    meta["kernel_us"] = float(kern["us"].sum()) / len(kern["rank"].unique()) if len(kern) else np.nan
    # kernel_wall_us: 单次执行的真实墙钟 = 每卡（最晚 end - 最早 begin），跨卡平均。
    # 与 kernel_us（每卡 84 核区间时长之和，"核时"资源成本口径）区分。
    if len(kern):
        wall = (kern.groupby("rank").apply(lambda g: g["end_cycle"].max() - g["begin_cycle"].min()) / freq)
        meta["kernel_wall_us"] = float(wall.mean())
    else:
        meta["kernel_wall_us"] = np.nan
    return meta, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--freq", type=float, default=1000.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    metas, stage_rows, wave_rows = [], [], []
    for run_dir in args.runs:
        meta, df = load_run(run_dir, args.freq)
        metas.append(meta)
        g = df.groupby("event")["us"]
        for stage, s in g:
            stage_rows.append({
                "run_id": meta["run_id"], "bs": meta["bs"], "stage": stage,
                "class": stage_class(stage), "count": len(s), "sum_us": s.sum(),
                "mean_us": s.mean(), "p50_us": s.median(), "p90_us": s.quantile(0.9),
                "max_us": s.max(),
                "active_cores": df[df["event"] == stage]["local_id"].groupby(
                    df[df["event"] == stage]["role"]).apply(lambda x: x.nunique()).to_dict(),
            })
        wdf = df[df["event"].isin(WAVE_STAGES)].copy()
        if len(wdf):
            wdf["wave"] = wdf["payload"].map(lambda p: (int(p) >> 16) & 0xFF)
            for (stage, wave), s in wdf.groupby(["event", "wave"])["us"]:
                wave_rows.append({"run_id": meta["run_id"], "bs": meta["bs"], "stage": stage,
                                  "wave": wave, "count": len(s), "sum_us": s.sum(),
                                  "mean_us": s.mean()})

    runs = pd.DataFrame(metas)
    stage = pd.DataFrame(stage_rows)
    wave = pd.DataFrame(wave_rows)

    # scaling: per-wave mean for wave-carrying stages, per-run total for others.
    scale_rows = []
    for _, m in runs.iterrows():
        row = {"bs": m["bs"], "n_waves": m["n_waves"], "kernel_us": m["kernel_us"]}
        sub = stage[stage["run_id"] == m["run_id"]]
        for st in KEY_STAGES:
            e = sub[sub["stage"] == st]
            if not len(e):
                continue
            if st in WAVE_STAGES and m["n_waves"]:
                row[st + "_per_wave_us"] = float(e["sum_us"].iloc[0]) / m["n_waves"]
            row[st + "_total_us"] = float(e["sum_us"].iloc[0])
        scale_rows.append(row)
    scaling = pd.DataFrame(scale_rows).drop_duplicates(subset=["bs"]).sort_values("bs")

    # derived: per-m-group compute cost + dispatch transfer stats from payload bytes.
    der = []
    for _, m in runs.iterrows():
        sub = stage[(stage["run_id"] == m["run_id"]) & (stage["class"] == "compute")]
        d = {"run_id": m["run_id"], "bs": m["bs"]}
        for st in ("GMM1", "GMM2", "ACT_QUANT"):
            e = sub[sub["stage"] == st]
            if len(e) and m["mGroupsPerWave"]:
                d[st + "_us_per_mgroup"] = float(e["sum_us"].iloc[0]) / (
                    m["n_waves"] * m["mGroupsPerWave"])
        xfer = pd.read_csv(os.path.join(
            next(r for r in args.runs if os.path.basename(r.rstrip("/")) == m["run_id"]),
            "raw", "prof_paired.csv"))
        xfer = xfer[xfer["event"] == "DISPATCH_XFER"]
        if len(xfer):
            rows_ = xfer["payload"].map(lambda p: int(p) & 0xFFF).astype(float)
            d["xfer_mean_rows"] = rows_.mean()
            d["xfer_mean_us_per_record"] = xfer["duration_cycles"].mean() / args.freq
        der.append(d)
    derived = pd.DataFrame(der)

    with pd.ExcelWriter(args.output, engine="openpyxl") as xw:
        runs.to_excel(xw, "runs", index=False)
        stage.to_excel(xw, "stage_summary", index=False)
        wave.to_excel(xw, "wave_stage", index=False)
        scaling.to_excel(xw, "scaling", index=False)
        derived.to_excel(xw, "derived", index=False)
    print(f"[OK] {args.output}: {len(runs)} runs, {len(stage)} stage rows, {len(wave)} wave rows")


if __name__ == "__main__":
    main()
