#!/usr/bin/env python3
"""标定域审计: 这次运行用到的标定值, 哪些落在它们量过的范围内.

用法:
  python tools/calibration_domain.py                       # 缺省场景 + 打点语料形状各一份
  python tools/calibration_domain.py --scenario examples/scenario_basic.toml
  python tools/calibration_domain.py --check               # 有越域项则退出码 1

为什么需要 —— `config/hardware.py` 的标定常数是模块级全局量, 一套数覆盖所有实现/编译点/
形状/拓扑, 而那些常数自己的注释写明了它们不是这样的: BW_L1_GM 按核数重拟是 28 核 45300 /
18 核 37000 (1.40 倍), BW_REMOTE_WRITE 在三个形状上是 9.5/7.8/4.5 GB/s (2.11 倍),
URMA_GET_LAT_US 的域是"4 卡 3 条流, 超出未验证"。模型此前没有地方记这件事, 也没有地方在
越域时提醒。

**不做自动外推**: 越域时不给"修正值"。那些依赖关系只有两三个点, 凭它们造一条曲线再外推,
比直接说"超出标定域"更坏。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import moe_cost_model as m                                               # noqa: E402
from moe_cost_model.implementations.calibration import (CORPUS_COMPILE,   # noqa: E402
                                                        default_table)
from moe_cost_model.implementations.identity import ImplementationId      # noqa: E402

#: 打点语料的形状 (data/*/config.json5 的交集), 作为"域内"对照
CORPUS_SHAPE = {"token_num": 128, "h": 5120, "hidden_dim": 4608, "topk": 6,
                "local_experts": 3}
CORPUS_TOPO = {"world_size": 4, "active_cores": 28}


def audit(label: str, shape, topology, impl: str, fingerprint: str) -> int:
    table = default_table()
    rows = table.audit(implementation=ImplementationId.parse(impl),
                       compile_fingerprint=fingerprint, shape=shape, topology=topology)
    print(f"\n# {label}")
    print(f"  实现 {impl}  编译指纹 {fingerprint}")
    print(f"  形状 {shape}")
    print(f"  拓扑 {topology}")
    if not rows:
        print("  这个 (实现, 编译指纹) 下没有登记任何标定值 —— 用的是模块全局量, 无域可查")
        return 0
    print(f"  {'标定值':24s}{'判定':14s}{'离散度':>8s}  说明")
    bad = 0
    for name, got in rows.items():
        spread = got.record.spread
        spread_text = "-" if spread is None else f"{spread:.2f}x"
        note = ""
        if got.verdict == "out_of_domain":
            bad += 1
            parts = []
            if got.out_of_range:
                parts.append(f"形状越界 {list(got.out_of_range)}")
            if got.topology_mismatch:
                parts.append(f"拓扑不符 {list(got.topology_mismatch)}")
            note = "; ".join(parts)
        elif got.verdict == "undeclared":
            note = f"这些维从未声明过范围 {list(got.undeclared)}"
        elif got.verdict == "wrong_key":
            bad += 1
            note = got.note
        print(f"  {name:24s}{got.verdict:14s}{spread_text:>8s}  {note}")
        if spread and spread > 1.3 and got.verdict != "in_domain":
            print(f"      其它条件下的观测: "
                  + ", ".join(f"{cond}={val:g}" for cond, val in got.record.observations))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="", help="场景文件 (用它的形状与拓扑)")
    ap.add_argument("--check", action="store_true", help="有越域项则退出码 1")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    impl = "ascend950.megamoe.a8w8_wave.v1"
    bad = audit("打点语料的形状 (对照: 应当全部域内)", CORPUS_SHAPE, CORPUS_TOPO,
                impl, CORPUS_COMPILE.fingerprint)

    if args.scenario:
        scenario = m.load_scenario(args.scenario)
        shape = {"token_num": scenario.workload.tokens, "h": scenario.h,
                 "hidden_dim": scenario.hidden_dim, "topk": scenario.workload.topk,
                 "local_experts": scenario.workload.local_experts or 0}
        topo = {"world_size": scenario.workload.world or 0,
                "active_cores": scenario.aic_num}
        bad += audit(f"场景 {Path(args.scenario).name}", shape, topo, impl,
                     scenario.kernel and
                     __import__("moe_cost_model").CompileConfig
                     .from_kernel_config(scenario.kernel).fingerprint)
    print(f"\n# 越域/错键共 {bad} 项")
    if bad:
        print("  越域不等于结果没用, 但它意味着这些数的来源条件与本次运行不同; "
              "要么重新标定, 要么把结论当成带未知偏差的参考。模型不替你外推。")
    return 1 if (args.check and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
