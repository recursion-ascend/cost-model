#!/usr/bin/env python3
"""未标定输入 -> 结论还站不站得住.

运行: python examples/run_uncertainty.py

模型吐出"换这个参数省 0.57%"看起来像结论, 但它下面垫着几个没标定到点的输入。
这张表把它们的区间推到每个 Δ 上: **区间跨 0 = 不可判定**, 别拿去做决策;
依赖"连范围都没有"的输入 = 区间本身不完整, 更不能用。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import moe_cost_model as m                                        # noqa: E402
from moe_cost_model.analysis.sensitivity import (RANGED,          # noqa: E402
                                                 propagate, report)

SCENARIO = Path(__file__).with_name("scenario_basic.toml")

#: 每个方案: 参数覆盖 + 它依赖哪些"连范围都没有"的输入
POINTS = {
    "GMM2 攒 2 个 tile": ({"options.granularity": {"gmm2": 2}}, ()),
    "combine 逐专家 (整片)": ({"options.granularity": {"combine": 0}}, ()),
    "晚绑定 (AIC+AIV1)": ({"options.late_bind_pools": ["AIC", "AIV1"]},
                          ("calibration.late_bind_fetch_us",)),
    "UB 深度 2": ({"options.links": [
        dict(producer="gmm1", consumer="activation", location="onchip",
             depth=2, colocated_by_hardware=True),
        dict(producer="activation", consumer="gmm2")]}, ()),
}


def main() -> int:
    base = m.load_scenario(SCENARIO)
    print(report())
    print()
    print(f"场景: {SCENARIO.name}\n")

    def delta_pct(knob, shifts):
        """这个方案相对基线的 Δ%, 在给定的未标定输入取值下."""
        b = m.simulate(base.with_overrides(dict(shifts)))["kernel_total_us"]
        ov = dict(knob)
        ov.update(shifts)
        t = m.simulate(base.with_overrides(ov))["kernel_total_us"]
        return (t - b) / b * 100.0

    for label, (knob, unknown_deps) in POINTS.items():
        iv = propagate(lambda shifts: delta_pct(knob, shifts),
                       depends_on_unknown=unknown_deps)
        verdict = "可判定" if iv.decidable else "**不可判定**"
        drv = f"  (区间主要由 {iv.driver} 撑开)" if iv.driver else ""
        print(f"{label:22s} Δ = {iv.format()}   {verdict}{drv}")

    print()
    print("读法: 点值是标定值下的结果; 方括号是把有区间的输入分别取两端重跑的 min/max")
    print("      (一次变一个, 不覆盖输入之间的交互)。")
    print("      '依赖未测量' 的行连区间都不完整 —— 那个量定下来之前, 这个比较没有答案。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
