#!/usr/bin/env python3
"""流水编排效率: 一个形状上, 每个流水旋钮值多少钱, 以及卡在什么上.

运行: python examples/run_pipeline_study.py

这张表回答的是"流水编排"那一类问题, 不是"算得快不算得快":
  * 五个 stage 之间怎么搭流水 (就绪粒度 / 片上驻留 / 槽数 / 波偏移)
  * 单核内部 load/cube/fix 三个相位怎么重叠 (相位流水 + 引擎队列深度)
  * 工作怎么落到核上 (晚绑定 vs 静态发牌 / 事件粒度)
  * 融合还是分段 (波间栅栏)

读表的方法 (每一列都要能说出因果, 说不出的收益不敢用):
  Δ%            负数是变快。
  AIC 忙碌%     = busy / (核数 x 墙钟)。**只在同一类行之间可比**: 开了相位流水的行
                把载入拆出去记到 MTE2 管道上 (一个 AI Core 一条 MTE2, GMM1 与 GMM2 的
                载入都占它), AIC 只剩计算份额, 于是这一列掉到 1.5% —— 工作量一点没少,
                是**记账口径**变了。拿拆相位的行和不拆的行比这一列会得出
                "流水让 Cube 闲下来了"的错结论。要看还剩多少可挖, 用
                rank_results[0]["bounds"]: 墙钟 / lower_us。
  不可免空闲     DAG 结构逼出来的: 依赖没到齐, 没活可干。只能改依赖结构, 调度救不了。
  可避免空闲     **有就绪的活却有核空着** —— 这是护栏, 不是 0 就说明这个方案的
                时长偏慢, 收益不可比。idle_decomposition 里逐段给出是哪些核在空、
                当时有哪些就绪事件在等, 直接拿去定位。
  关键路径卡在   把关键路径上的每一步按 critical_reason 归类:
                resource = 等核 (核不够或分配不均) / dependency = 等上游数据。
                两者的比例决定下一步该加并行度还是该改依赖。
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import moe_cost_model as m                                       # noqa: E402

SCENARIO = Path(__file__).with_name("scenario_basic.toml")

#: gmm1->activation 的同核是硬件强制 (Fixpipe 直给配对 AIV0), 不是编排选择。
def _edges(depth=1, **act_to_gmm2):
    return [dict(producer="gmm1", consumer="activation", location="onchip",
                 depth=depth, colocated_by_hardware=True),
            dict(producer="activation", consumer="gmm2", **act_to_gmm2)]


POINTS = {
    "基线 (场景文件原样)": {},
    # --- 单核内部: 三个相位重叠 ---
    "相位流水 (队列深 2)": {"options.pipeline": {"queues": {"mte_aic": 2, "cube": 2}}},
    "相位流水 (队列深 4)": {"options.pipeline": {"queues": {"mte_aic": 4, "cube": 4}}},
    # --- stage 之间: 就绪粒度与片上驻留 ---
    "GMM2 逐 K 块就绪": {"options.links": _edges(readiness="per_chunk")},
    "GMM2 首块先开工": {"options.links": _edges(readiness="first_chunk")},
    "GMM2 均分 4 段就绪": {"options.links": _edges(readiness=4)},
    "UB 深度 2": {"options.links": _edges(2)},
    "UB 不设限 (上界)": {"options.links": _edges(0)},
    "ACT 不物化 (留片上)": {"options.links": _edges(location="onchip")},
    # --- 波之间: 偏移 ---
    "dispatch 超前 4 波": {"policy.dispatch_lookahead": 4},
    "GMM2 滞后 1 波": {"policy.gmm2_lag_waves": 1},
    # --- 工作落到核上 ---
    "晚绑定 (AIC)": {"options.late_bind_pools": ["AIC"]},
    "晚绑定 (AIC+AIV1)": {"options.late_bind_pools": ["AIC", "AIV1"]},
    "GMM2 攒 2 个 tile": {"options.granularity": {"gmm2": 2}},
    # --- 融合 vs 分段 ---
    "波间全核栅栏 (分段)": {"options.barriers": ["wave"]},
}


def _row(label, res, base_us):
    rr = res["rank_results"][0]
    total = res["kernel_total_us"]
    aic = rr["idle_decomposition"].get("R0.AIC")
    busy_pct = 100.0 * aic.busy_us / aic.capacity_us if aic else float("nan")
    forced = aic.forced_idle_us if aic else float("nan")
    avoid = aic.avoidable_idle_us if aic else float("nan")
    why = collections.Counter(
        step["critical_reason"].split(":")[0] for step in rr["critical_path"])
    top = " ".join(f"{k}{v}" for k, v in why.most_common(3))
    delta = 0.0 if base_us is None else (total - base_us) / base_us * 100.0
    guard = "ok" if avoid == 0 else f"违反 {avoid:.0f}"
    return (f"{label:22s} {total:9.2f} {delta:+7.2f}%  "
            f"AIC忙碌 {busy_pct:5.1f}%  不可免 {forced:8.1f}  "
            f"可避免 {guard:>10s}  关键路径 {top}")


def main() -> int:
    base_scenario = m.load_scenario(SCENARIO)
    base_us = None
    print(f"场景: {SCENARIO.name}  (每一行只改一个旋钮)\n")
    for label, overrides in POINTS.items():
        try:
            res = m.simulate(base_scenario.with_overrides(overrides))
        except ValueError as exc:                 # 物理上拒绝的组合也要看得见
            print(f"{label:22s} 被拒绝: {exc}")
            continue
        if base_us is None:
            base_us = res["kernel_total_us"]
        print(_row(label, res, base_us))
    print("\n可避免空闲不为 0 的行: 去 idle_decomposition['R0.AIC'].segments 看逐段是哪些核在空、"
          "\n当时有哪些就绪事件在等 —— 那是下一步该动的地方。")
    print("注意 1: 开了相位流水的行, AIC 忙碌% 与别的行不可比 (载入改记到 MTE2 管道)。")
    print("注意 3: 2026-10-05 之前这张表把相位流水报成 -30.24% —— 那时载入不占任何资源,")
    print("        等于把搬运算成免费。修掉之后真实收益 0.00% (与基线逐位相同):")
    print("        这个形状是带宽绑定, 把载入与计算重叠不会让载入变快。")
    print("注意 2: 本模型**不建模带宽争用**, 所以流水做得越满, 低估越多 (见 README 精度边界)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
