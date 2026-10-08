#!/usr/bin/env python3
"""参数覆盖审计: 每一个可调参数, 模型到底看不看得见.

动机 —— 算法工程师改编排 / 编译期 / 运行期参数时, 一个 "0 收益" 有四种意思, 它们
对下一步的指示完全相反, 所以必须分开:

  生效      改了它, DAG 事件图或时长真的变了 -> 这个取舍可以照着做
  需对的形状 它在**某些**形状上生效, 本形状没有作用对象 (只有一波就谈不上超前几波,
            只有一个 K 块就谈不上逐块就绪) -> 换形状再扫, 不是模型不建模
  被拒      模型显式拒绝这个取值 (缺标定常数 / 这条路径没实现) -> 拒绝是诚实的
  未建模    模型里**没有读者**, 任何形状都不会变 -> 这是陷阱: 扫出来的 "0 收益"
            会被当成 "硬件上也没收益"。审计的硬结论就是让这一类无处藏身

扫法三条:
  1. 从**本场景的生效值**出发扰动, 不是从 dataclass 缺省值出发 —— 场景带 profile 时
     两者不同, 拿缺省值当基线会把 "值根本没变" 误判成 "没有读者"。
  2. 一个参数给**一串**候选取值, 任一取值动了就算生效 —— 翻倍常落在无语义的档上
     (l1_buf_num 2->4 与 2 同构, 2->1 才是关 ping-pong)。
  3. 有**前置条件**的参数带 CONTEXT: 它的作用对象依赖另一个参数。前置条件同时加到基线与
     扰动上, 比的是同一前提下的两个点。(当前为空 —— 唯一用过它的
     EngineQueueDepths 已经因为"在这个事件代数里无可表达的后果"被删掉。)

比对五项: 墙钟 / 事件数 / 事件名集合 / 逐事件时长 / 逐信道字节。五项全同 = 没动静。

判定固定在 EXPECTED 里, tests/test_knob_coverage.py 守它: 新参数忘接线、老参数被改没了、
"动不了"的声明过期了, 三种都会红。参数树自动走 (dataclasses.fields + scenario._NESTED),
所以新字段自动进审计。

运行: python tools/knob_audit.py            全量 (五形状, 分钟级)
      python tools/knob_audit.py --quiet    只列非"每形状都生效"的
      python tools/knob_audit.py --emit     重新生成 EXPECTED (贴回本文件)
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import moe_cost_model as m                                        # noqa: E402
from moe_cost_model import load_scenario                          # noqa: E402
from moe_cost_model.scenario import _NESTED                       # noqa: E402

SCENARIO = Path(__file__).resolve().parents[1] / "examples" / "scenario_basic.toml"

#: 五个互补形状。每个都给某一类参数作用对象, 合起来覆盖"波 / 组 / tile / K 块 / 核"
#: 这五个维度的多与少。hidden_dim=2048 -> GMM2 的 K=1024, 4 个 kL1 块 (逐块就绪有对象)。
SHAPES = {
    "多波核紧": {"workload.tokens": 1024, "workload.world": 2,
                 "workload.local_experts": 4, "aic_num": 2,
                 "h": 1024, "hidden_dim": 2048, "p1_override": 4},
    "单波核宽": {"workload.tokens": 64, "workload.world": 2,
                 "workload.local_experts": 4, "aic_num": 8,
                 "h": 1024, "hidden_dim": 2048, "p1_override": 1},
    # 多核 + 不均衡路由: 分波 / 分核 / 晚绑定有作用对象 (均衡路由下各策略等价)。
    # hidden_dim 取 1024: 列块少, tile->核 的分配差异才不被 K 维的量掩掉。
    "多核不均": {"workload.tokens": 1024, "workload.world": 2,
                 "workload.local_experts": 8, "aic_num": 16,
                 "h": 1024, "hidden_dim": 1024, "p1_override": 2,
                 "workload.routing": "random"},
    # 核数不整分 tile 数: cursor 共振 (GMM2 把游标正好推回起点) 只在这里出现
    "核非整分": {"workload.tokens": 256, "workload.world": 2,
                 "workload.local_experts": 4, "aic_num": 3,
                 "h": 1024, "hidden_dim": 1024, "p1_override": 2},
    "URMA": {"workload.tokens": 512, "workload.world": 2,
             "workload.local_experts": 4, "aic_num": 4,
             "h": 1024, "hidden_dim": 2048, "p1_override": 2,
             "kernel.topo_urma": True},
}

#: 候选取值 (按顺序试, 任一动了就算生效)。给出理由的三类:
#:   非标量 (对象数组 / 映射 / 策略名) —— 自动扰动给不出取值
#:   生效值是 None —— 翻倍给不出取值
#:   翻倍落在无语义的档上 —— 取另一个有语义的档
CANDIDATES = {
    "options.links": [[dict(producer="gmm1", consumer="activation", location="onchip",
                            depth=1, colocated_by_hardware=True),
                       dict(producer="activation", consumer="gmm2",
                            readiness="per_chunk")]],
    "options.granularity": [{"gmm1": 2}, {"combine": 2}],
    "options.roles": [{"combine": "AIV0"}, {"activation": "AIV1"}],
    "options.late_bind_pools": [["AIC"]],
    "options.barriers": [["wave"]],
    "options.combine_granularity": ["per_expert"],
    "options.combine_layout": ["expert_contiguous"],
    "options.dispatch_pacing": ["per_core", "wave"],
    "options.dispatch_partition": ["precut", "pooled"],
    "wave_packing": ["balanced_waves", "longest_expert_first"],
    "core_assignment": ["contiguous_block", "greedy_least_busy"],
    "scheduling_policy": ["priority_by_stage", "critical_path_first"],
    "options.gmm2_kl1": [128],
    "options.pipeline": [{"queues": {"mte_aic": 2}}],
    "policy.gmm2_lag_waves": [1],
    "policy.gmm2_combine_credit": [1, 2],
    "policy.wave_offsets": [{"dispatch": 3, "gmm2": -1}],
    "kernel.l1_buf_num": [1, 3],          # 1 = 关 ping-pong; 4 与 2 同构
    "kernel.swizzle_offset": [1, 2],
    # 缺省已是 0 (与 kernel 的 BlockSchedulerSwizzle<3,0> 一致, 2026-10-05 更正),
    # 所以要扫的是**另一个**取值 1 —— 扫 0 等于什么都没扫。
    "kernel.swizzle_direction": [1],      # 1 = N 维在外层
    "kernel.l1_size": [256 * 1024],       # 减半才可能翻转 select_kl1 的 can_double
    "kernel.combine_quant_mode": [1],
    "kernel.activation_n_half": [1],
    "kernel.topk_weights_prefetch": [True],   # epilogue 行块 256->128 + GM 往返
    "policy.gmm2_lag_threshold": [512],   # 阈值要跨过本形状的 token 数才有作用
    "options.epilogue_overheads": [{"literal": True}, {"counts_export_us": 5.0}],
}

#: 前置条件: 这个参数的作用对象依赖另一个参数。同时加到基线与扰动上。
CONTEXT = {}

#: 已核实"模型里动不了"的参数 -> 理由。审计的硬判据: 可疑集合必须是它的子集,
#: 多出来一个就是新的无作用的参数 (或者某个形状不再给它作用对象了), 两种都要人看。
DECLARED_UNREAD = {
    # 写侧: 布局只经 spread_slots 进 AnalyticalCombineCosts.scatter_us, 而那里
    #   额外时长 = m · scatter_us_per_row · (spread_slots/m) ** scatter_exponent。
    #   scatter_exponent 缺省 0 -> 指数项恒 1 -> 两种布局算出同一个数, 系数给多大都一样。
    #   exponent 不是保守取 0, 是实测把"落点跨度"这个机制否掉了 (costs.py 的
    #   AnalyticalCombineCosts 文档: bs128 与 bs8192 跨度差 64 倍而更稀的那个反而快一倍)。
    # 读侧: UNPERMUTE 从顺序读变 gather 的代价**完全没建模** (字节 / BW_UNPERMUTE_AGG
    #   一个除法, 与落点无关) —— 见 docs/design_space_gaps.md 缺口 10。
    # 所以扫这个参数只会得到 0, 那是模型的空白, 不是硬件上没有差别。
    "options.combine_layout": "写侧要 scatter_exponent>0 (实测已否掉该机制), "
                              "读侧 UNPERMUTE gather 未建模 (缺口 10)",
}

#: 全量扫描 (五个形状) 的判定, 固定在这里。tools/knob_audit.py --emit 重新生成。
#:   "生效"   每个形状上都动 —— 它与形状无关
#:   "生效*"  至少一个形状上动 —— 需要对的形状才有作用对象 (见 SHAPES 的注释)
#:   "被拒"   候选取值被模型显式拒绝 (缺标定 / 这条路径没实现)
#:   "动不了" 已核实在模型里没有可表达的后果 —— 理由见 DECLARED_UNREAD
EXPECTED = {
    "core_assignment": "生效*",
    "kernel.activation_n_half": "生效",
    "kernel.combine_meta_bytes_per_row": "生效",
    "kernel.combine_quant_mode": "生效",
    "kernel.gmm1_b_reuse_frac": "生效*",
    "kernel.gmm1_interleaved": "生效",
    "kernel.l1_buf_num": "生效",
    "kernel.l1_size": "生效*",
    "kernel.l1_tile_k": "生效*",
    "kernel.swizzle_direction": "生效*",
    "kernel.swizzle_offset": "生效*",
    "kernel.tile_m": "生效*",
    "kernel.tile_n": "生效",
    "kernel.topk_weights_prefetch": "生效",
    "kernel.topo_urma": "生效",
    "kernel.weight_nz": "被拒",
    "options.barriers": "生效",
    "options.combine_granularity": "生效*",
    "options.combine_layout": "动不了",
    "options.dispatch_pacing": "生效*",
    "options.dispatch_partition": "生效*",
    "options.dispatch_rows_per_item": "生效*",
    "options.epilogue_overheads": "生效",
    "options.gmm2_kl1": "生效",
    "options.granularity": "生效*",
    "options.late_bind_pools": "生效*",
    "options.links": "生效*",
    "options.m_groups_per_wave": "生效*",
    "options.pipeline": "生效",
    "options.roles": "生效*",
    "options.serialize_dispatch_comm": "生效*",
    "policy.cursor_resonance_fix": "生效*",
    "policy.dispatch_lookahead": "生效*",
    "policy.gmm2_combine_credit": "生效*",
    "policy.gmm2_lag_threshold": "生效*",
    "policy.gmm2_lag_waves": "生效*",
    "policy.wave_offsets": "生效*",
    "scheduling_policy": "生效*",
    "wave_packing": "生效*",
}

#: 不参与审计: 不是参数 (标定常数 / 输入形状 / 真值文件 / 直接给对象)。
SKIP_PREFIX = ("calibration.", "workload.", "tiling.", "costs")
SKIP = {"name", "profile", "platform", "h", "hidden_dim", "aic_num", "tiling",
        "p1_override", "p2_override", "restructure", "tile_grid", "orchestration"}


def auto_candidates(value):
    """从生效值出发给"必然不同"的取值; 给不出来返回 []."""
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value * 2 if value else 1]
    if isinstance(value, float):
        return [value + 0.5 if value == 0.0 else value * 0.5]
    return []


def knobs(obj, prefix="", out=None):
    """走参数树, 收 (点分路径, 生效值)。嵌套对象按 scenario._NESTED 下探,
    但带 CANDIDATES 的整体换掉 (links / pipeline 这种要整个对象一起给)。"""
    out = [] if out is None else out
    for f in dc.fields(obj):
        path = f"{prefix}{f.name}"
        if path in SKIP or path.startswith(SKIP_PREFIX):
            continue
        val = getattr(obj, f.name)
        if (path not in CANDIDATES and (type(obj), f.name) in _NESTED
                and dc.is_dataclass(val)):
            knobs(val, path + ".", out)
            continue
        out.append((path, val))
    return out


def signature(scenario):
    r = m.simulate(scenario)["rank_results"][0]
    ev = r["events"]

    def h(x):
        return hashlib.sha256(repr(x).encode()).hexdigest()[:10]

    return (round(r["total_us"], 6), len(ev),
            h(tuple(sorted(e.name for e in ev))),
            h(tuple(sorted((e.name, round(e.end_us - e.start_us, 6)) for e in ev))),
            h({k: round(v, 3) for k, v in sorted(r["traffic_bytes"].items())}))


def diff(base, got):
    out = []
    if got[0] != base[0]:
        out.append(f"墙钟 {base[0]:.1f}->{got[0]:.1f}")
    if got[1] != base[1]:
        out.append(f"事件数 {base[1]}->{got[1]}")
    if got[2] != base[2]:
        out.append("事件名")
    elif got[3] != base[3]:
        out.append("时长")
    if got[4] != base[4]:
        out.append("字节")
    return out


def audit_shape(base_scenario, overrides):
    scenario = base_scenario.with_overrides(overrides)
    rows, cache = {}, {}
    for path, val in knobs(scenario):
        cands = CANDIDATES.get(path) or auto_candidates(val)
        if not cands:
            rows[path] = ("跳过", f"无候选取值 (生效值 {val!r})")
            continue
        ctx = CONTEXT.get(path) or {}
        key = repr(sorted(ctx.items()))
        if key not in cache:
            cache[key] = signature(scenario.with_overrides(ctx) if ctx else scenario)
        base = cache[key]
        outcome = None
        for cand in cands:
            try:
                got = signature(scenario.with_overrides({**ctx, path: cand}))
            except Exception as exc:
                head = str(exc).splitlines()[0]
                outcome = outcome or ("被拒", f"{type(exc).__name__}: {head[:60]}")
                continue
            d = diff(base, got)
            if d:
                outcome = ("生效", ", ".join(d))
                break
            outcome = ("无动静", f"{val!r} -> {cand!r}")
        rows[path] = outcome
    return cache[repr(sorted({}.items()))], rows


VERDICT = {"生效": "生效", "生效 (需对的形状)": "生效*",
           "被拒 (该取值未实现/缺标定)": "被拒", "跳过": "跳过", "未建模(疑)": "动不了"}


def verdict_of(cells):
    kinds = {c[0] for c in cells}
    if kinds == {"生效"}:
        return "生效"
    if "生效" in kinds:
        return "生效 (需对的形状)"
    if "被拒" in kinds:
        return "被拒 (该取值未实现/缺标定)"
    if kinds == {"跳过"}:
        return "跳过"
    return "未建模(疑)"


#: 参数表里一行的"作用": 路径 -> 一句话。缺了就在 --markdown 时报错 ——
#: README 的表由代码生成, 新参数必须在这里给一句说明, 否则表里会出现空格子。
WHAT = {
    "core_assignment": "tile 分给哪个核 (三种策略)",
    "scheduling_policy": "就绪集里谁先跑 (三种策略)",
    "wave_packing": "专家怎么组成波 (三种策略)",
    "kernel.activation_n_half": "SwiGLU 的投影数 (gate+up)",
    "kernel.combine_meta_bytes_per_row": "COMBINE 每行搬几字节路由元数据",
    "kernel.combine_quant_mode": "COMBINE 的数据格式 (BF16 / FP8+scale)",
    "kernel.gmm1_b_reuse_frac": "非首个 m-group 的 tile 付几成 B 流",
    "kernel.gmm1_interleaved": "GMM1 的 gate/up 是否在 tile 内按列交织",
    "kernel.l1_buf_num": "L1 ping-pong 缓冲块数 (1 = 关)",
    "kernel.l1_size": "L1 容量 (进 select_kl1 的容量判据)",
    "kernel.l1_tile_k": "K 窗基线",
    "kernel.swizzle_direction": "tile 遍历的外层维 (0 = M 在外)",
    "kernel.swizzle_offset": "swizzle 的分组宽度",
    "kernel.tile_m": "一个 m-group 的行数",
    "kernel.tile_n": "一个 N-tile 的列数",
    "kernel.topk_weights_prefetch": "topk 权重在 epilogue 里乘; 行块 256->128 且 GMM1 输出走 GM 往返",
    "kernel.topo_urma": "通信路径: MTE 波循环 / URMA Layered 宏波循环 (换建图代码)",
    "kernel.weight_nz": "权重 GM 布局 Z / NZ (开启须显式给 NZ 带宽)",
    "options.barriers": "全核栅栏: 不加 / 波间 / 波内每 stage 后",
    "options.combine_granularity": "一个 COMBINE 事件覆盖几个 GMM2 tile",
    "options.combine_layout": "COMBINE 写出的落点跨度",
    "options.dispatch_pacing": "dispatch 的发起配速",
    "options.dispatch_partition": "dispatch 的行按核预切还是不预切",
    "options.dispatch_rows_per_item": "一份 dispatch 工作覆盖多少行",
    "options.epilogue_overheads": "尾段五项固定开销",
    "options.gmm2_kl1": "GMM2 的 kL1 (不给则自适应)",
    "options.granularity": "每个 stage 一个事件覆盖多少个单元",
    "options.late_bind_pools": "哪些引擎晚绑定 (派发时刻才定核)",
    "options.links": "stage 边: 就绪粒度 / 落点 / 片上槽数",
    "options.m_groups_per_wave": "波宽: 每波装几个 m-group",
    "options.pipeline": "相位拆分 (load/cube/fix) + 每核队列深度",
    "options.roles": "哪个 stage 跑在哪个引擎角色上",
    "options.serialize_dispatch_comm": "跨卡搬运是否串行化",
    "policy.cursor_resonance_fix": "游标共振修正",
    "policy.dispatch_lookahead": "dispatch 超前几波",
    "policy.gmm2_combine_credit": "GMM2->COMBINE 的固定 credit",
    "policy.gmm2_lag_threshold": "GMM2 滞后生效的 token 阈值",
    "policy.gmm2_lag_waves": "GMM2 滞后几波",
    "policy.wave_offsets": "各 stage 的波偏移组合",
}


def markdown_table() -> str:
    """按 EXPECTED 生成 README 里那张参数表 (代码是唯一来源)."""
    missing = sorted(set(EXPECTED) - set(WHAT))
    if missing:
        raise SystemExit(f"knob_audit.WHAT 缺这些参数的说明: {missing}")
    stale = sorted(set(WHAT) - set(EXPECTED))
    if stale:
        raise SystemExit(f"knob_audit.WHAT 里有已不存在的参数: {stale}")
    out = ["| 参数 | 判定 | 作用 |", "| --- | --- | --- |"]
    for path in sorted(EXPECTED):
        out.append(f"| `{path}` | {EXPECTED[path]} | {WHAT[path]} |")
    return "\n".join(out)


def main(argv):
    if "-h" in argv or "--help" in argv:
        # 不认 --help 的后果是: 新用户想看用法, 却触发一次几分钟的全量扫描。
        print(__doc__.strip())
        return 0
    if "--markdown" in argv:
        print(markdown_table())
        return 0
    quiet = "--quiet" in argv
    emit = "--emit" in argv
    scenario = load_scenario(SCENARIO)
    results, tags = {}, list(SHAPES)
    for tag in tags:
        base, results[tag] = audit_shape(scenario, SHAPES[tag])
        print(f"# 形状 {tag}: 墙钟 {base[0]:.1f} µs, {base[1]} 事件")
    paths = sorted(set().union(*(r.keys() for r in results.values())))
    print("\n" + "参数".ljust(44) + "".join(t.ljust(26) for t in tags) + "判定")
    suspect, emitted = [], {}
    for p in paths:
        cells = [results[t].get(p, ("—", "")) for t in tags]
        verdict = verdict_of(cells)
        emitted[p] = VERDICT[verdict]
        if verdict == "未建模(疑)":
            suspect.append(p)
        if quiet and verdict.startswith("生效"):
            continue
        print(p.ljust(44) + "".join((c[0] + (": " + c[1][:17] if c[1] else "")).ljust(26)
                                    for c in cells) + verdict)
    print(f"\n全部形状都无动静, 需人核读者: {len(suspect)}")
    for p in suspect:
        print("  ", p, "—", DECLARED_UNREAD.get(p, "未声明"))
    if emit:
        print("\nEXPECTED = {")
        for p, v in emitted.items():
            print(f"    {p + chr(34)*0!r}: {v!r},".replace("'" + p + "'", f'"{p}"'))
        print("}")
    mismatched = {p: (emitted[p], EXPECTED.get(p))
                  for p in set(emitted) | set(EXPECTED)
                  if emitted.get(p) != EXPECTED.get(p)}
    if mismatched:
        print("\n与约束的 EXPECTED 不符 (新参数 / 删掉的参数 / 判定变了):")
        for p, (now, was) in sorted(mismatched.items()):
            print(f"   {p}: 现在 {now!r}, 钉的是 {was!r}")
    undeclared = [p for p in suspect if p not in DECLARED_UNREAD]
    return 1 if (undeclared or (mismatched and not emit)) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
