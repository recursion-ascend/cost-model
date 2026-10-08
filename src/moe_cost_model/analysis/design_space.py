"""设计空间扫描: 给若干编排方案, 回答"这个编排选择的代价, 钱花在哪".

这是模型对算子工程师的直接交付物。工程师手上的问题从来不是"这个配置多少 us",
而是:

  * 我把这条边改成逐块就绪, 能省多少? 省在哪个 stage?
  * 省下来之后瓶颈换人了吗? 换成了什么等待?
  * 我少搬了多少字节? 有没有把核闲出来?
  * 哪个方案在我的形状上根本不值得做?

所以每一行给的不是一个数, 而是: 时长差 / 哪个 stage 变了 / 关键路径上的等待构成 /
访存量 / 可避免空闲。最后一项是一致性检查 —— 它不为 0 说明这个方案下模型自己的不变量
没守住 (有就绪的活却有核空闲), 那一行的收益不能信。

用法:

    from moe_cost_model import design_space, format_design_space, StageLink as L

    def run(options):
        return m.simulate_routing_counts(..., options=options)

    rows = design_space(run, {
        "基线":        m.ModelOptions(),
        "逐 K 块就绪":  m.ModelOptions(links=(..., L("activation", "gmm2", readiness="per_chunk"))),
        "不物化":       m.ModelOptions(links=(..., L("activation", "gmm2", location="onchip"))),
        "分段式执行":   m.ModelOptions(barriers=("stage",)),
    })
    print(format_design_space(rows))
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Tuple

from .critical_path import critical_path_breakdown, extract_critical_path


def _slowest(result: Mapping) -> Dict:
    ranks = result["rank_results"]
    slow = max(ranks, key=lambda r: float(ranks[r]["total_us"]))
    return ranks[slow]


def _utilization_by_role(rr: Mapping) -> Dict[str, float]:
    """每个执行角色的平均利用率 = 该角色各核的忙碌之和 / 跨度之和.

    角色名从资源名里取 ("R0.AIC:7" -> "AIC"), 不写死哪几个角色 —— 换一份实现
    (config/stages 的词汇表) 角色集合就不同。按"忙碌和 / 跨度和"而不是逐核平均再
    取平均: 后者会让只干了一件事的核与满载的核等权。
    """
    busy = rr.get("resource_busy_us", {})
    span = rr.get("resource_span_us", {})
    agg: Dict[str, list] = {}
    for name, b in busy.items():
        if ":" not in name:
            continue                      # 非"角色:核号"的资源 (如 DISPATCH_COMM)
        role = name.rsplit(":", 1)[0].split(".")[-1]
        slot = agg.setdefault(role, [0.0, 0.0])
        slot[0] += float(b)
        slot[1] += float(span.get(name, 0.0))
    return {role: (b / sp if sp > 0 else 0.0) for role, (b, sp) in sorted(agg.items())}


def _late_bound_pools(result: Mapping) -> Optional[Tuple[str, ...]]:
    """这次运行哪些角色池是派发时刻绑定的 (ModelOptions.late_bind_pools).

    为什么这一行要进表: "有就绪的活却有核空闲"这个量的判读**取决于绑定方式**
    (analysis/idle.py 的判读规则)。
      晚绑定的池     avoidable 必须为 0 —— 不为 0 是模型自己的不变量没守住,
                     那一行的收益不可信;
      静态钉核的池   avoidable 是**真帐**: 它量的就是那种分核方式留下的可回收空闲,
                     是结论而不是缺陷 (换晚绑定就能回收)。
    两者混在一个"违反"里, 会把一个结论读成一个 bug。
    """
    scenario = result.get("scenario")
    opts = getattr(scenario, "options", None)
    pools = getattr(opts, "late_bind_pools", None)
    return tuple(pools) if pools is not None else None


def _point_row(result: Mapping) -> Dict[str, object]:
    """一个方案的可比指标 (与基线无关的部分)."""
    rr = _slowest(result)
    br = critical_path_breakdown(extract_critical_path(rr["events"]))
    avoidable = {k.rsplit(".", 1)[-1].rstrip(":"): v.avoidable_idle_us
                 for k, v in rr.get("idle_decomposition", {}).items()}
    # 物理下界: 同一形状换任何编排都不变, 所以"离下界还有多远"是跨方案可比的
    # 瓶颈判据 (analysis/bounds.py)。binding 说的是三条里哪条最高。
    bounds = dict(rr.get("bounds", {}) or {})
    lower = float(bounds.get("lower_us", 0.0) or 0.0)
    return {
        "total_us": float(rr["total_us"]),
        "dag_end_us": float(rr["dag_end_us"]),
        "utilization": _utilization_by_role(rr),
        "resource_idle_us": dict(rr.get("resource_idle_us", {})),
        "late_bind_pools": _late_bound_pools(result),
        "binding_bound": str(bounds.get("binding", "") or "-"),
        "lower_us": lower,
        "over_bound_pct": (100.0 * (float(rr["total_us"]) - lower) / lower
                           if lower > 0 else None),
        "stage_busy_us": dict(rr.get("stage_busy_us", {})),
        "wait_us": float(br["wait_us"]),
        "path_stage_us": dict(br["by_stage"]),
        "wait_by_mechanism": dict(br["by_mechanism"]),
        "traffic_bytes": dict(rr.get("traffic_bytes", {})),
        "avoidable_idle_us": avoidable,
        "events": len(rr["events"]),
    }


def design_space(run: Callable[[object], Mapping],
                 points: Mapping[str, object],
                 *, baseline: Optional[str] = None,
                 platform=None) -> List[Dict[str, object]]:
    """逐方案仿真并算出相对基线的差.

    run:      run(options) -> simulate_routing_counts 的返回值
    points:   {方案名: ModelOptions}; 保持插入序
    baseline: 用哪个方案做基线; 不给就取第一个
    platform: config.platform.PlatformSpec。给了就多算一项检查: 该方案申报的 GM
              访存量 / 墙钟 = 它需要的聚合带宽, 超过规格聚合 HBM 带宽就是**物理上
              不可能** —— 那一行的时长不可信 (模型不建带宽争用, 只能事后核对)。

    返回每个方案一行 (含 delta_us / delta_pct / 变化最大的 stage / 瓶颈等待构成 /
    访存量差 / 可避免空闲)。
    """
    if not points:
        raise ValueError("points 不能为空")
    names = list(points)
    base_name = baseline if baseline is not None else names[0]
    if base_name not in points:
        raise ValueError(f"baseline {base_name!r} 不在 points 里")

    raw = {name: _point_row(run(points[name])) for name in names}
    base = raw[base_name]
    rows: List[Dict[str, object]] = []
    for name in names:
        r = dict(raw[name])
        r["name"] = name
        r["is_baseline"] = (name == base_name)
        r["delta_us"] = r["total_us"] - base["total_us"]
        r["delta_pct"] = (100.0 * r["delta_us"] / base["total_us"]
                          if base["total_us"] else 0.0)
        moves = {st: r["stage_busy_us"].get(st, 0.0) - base["stage_busy_us"].get(st, 0.0)
                 for st in set(r["stage_busy_us"]) | set(base["stage_busy_us"])}
        r["stage_delta_us"] = {k: v for k, v in sorted(
            moves.items(), key=lambda kv: -abs(kv[1])) if abs(v) > 1e-9}
        # 关键路径上各 stage 占的时长差 —— 时长变了但总忙碌不变时, 变化就在这里:
        # 同样的工作量换了一条更短/更长的路径。
        pmoves = {st: r["path_stage_us"].get(st, 0.0) - base["path_stage_us"].get(st, 0.0)
                  for st in set(r["path_stage_us"]) | set(base["path_stage_us"])}
        r["path_stage_delta_us"] = {k: v for k, v in sorted(
            pmoves.items(), key=lambda kv: -abs(kv[1])) if abs(v) > 1e-9}
        r["traffic_delta_bytes"] = {
            k: r["traffic_bytes"].get(k, 0.0) - base["traffic_bytes"].get(k, 0.0)
            for k in set(r["traffic_bytes"]) | set(base["traffic_bytes"])}
        r["traffic_delta_bytes"] = {k: v for k, v in sorted(
            r["traffic_delta_bytes"].items(), key=lambda kv: -abs(kv[1]))
            if abs(v) > 1e-9}
        # 关键路径上最大的那种等待。"root" 是路径首事件的占位 (之前没有上游),
        # 等待恒为 0, 不是一种机制 —— 排掉它, 否则表里恒显示 "root 0us"。
        top = sorted(((k, v) for k, v in r["wait_by_mechanism"].items()
                      if k != "root" and v > 1e-9), key=lambda kv: -kv[1])
        r["top_wait"] = top[0] if top else ("-", 0.0)
        r["delta_wait_us"] = r["wait_us"] - base["wait_us"]
        r["utilization_delta"] = {
            k: r["utilization"].get(k, 0.0) - base["utilization"].get(k, 0.0)
            for k in set(r["utilization"]) | set(base["utilization"])}
        r["utilization_delta"] = {k: v for k, v in sorted(
            r["utilization_delta"].items(), key=lambda kv: -abs(kv[1]))
            if abs(v) > 1e-9}
        # 晚绑定的池: avoidable 不为 0 = 不变量没守住, 这一行的收益不可信。
        # 静态钉核的池: avoidable 是那种分核方式留下的可回收空闲 (结论, 不是缺陷)。
        # 绑定方式未知 (直接给 run 回调, 结果里没有 scenario) 时按原来的口径, 全算违规。
        late = r["late_bind_pools"]
        nonzero = {k: v for k, v in r["avoidable_idle_us"].items() if v > 1e-6}
        if late is None:
            bad, static = nonzero, {}
        else:
            bad = {k: v for k, v in nonzero.items() if k in late}
            static = {k: v for k, v in nonzero.items() if k not in late}
        r["invariant_ok"] = not bad
        r["invariant_violations"] = bad
        r["static_bind_idle_us"] = static
        # 带宽上界检查: 申报的 GM 访存量 / 墙钟 = 这个方案需要的聚合带宽
        gm = sum(v for k, v in r["traffic_bytes"].items() if k.endswith("gm_to_l1"))
        r["gm_bytes"] = gm
        r["gm_bw_needed"] = gm / r["total_us"] if r["total_us"] else 0.0
        if platform is not None:
            r["hbm_pct"] = 100.0 * r["gm_bw_needed"] / platform.hbm_bytes_per_us
            r["bandwidth_ok"] = r["hbm_pct"] <= 100.0
        else:
            r["hbm_pct"] = None
            r["bandwidth_ok"] = True
        rows.append(r)
    return rows


def compare_variants(scenario, variants: Mapping[str, Mapping[str, object]],
                     *, baseline: Optional[str] = None, platform=None,
                     check_bounds: bool = True) -> List[Dict[str, object]]:
    """比较若干**完整方案**: 编排、tiling 几何、分核/打包/调度策略都能换.

    design_space 的入口只换 ModelOptions, 所以 tile 几何 (kernel.tile_m / tile_n /
    l1_tile_k / l1_buf_num)、tile 切法 (tile_grid)、分核 (core_assignment)、波打包
    (wave_packing)、调度策略 (scheduling_policy) 这些**同样属于编排与 tiling 的维度
    换不了** —— 它们不在 ModelOptions 上, 而在 Scenario 上。这个入口以 Scenario 的
    点分路径为方案描述, 覆盖面因此与场景文件一致。

        rows = compare_variants(scenario, {
            "基线":          {},
            "tile_m 128":    {"kernel.tile_m": 128},
            "GMM2 粒度 2":   {"options.granularity": {"gmm2": 2}},
            "逐 K 块就绪":   {"options.links": [{"producer": "activation",
                                                 "consumer": "gmm2",
                                                 "readiness": "per_chunk"}]},
            "最闲核优先":    {"core_assignment": "greedy_least_busy"},
            "长专家优先":    {"wave_packing": "longest_expert_first"},
        })
        print(format_design_space(rows))

    每行给: 时长与差值、各执行角色的利用率、三条物理下界里哪条绑定与超出多少、
    关键路径上各 stage 的变化、总忙碌变化、最大的那种等待、访存量差、以及工作守恒
    不变量成立不成立 (不成立那一行的收益不能信)。

    check_bounds=False 只把下界记在结果里而不抛异常 —— 扫描时若某个方案穿透下界,
    那是模型漏算了代价, 不该让它直接中断整张表; 但那一行必须当成不可信。
    """
    from ..scenario import simulate as _simulate
    if not variants:
        raise ValueError("variants 不能为空")
    runs = {name: _simulate(scenario.with_overrides(dict(ov)), platform=platform,
                            check_bounds=check_bounds)
            for name, ov in variants.items()}
    return design_space(lambda name: runs[name], {n: n for n in variants},
                        baseline=baseline, platform=platform)


def format_design_space(rows: List[Dict[str, object]]) -> str:
    """把 design_space 的结果排成一张能直接读的表.

    收益 (负的 Δ 是变快), 哪个 stage 让它变快/变慢, 关键路径上最大的那种等待,
    GM 访存量差, 以及不变量是否成立。
    """
    def mb(x: float) -> str:
        return f"{x / 1e6:+.2f}MB" if abs(x) >= 1e4 else f"{x:+.0f}B"

    w = max(len(str(r["name"])) for r in rows)
    out = [f"{'方案'.ljust(w)}  {'时长':>9}  {'Δ':>9}  {'Δ%':>7}  {'利用率':<22}  "
           f"{'瓶颈':<16}  {'关键路径变化 (stage)':<26}  {'总忙碌变化':<18}  "
           f"{'最大等待':<16}  {'访存量差':<14}  不变量"]
    for r in rows:
        # 总忙碌不变 = 工作量没变, 变的是走的路 —— 这本身是结论, 不是缺数据
        busy = ", ".join(f"{k}{v:+.0f}" for k, v in
                         list(r["stage_delta_us"].items())[:2]) or "(工作量不变)"
        path = ", ".join(f"{k}{v:+.1f}" for k, v in
                         list(r["path_stage_delta_us"].items())[:3]) or "-"
        kind, amount = r["top_wait"]
        wait = f"{kind or '-'} {amount:.0f}us"
        tr = ", ".join(f"{k.split('.')[-1]}{mb(v)}"
                       for k, v in list(r["traffic_delta_bytes"].items())[:1]) or "-"
        if r["invariant_ok"]:
            guard = "ok"
        else:
            guard = "违反 " + ", ".join(f"{k} {v:.0f}"
                                        for k, v in r["invariant_violations"].items())
        static = r.get("static_bind_idle_us") or {}
        if static:
            guard += (" | 静态钉核可回收 "
                      + ", ".join(f"{k} {v:.0f}核·us" for k, v in static.items()))
        if r["hbm_pct"] is not None:
            guard += f" | HBM {r['hbm_pct']:.0f}%" + ("" if r["bandwidth_ok"] else " 超!")
        tag = " (基线)" if r["is_baseline"] else ""
        util = " ".join(f"{k}{100 * v:.0f}%" for k, v in r["utilization"].items()) or "-"
        over = r["over_bound_pct"]
        bound = (f"{r['binding_bound']} +{over:.0f}%" if over is not None
                 else r["binding_bound"])
        out.append(
            f"{str(r['name']).ljust(w)}  {r['total_us']:9.2f}  {r['delta_us']:+9.2f}  "
            f"{r['delta_pct']:+6.1f}%  {util:<22}  {bound:<16}  {path:<26}  "
            f"{busy:<18}  {wait:<16}  {tr:<14}  {guard}{tag}")
    return "\n".join(out)
