"""设计空间扫描: 给若干编排方案, 回答"这个选择值多少钱, 钱花在哪".

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

from typing import Callable, Dict, List, Mapping, Optional

from .critical_path import critical_path_breakdown, extract_critical_path


def _slowest(result: Mapping) -> Dict:
    ranks = result["rank_results"]
    slow = max(ranks, key=lambda r: float(ranks[r]["total_us"]))
    return ranks[slow]


def _point_row(result: Mapping) -> Dict[str, object]:
    """一个方案的可比指标 (与基线无关的部分)."""
    rr = _slowest(result)
    br = critical_path_breakdown(extract_critical_path(rr["events"]))
    avoidable = {k.rsplit(".", 1)[-1].rstrip(":"): v.avoidable_idle_us
                 for k, v in rr.get("idle_decomposition", {}).items()}
    return {
        "total_us": float(rr["total_us"]),
        "dag_end_us": float(rr["dag_end_us"]),
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
        bad = {k: v for k, v in r["avoidable_idle_us"].items() if v > 1e-6}
        r["invariant_ok"] = not bad
        r["invariant_violations"] = bad
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


def format_design_space(rows: List[Dict[str, object]]) -> str:
    """把 design_space 的结果排成一张能直接读的表.

    收益 (负的 Δ 是变快), 哪个 stage 让它变快/变慢, 关键路径上最大的那种等待,
    GM 访存量差, 以及不变量是否成立。
    """
    def mb(x: float) -> str:
        return f"{x / 1e6:+.2f}MB" if abs(x) >= 1e4 else f"{x:+.0f}B"

    w = max(len(str(r["name"])) for r in rows)
    out = [f"{'方案'.ljust(w)}  {'时长':>9}  {'Δ':>9}  {'Δ%':>7}  "
           f"{'关键路径变化 (stage)':<26}  {'总忙碌变化':<18}  {'最大等待':<16}  "
           f"{'访存量差':<14}  不变量"]
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
        guard = "ok" if r["invariant_ok"] else \
            "违反 " + ", ".join(f"{k} {v:.0f}" for k, v in r["invariant_violations"].items())
        if r["hbm_pct"] is not None:
            guard += f" | HBM {r['hbm_pct']:.0f}%" + ("" if r["bandwidth_ok"] else " 超!")
        tag = " (基线)" if r["is_baseline"] else ""
        out.append(
            f"{str(r['name']).ljust(w)}  {r['total_us']:9.2f}  {r['delta_us']:+9.2f}  "
            f"{r['delta_pct']:+6.1f}%  {path:<26}  {busy:<18}  "
            f"{wait:<16}  {tr:<14}  {guard}{tag}")
    return "\n".join(out)
