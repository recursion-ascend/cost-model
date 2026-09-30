#!/usr/bin/env python3
"""把一个场景的调度结果渲染成可交互 HTML 报告.

用法:
  python tools/make_report.py examples/bs72_8rank.toml
  python tools/make_report.py examples/bs72_8rank.toml -o out.html --rates 1e7,2.7e7,5e7

报告内容: 执行时间、各 stage 时间线、逐核数据流时间线 (含同核排队)、关键路径、
路由矩阵。页面骨架在 tools/report_template.html, 数据以 JSON 注入其中的 /*DATA*/。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import moe_cost_model as m  # noqa: E402
from moe_cost_model import load_scenario, simulate  # noqa: E402
from moe_cost_model.model import A8W8WaveCostModel  # noqa: E402
from moe_cost_model.shape import MegaMoeShape  # noqa: E402

TEMPLATE = Path(__file__).with_name("report_template.html")
# 不入图的辅助事件: 零时长标记, 画出来只会挡住真正的事件
SKIP_STAGES = {"moe_stage_done", "dispatch_ready"}
STAGE_KEYS = ("dispatch_call", "dispatch", "gmm1", "activation", "gmm2", "combine", "epilogue")


def _strip_rank(name: str) -> str:
    return name.split(".", 1)[1]


def _shape_of(scenario, rank: int) -> MegaMoeShape:
    rows = scenario.workload.routing_counts()[rank]
    return MegaMoeShape(
        expert_tokens=tuple(sum(r) for r in rows), token_num=scenario.workload.tokens,
        h=scenario.h, hidden_dim=scenario.hidden_dim, aic_num=scenario.aic_num,
        rank_id=rank, p1_override=scenario.p1_override, p2_override=scenario.p2_override,
        expert_source_tokens=rows, dispatch_layout=m.DispatchDataLayout.from_hidden(scenario.h),
        topk=scenario.workload.topk, shared_expert_num=scenario.workload.shared_expert_num,
        kernel=scenario.kernel, policy=scenario.policy)


def _spec(scenario, counts, rank: int) -> dict:
    """表头规格与路由事实, 从场景与真实矩阵算出 (不写死)."""
    wl = scenario.workload
    world, local = len(counts), len(counts[0])
    sent = [sum(counts[d][e][s] for d in range(world) for e in range(local))
            for s in range(len(counts[0][0]))]
    recv = [sum(sum(r) for r in counts[d]) for d in range(world)]
    per_expert = sorted({sum(counts[d][e]) for d in range(world) for e in range(local)})
    cells = sorted({counts[d][e][s] for d in range(world) for e in range(local)
                    for s in range(len(counts[0][0]))})
    one = lambda xs: str(xs[0]) if len(xs) == 1 else f"{xs[0]}~{xs[-1]}"
    orch = "Layered 宏波循环" if scenario.kernel.topo_urma else "MTE 波循环"
    rows = [
        ("卡数", str(world)),
        ("每卡 token (bs)", str(wl.tokens)),
        ("topk", str(wl.topk)),
        ("每卡专家", f"{local}（共 {world * local}）"),
        ("AIC 核", str(scenario.aic_num)),
        ("h", str(scenario.h)),
        ("intermediate", str(scenario.hidden_dim)),
        ("编排", orch),
        ("p1 / p2", f"{scenario.p1_override} / {scenario.p2_override}"),
    ]
    if wl.shared_expert_num:
        rows.append(("共享专家", str(wl.shared_expert_num)))
    same = len(set(recv)) == 1
    return {
        "title": f"{world} 卡 · 每卡 {wl.tokens} token · topk {wl.topk} · "
                 + {"uniform": "均匀路由"}.get(wl.routing, wl.routing + " 路由"),
        "lede": "下面的时刻都是 DAG 调度器排出来的事件起止时间。执行时间记到最后一个 COMBINE "
                "结束，尾段照常调度但不计入。"
                + (f"{world} 张卡的负载相同，图里画的是其中一张。" if same
                   else f"各卡负载不同，图里画的是第 {rank} 张。"),
        "rows": rows,
        "routeNote": "每格是行数。"
                     + (f"{world} 张目的卡的矩阵完全相同，这里显示其中一张。" if same
                        else f"这里显示第 {rank} 张目的卡。"),
        "facts": [
            ("每张源卡发出", f"{wl.tokens} × {wl.topk} = {one(sorted(set(sent)))} 行"),
            ("均分到", f"{world} × {local} = {world * local} 个专家"),
            ("每格 C[目的卡][专家][源卡]", f"{one(cells)} 行"),
            ("每个专家共收", f"{one(per_expert)} 行"),
            ("每张卡共收", f"{one(sorted(set(recv)))} 行"),
        ],
    }


def collect(scenario, rates, rank: int = 0) -> dict:
    """事件 + 依赖边 + 每档速率的调度时刻."""
    built, _ = A8W8WaveCostModel(
        scenario.with_overrides({"calibration.cube_mac_per_us": rates[0]}).build_costs(),
        scenario.resolved_options()).build_events(_shape_of(scenario, rank))
    by_name = {e.name: e for e in built}
    nodes = [e for e in built if str(e.meta.get("stage")) not in SKIP_STAGES]
    index = {e.name: i for i, e in enumerate(nodes)}

    def sources(dep: str):
        """跳过零时长的就绪标记, 把边接到真正产出数据的事件上."""
        node = by_name[dep]
        if str(node.meta.get("stage")) in SKIP_STAGES:
            return [s for x in node.deps for s in sources(x)]
        return [dep] if dep in index else []

    edges = []
    for e in nodes:
        seen = set()
        for dep in e.deps:
            for s in sources(dep):
                if s not in seen:
                    seen.add(s)
                    edges.append([index[s], index[e.name]])

    out = {
        "events": [[_strip_rank(e.name), str(e.meta.get("stage")),
                    e.resources[0] if e.resources else "",
                    int(e.meta.get("expert", -1) if e.meta.get("expert") is not None else -1),
                    int(e.meta.get("ntile", -1) if e.meta.get("ntile") is not None else -1),
                    str(e.meta.get("part") or ""), int(e.meta.get("m_rows") or 0)] for e in nodes],
        "edges": edges,
        "routing": scenario.workload.routing_counts()[rank],
        "spec": _spec(scenario, scenario.workload.routing_counts(), rank),
        "rates": [],
    }
    for rate in rates:
        res = simulate(scenario.with_overrides({"calibration.cube_mac_per_us": rate}))
        r = res["rank_results"][rank]
        sched = {e.name: e for e in r["events"]}
        missing = set(index) - set(sched)
        if missing:
            raise ValueError(f"事件图与调度结果不一致, 缺 {sorted(missing)[:4]}")
        crit = {c["name"] for c in r["critical_path"]}
        out["rates"].append({
            "rate": rate,
            "total": round(r["total_us"], 3),
            "dag_end": round(r["dag_end_us"], 3),
            "waves": r["wave_count"],
            "n_events": len(r["events"]),
            "stages": {s: [round(r["stage_first_start_us"][s], 3),
                           round(r["stage_last_end_us"][s], 3),
                           sum(1 for e in r["events"] if e.meta.get("stage") == s),
                           round(min(e.end_us - e.start_us for e in r["events"]
                                     if e.meta.get("stage") == s), 3),
                           round(max(e.end_us - e.start_us for e in r["events"]
                                     if e.meta.get("stage") == s), 3)]
                       for s in STAGE_KEYS if s in r["stage_first_start_us"]},
            "times": [[round(sched[e.name].start_us, 3), round(sched[e.name].end_us, 3),
                       1 if e.name in crit else 0,
                       sched[e.name].critical_reason.split(":")[0],
                       round(sched[e.name].dependency_ready_us, 3)] for e in nodes],
            "critical": [index[c["name"]] for c in r["critical_path"] if c["name"] in index],
        })
    return out


def render(data: dict, out_path: Path) -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("/*DATA*/") != 1:
        raise ValueError(f"{TEMPLATE} 必须恰好有一处 /*DATA*/ 占位")
    body = template.replace("/*DATA*/", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    # <title> 是浏览器标签名, 跟着场景走
    body = re.sub(r"<title>.*?</title>", "<title>" + data["spec"]["title"] + "</title>",
                  body, count=1, flags=re.S)
    head, rest = body.split("</title>", 1)
    page = ('<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            + head + "</title>\n"
            '<style>body{margin:0}[hidden]{display:none!important}</style>\n'
            + rest.replace("</style>", "</style>\n</head>\n<body>", 1)
            + "\n</body>\n</html>\n")
    out_path.write_text(page, encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenario", type=Path, help="场景文件 (.toml/.json)")
    ap.add_argument("-o", "--out", type=Path, help="输出 HTML (默认与场景同名 _report.html)")
    ap.add_argument("--rates", default="1e7,2.7e7,5e7", help="逗号分隔的 Cube 速率 (MAC/µs)")
    ap.add_argument("--rank", type=int, default=0, help="画哪张卡 (默认 0)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    scenario = load_scenario(args.scenario)
    rates = [float(x) for x in args.rates.split(",")]
    out = args.out or args.scenario.with_name(args.scenario.stem + "_report.html")
    data = collect(scenario, rates, args.rank)
    render(data, out)
    for rec in data["rates"]:
        print(f"  R_cube={rec['rate']:.1e}  执行时间 {rec['total']:8.3f} us  "
              f"含尾段 {rec['dag_end']:8.3f} us")
    print(f"written {out}  ({out.stat().st_size // 1024} KB, {len(data['events'])} 事件, "
          f"{len(data['edges'])} 条依赖边)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
