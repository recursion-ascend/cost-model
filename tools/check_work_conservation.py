#!/usr/bin/env python3
"""核空闲分解: 区分"DAG 逼出来的"与"有活却空着", 并列出每一处违规.

为什么要这个工具
----------------
"核不能有空闲"作为绝对约束在逻辑上不可能: 事件必须等前置完成, 而 t=0 时 GMM1 还在等
dispatch_ready (AIV1 产出), 所以开头所有 AIC 必须空着。能成立的不变量是 **work-conserving**
—— 核不得在"存在已就绪的活"时空闲。本工具把空闲拆成两类并给出违规清单:

  forced     此刻该池里没有已就绪未开始的事件 -> 消不掉, 只能改 DAG 结构
  avoidable  有就绪的活却有核空着 -> work-conservation 违规, 换绑定方式可回收 (上界)

用法
----
  python tools/check_work_conservation.py examples/scenario_basic.toml
  python tools/check_work_conservation.py --sweep          # 扫几个形状做对比
  python tools/check_work_conservation.py <scenario> --role AIC: --segments 10
  python tools/check_work_conservation.py <scenario> --assert-conserving   # CI 用, 违规则非零退出
"""
import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import moe_cost_model as m  # noqa: E402
from moe_cost_model.analysis import idle_decomposition  # noqa: E402

ROLES = ("AIC:", "AIV0:", "AIV1:")


def report(tag, rank_result, roles, n_seg):
    rows = []
    worst = 0.0
    for role in roles:
        for rank_tag, rep in sorted(idle_decomposition(rank_result["events"], role).items()):
            rows.append((f"{rank_tag}.{role.rstrip(':')}", rep))
            worst = max(worst, rep.avoidable_idle_us)
    print(f"\n=== {tag} ===")
    print(f"  dag_end={rank_result['dag_end_us']:.1f}us  total={rank_result['total_us']:.1f}us")
    print(f"  {'池':<14}{'核数':>5}{'忙%':>8}{'busy':>11}{'forced':>11}{'avoidable':>11}  WC")
    for name, rep in rows:
        print(f"  {name:<14}{len(rep.pool):>5}{100 * rep.utilization:7.1f}%"
              f"{rep.busy_us:11.1f}{rep.forced_idle_us:11.1f}{rep.avoidable_idle_us:11.1f}"
              f"  {'OK' if rep.work_conserving else 'VIOLATION'}")
        if not rep.work_conserving and n_seg:
            for seg in rep.segments[:n_seg]:
                waiting = ", ".join(seg.waiting_ready) or "(未取样)"
                print(f"      [{seg.t_begin:8.2f},{seg.t_end:8.2f}) "
                      f"{len(seg.idle_resources):>2} 核空闲 ({seg.core_us:7.2f} 核·us) "
                      f"等着的就绪事件: {waiting}")
    return worst


def sweep(n_seg):
    """扫几个形状: 3/6 专家 x 三个 hidden_dim, 全远端拉取, aic=28."""
    worst = 0.0
    for hd in (9216, 14336, 18432):
        for local in (3, 6):
            W, PER = 5, 64
            rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)]
                  for d in range(W)]
            tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
            res = m.simulate_routing_counts(
                routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hd,
                aic_num=28, costs=m.build_analytical_costs(
                    h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
                p1_override=1, p2_override=1, topk=6)
            worst = max(worst, report(f"hidden_dim={hd} 专家={local} tok={tok}",
                                      res["rank_results"][0], ("AIC:",), n_seg))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", help="场景文件 (.toml/.json)")
    ap.add_argument("--sweep", action="store_true", help="扫内置的几个形状, 不读场景文件")
    ap.add_argument("--role", action="append", choices=list(ROLES),
                    help="只看某个角色池; 可重复。缺省三个都看")
    ap.add_argument("--segments", type=int, default=5, help="每个池打印几段 avoidable 明细")
    ap.add_argument("--rank", type=int, default=None, help="只看某个 rank (缺省最慢的那个)")
    ap.add_argument("--assert-conserving", action="store_true",
                    help="存在 avoidable 空闲时以非零状态退出 (CI 用)")
    args = ap.parse_args()

    roles = tuple(args.role) if args.role else ROLES
    if args.sweep:
        if args.scenario:
            ap.error("--sweep 不能与场景文件同用")
        worst = sweep(args.segments)
    else:
        if not args.scenario:
            ap.error("给一个场景文件, 或用 --sweep")
        res = m.simulate(m.load_scenario(args.scenario))
        rank = args.rank if args.rank is not None else res["slowest_rank"]
        worst = report(f"{args.scenario} rank{rank}", res["rank_results"][rank],
                       roles, args.segments)

    print(f"\n最大 avoidable 空闲: {worst:.1f} 核·us")
    if args.assert_conserving and worst > 1e-9:
        print("work-conservation 不成立: 有就绪的活时仍有核空闲。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
