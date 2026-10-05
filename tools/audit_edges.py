"""审计 DAG 依赖边: 核对边指控 + 程序序补边影响实验.

用例 A: aic=16 p1=3, 17 个单组专家 (mgw=3) — ACT 孤儿比例
用例 B: aic=16 p1=4, 8 个 4 组专家 (mgw=4) — 蛇形反转触发, head 挂错 K 块
"""
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from artifacts import cube_rate
from moe_cost_model.model import A8W8WaveCostModel
from moe_cost_model.profiles import MEGAMOE_A8W8 as PROFILE
from moe_cost_model.shape import MegaMoeShape
from moe_cost_model.config.hardware import T_COUNT_GATE
from moe_cost_model.scheduler import MultiResourceScheduler
from moe_cost_model.costs import (
    PrimitiveCosts, DispatchMechanisticLatency, AnalyticalGmmCosts,
    AnalyticalActCosts, AnalyticalCombineCosts)


def make_costs():
    return PrimitiveCosts(
        dispatch_mechanistic=DispatchMechanisticLatency(),
        gmm1_tile=AnalyticalGmmCosts(cube_mac_per_us=cube_rate()).gmm1_tile,
        gmm2_tile=AnalyticalGmmCosts(cube_mac_per_us=cube_rate()).gmm2_tile,
        activation_tile=AnalyticalActCosts().tile,
        activation_store_bytes=AnalyticalActCosts().store_bytes,
        combine_tile=AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=AnalyticalCombineCosts().write_bytes_per_row,
        combine_read_bytes=AnalyticalCombineCosts().read_bytes)


def make_shape(aic, p1, local, per_expert_rows, world=2, h=6144, hidden=4096):
    per_src = per_expert_rows // world
    token = per_src * world * local // 8
    C = [[[per_src] * world for _ in range(local)] for _ in range(world)]
    expert_tokens = tuple(per_src * world for _ in range(local))
    src_tokens = tuple(tuple(per_src for _ in range(world)) for _ in range(local))
    return MegaMoeShape(
        expert_tokens=expert_tokens, token_num=token, h=h, hidden_dim=hidden,
        aic_num=aic, rank_id=0, p1_override=p1, p2_override=1,
        expert_source_tokens=src_tokens, **PROFILE.shape_kw())


def strip_fab(events):
    for ev in events:
        ev.channel_bytes = tuple(t for t in ev.channel_bytes
                                 if not t[0].startswith("fab_"))


def schedule(events, aic):
    caps = {}
    for core in range(aic):
        # 每核引擎队列容量恒 1 (持核事件独占该核, 更深的队列无可表达的后果)
        caps[f"Q:aic:c{core}"] = 1
        caps[f"Q:vec0:c{core}"] = 1
        caps[f"Q:aiv1:c{core}"] = 1
    total, sched = MultiResourceScheduler().schedule(events, capacities=caps)
    return total, sched


def audit(tag, aic, p1, local, per_expert_rows, chain=True):
    shape = make_shape(aic, p1, local, per_expert_rows)
    model = A8W8WaveCostModel(make_costs(), PROFILE.options)
    events, trace = model.build_events(shape)
    strip_fab(events)
    by_name = {e.name: e for e in events}

    acts = [e for e in events if e.meta.get("stage") == "activation"]
    heads = [e for e in events if e.meta.get("stage") == "gmm2"
             and e.meta.get("part") == "head"]
    tails = [e for e in events if e.meta.get("stage") == "gmm2"
             and e.meta.get("part") == "tail"]
    combines = [e for e in events if e.meta.get("stage") == "combine"]
    n_groups = len(set((a.meta["expert"], a.meta["mgroup"]) for a in acts))
    print(f"== {tag}: aic={aic} p1={p1}, 组={n_groups}, ACT={len(acts)}, "
          f"head={len(heads)}, combine={len(combines)}, "
          f"mgw={model.m_groups_per_wave(shape)} ==")

    used = set()
    for e in events:
        if e.meta.get("stage") == "gmm2":
            for d in e.deps:
                dd = by_name.get(d)
                if dd is not None and dd.meta.get("stage") == "activation":
                    used.add(d)
    n_orphan = len(acts) - len(used)
    print(f"  [1] 无 GMM2 依赖边的 ACT: {n_orphan}/{len(acts)} "
          f"({100*n_orphan/len(acts):.0f}%)")

    wrong = sum(
        1 for e in heads
        if [by_name[d] for d in e.deps
            if by_name.get(d) is not None
            and by_name[d].meta.get("stage") == "activation"]
        and all(by_name[d].meta.get("ntile") != 0
                for d in e.deps if by_name.get(d) is not None
                and by_name[d].meta.get("stage") == "activation"))
    print(f"  [1] head 依赖不含 ntile0 (挂错 K 块): {wrong}/{len(heads)}")

    base_total, sched = schedule(events, aic)
    sby = {e.name: e for e in sched}
    print(f"  基线总时长 = {base_total:.1f} µs")

    s_acts = {}
    for e in sched:
        if e.meta.get("stage") == "activation":
            s_acts.setdefault((e.meta["expert"], e.meta["mgroup"]), []).append(e)
    early_cnt, max_early = 0, 0.0
    for t in tails:
        key = (t.meta["expert"], t.meta["mgroup"])
        last_end = max(a.end_us for a in s_acts[key])
        gap = last_end - sby[t.name].start_us
        if gap > 1e-9:
            early_cnt += 1
            max_early = max(max_early, gap)
    print(f"  [1] tail 早于该组全部 ACT 完成: {early_cnt}/{len(tails)}, "
          f"最大提前 {max_early:.1f} µs")

    ce = sby.get("epilogue.counts_export")
    if ce:
        all_cb = max(c.end_us for c in combines
                     if c.name in sby) if combines else 0
        print(f"  [epilogue] counts_export start={ce.start_us:.1f} vs "
              f"全部 combine 最晚 end={all_cb:.1f}, 提前 {all_cb-ce.start_us:.1f}")

    for role in ("AIC", "AIV0", "AIV1"):
        engines = {}
        for e in sched:
            for r in e.resources:
                if r.startswith(role):
                    engines.setdefault(r, []).append(e)
        inv = tot = 0
        for r, evs in engines.items():
            evs.sort(key=lambda x: x.start_us)
            orders = [x.order for x in evs]
            for i in range(len(orders)):
                for j in range(i + 1, len(orders)):
                    tot += 1
                    if orders[i] > orders[j]:
                        inv += 1
        print(f"  [2] {role} 执行序逆序对: {inv}/{tot} ({100*inv/max(tot,1):.0f}%)")

    seg_before = 0
    calls = {e.name: e for e in sched if e.meta.get("stage") == "dispatch_call"}
    n_seg = 0
    for e in sched:
        if e.meta.get("stage") == "dispatch":
            n_seg += 1
            cn = f"W{e.meta['wave']}.dispatch_call.c{e.meta['core']}"
            if cn in calls and e.start_us < calls[cn].start_us - 1e-9:
                seg_before += 1
    print(f"  [2] dispatch 段早于本波 call: {seg_before}/{n_seg}")

    interleave = 0
    for t in tails:
        hn = t.name + ".h"
        if hn not in sby:
            continue
        he, te = sby[hn], sby[t.name]
        res = he.resources[0] if he.resources else None
        if not res:
            continue
        for e in sched:
            if e.name in (hn, t.name) or res not in e.resources:
                continue
            if he.end_us <= e.start_us < te.start_us:
                interleave += 1
                break
    print(f"  [2] head 与 tail 之间被插入其他事件的 tile: "
          f"{interleave}/{len(tails)}")

    if chain:
        events2 = [copy.copy(e) for e in events]
        by_res = {}
        for e in events2:
            for r in e.resources:
                by_res.setdefault(r, []).append(e)
        added = 0
        for r, evs in by_res.items():
            evs.sort(key=lambda x: x.order)
            for a, b in zip(evs, evs[1:]):
                if a.name not in b.deps:
                    b.deps = tuple(dict.fromkeys(b.deps + (a.name,)))
                    added += 1
        chain_total, _ = schedule(events2, aic)
        print(f"  [3] 补程序序边 {added} 条: 总时长 {chain_total:.1f} µs, "
              f"上升 {100*(chain_total/base_total-1):.1f}%")


if __name__ == "__main__":
    audit("用例A 单组slice", aic=16, p1=3, local=17, per_expert_rows=256)
    audit("用例B 4组slice", aic=16, p1=4, local=8, per_expert_rows=1024)
    audit("用例C 回归配置", aic=28, p1=2, local=4, per_expert_rows=2048)
