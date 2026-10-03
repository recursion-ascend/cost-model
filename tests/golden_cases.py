"""golden 用例定义 + 调度指纹.

用途: 引擎/建图重构时确认调度结果逐位不变. 指纹覆盖每个事件的
起止时刻、等待归因与关键父事件 (sha256), 任何一位浮点差异都会改变摘要.

快照文件: tests/golden/schedule_fingerprints.json
重新生成: python tools/gen_golden.py  (仅在有意改变模型行为时执行)
"""
from __future__ import annotations

import hashlib
from typing import Callable, Dict, List

import moe_cost_model as m
from moe_cost_model.api import _rebind_costs_to_kernel
from moe_cost_model.model import A8W8WaveCostModel
from moe_cost_model.shape import MegaMoeShape

H, HIDDEN, AIC = 6144, 4096, 28

# golden 固定的是"复现那份实现"的那组取值, 所以显式引用 profile 而不是靠缺省值。
# 模型缺省是"最少假设", 跟这份实现不是一个点 —— 两者不同是信息, 不是回归。
P = m.MEGAMOE_A8W8


# ---------------------------------------------------------------------------
# 路由与公式
# ---------------------------------------------------------------------------

# 测试夹具值, 非标定常数: Cube 速率没有缺省, 测试统一取此值
CUBE_RATE = 2.7e7


def skewed_routing():
    """4 rank × 64 专家, 4 个非空专家 (300/64/13/256 行). 与 test_api_smoke 锚点同源."""
    world, local = 4, 64
    counts = [[[0] * world for _ in range(local)] for _ in range(world)]
    for dst in range(world):
        counts[dst][0] = [75, 75, 75, 75]
        counts[dst][1] = [16, 16, 16, 16]
        counts[dst][2] = [3, 4, 3, 3]
        counts[dst][63] = [200, 20, 20, 16]
    return counts


def uniform_routing(world: int, local: int, per_src: int):
    return [[[per_src] * world for _ in range(local)] for _ in range(world)]


def manual_costs(cube_rate=CUBE_RATE):
    return m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=m.AnalyticalGmmCosts(cube_mac_per_us=cube_rate).gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts(cube_mac_per_us=cube_rate).gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        count_table_prepare_us=m.T_COUNT_GATE,
    )


def analytical_costs(kernel=None):
    return m.build_analytical_costs(
        h=H, kernel=kernel, dispatch_mechanistic=m.DispatchMechanisticLatency(),
        cube_mac_per_us=CUBE_RATE)


# ---------------------------------------------------------------------------
# 运行入口
# ---------------------------------------------------------------------------

def run_api(routing, token_num, *, costs=None, aic_num=AIC, p1=2, p2=1, **kw):
    kw = dict(P.shape_kw(options=P.options), **kw)
    return m.simulate_routing_counts(
        routing_counts=routing, token_num_per_rank=token_num, h=H, hidden_dim=HIDDEN,
        aic_num=aic_num, costs=costs if costs is not None else manual_costs(),
        p1_override=p1, p2_override=p2, **kw)


def run_shapes(routing, token_num, *, costs=None, aic_num=AIC, p1=2, p2=1,
               kernel=None, policy=None, options=None, restructure=None,
               **shape_kw) -> Dict[str, object]:
    """经 MegaMoeShape 直达 model: 覆盖 api 入口未暴露的策略旋钮
    (scheduling_policy / core_assignment / wave_packing)."""
    kernel = kernel if kernel is not None else P.kernel
    costs = _rebind_costs_to_kernel(costs if costs is not None else manual_costs(), kernel)
    model = A8W8WaveCostModel(costs, options or P.options)
    shapes = []
    for dst, rows in enumerate(routing):
        src_rows = tuple(tuple(int(x) for x in r) for r in rows)
        shapes.append(MegaMoeShape(
            expert_tokens=tuple(sum(r) for r in src_rows), token_num=token_num,
            h=H, hidden_dim=HIDDEN, aic_num=aic_num, rank_id=dst,
            p1_override=p1, p2_override=p2, expert_source_tokens=src_rows,
            dispatch_layout=m.DispatchDataLayout.from_hidden(H),
            **dict(P.shape_kw(kernel=kernel,
                              policy=policy if policy is not None else P.policy),
                   **shape_kw)))
    ranks = model.simulate_multi(shapes, restructure=restructure)
    slowest = max(ranks, key=lambda r: float(ranks[r]["total_us"]))
    return {"kernel_total_us": float(ranks[slowest]["total_us"]),
            "kernel_dag_end_us": max(float(r["dag_end_us"]) for r in ranks.values()),
            "slowest_rank": slowest, "rank_results": ranks}


def _pipeline(**kw):
    return P.with_options(pipeline=m.PipelineConstraints(**kw))


# ---------------------------------------------------------------------------
# 用例表
# ---------------------------------------------------------------------------

def _cases() -> Dict[str, Callable[[], Dict[str, object]]]:
    sk = skewed_routing
    w3 = lambda: uniform_routing(2, 6, 256)          # 3 波: 6 专家 × 512 行
    steal = lambda: uniform_routing(2, 4, 256)       # 4 专家 × 512 行, 转移生效的最小规模
    off = m.StageWaveOffsets
    pol = P.with_policy          # 以 profile 的策略为底, 只改指定项
    kc = P.with_kernel
    split = dict(queues=m.QueueDepths(mte_aic=2, cube=2, fix=2))

    c: Dict[str, Callable[[], Dict[str, object]]] = {}

    # ---- MTE 路径: 默认与波偏移 ----
    c["mte_skewed_default"] = lambda: run_api(sk(), 64)
    c["mte_skewed_lag1"] = lambda: run_api(sk(), 64, policy=pol(gmm2_lag_waves=1))
    c["mte_uniform_bs64"] = lambda: run_api(
        uniform_routing(4, 64, 2), 64, topk=8, kernel=kc(),
        costs=analytical_costs(kc()), policy=pol())
    c["mte_3wave_lag0"] = lambda: run_api(w3(), 512, policy=pol(gmm2_lag_waves=0))
    c["mte_3wave_lag2"] = lambda: run_api(w3(), 512, policy=pol(gmm2_lag_waves=2))
    c["mte_3wave_lookahead3"] = lambda: run_api(w3(), 512, policy=pol(dispatch_lookahead=3))
    c["mte_3wave_offsets_2_m2"] = lambda: run_api(
        w3(), 512, policy=pol(wave_offsets=off(dispatch=2, gmm2=-2)))
    c["mte_3wave_p1_4"] = lambda: run_api(w3(), 512, p1=4)
    c["mte_lag_by_threshold"] = lambda: run_api(uniform_routing(2, 4, 4096), 4096)
    c["mte_shared_expert"] = lambda: run_api(sk(), 64, shared_expert_num=1)
    c["mte_aic16"] = lambda: run_api(w3(), 512, aic_num=16)

    # ---- MTE 路径: 流控与编译期旋钮 ----
    c["mte_act_depth2_credit2"] = lambda: run_api(
        w3(), 512, policy=pol(gmm2_combine_credit=2),
        options=P.with_options(links=(
            m.StageLink("gmm1", "activation", location="onchip", depth=2,
                        colocated_by_hardware=True),
            m.StageLink("activation", "gmm2", readiness=2))))
    c["mte_tile_m128"] = lambda: run_api(sk(), 64, kernel=kc(tile_m=128))
    c["mte_tile_n128"] = lambda: run_api(sk(), 64, kernel=kc(tile_n=128))
    c["mte_l1buf1_quant1"] = lambda: run_api(
        sk(), 64, kernel=kc(l1_buf_num=1, combine_quant_mode=1),
        costs=analytical_costs(kc(l1_buf_num=1, combine_quant_mode=1)))
    c["mte_l1_tile_k512"] = lambda: run_api(sk(), 64, kernel=kc(l1_tile_k=512))
    c["mte_b_reuse"] = lambda: run_api(w3(), 512, kernel=kc(gmm1_b_reuse=True))
    c["mte_gmm2_kl1_256"] = lambda: run_api(sk(), 64, options=P.with_options(gmm2_kl1=256))

    # ---- 策略旋钮 (经 MegaMoeShape) ----
    c["policy_priority_by_stage"] = lambda: run_shapes(
        w3(), 512, scheduling_policy=m.PriorityByStage())
    c["policy_critical_path_first"] = lambda: run_shapes(
        w3(), 512, scheduling_policy=m.CriticalPathFirst())
    c["core_greedy_least_busy"] = lambda: run_shapes(
        sk(), 64, core_assignment=m.GreedyLeastBusy())
    c["core_contiguous_block"] = lambda: run_shapes(
        sk(), 64, core_assignment=m.ContiguousBlock())
    c["packing_balanced"] = lambda: run_shapes(sk(), 64, wave_packing=m.BalancedWaves())
    c["packing_longest_first"] = lambda: run_shapes(
        sk(), 64, wave_packing=m.LongestExpertFirst())

    # ---- Layered 路径 ----
    c["layered_skewed"] = lambda: run_api(sk(), 64, kernel=kc(topo_urma=True))
    c["layered_2wave"] = lambda: run_api(
        uniform_routing(2, 16, 64), 256, kernel=kc(topo_urma=True),
        costs=analytical_costs(kc(topo_urma=True)))
    c["layered_uniform_bs64"] = lambda: run_api(
        uniform_routing(4, 64, 2), 64, topk=8, kernel=kc(topo_urma=True),
        costs=analytical_costs(), p1=0, p2=0)
    c["layered_credit1"] = lambda: run_api(
        uniform_routing(2, 16, 64), 256, kernel=kc(topo_urma=True),
        policy=pol(gmm2_combine_credit=1))

    # ---- 相位流水 / 容量 / 信道 ----
    c["pipeline_neutral"] = lambda: run_api(sk(), 64, options=_pipeline())
    c["pipeline_queue_depth2"] = lambda: run_api(
        sk(), 64, options=_pipeline(queues=m.QueueDepths(mte_aic=2)))
    c["pipeline_sync_latency"] = lambda: run_api(
        sk(), 64, options=_pipeline(sync=m.SyncLatency(
            gmm1_act_handshake_us=0.5, act_gmm2_ready_us=0.3, gmm2_combine_ack_us=0.2)))
    c["pipeline_split"] = lambda: run_api(sk(), 64, options=_pipeline(**split))
    c["pipeline_compute_bound"] = lambda: run_api(
        sk(), 64, costs=manual_costs(cube_rate=6.75e6),
        options=_pipeline(queues=m.QueueDepths(mte_aic=2)))
    c["pipeline_fix_phase"] = lambda: run_api(
        sk(), 64, options=_pipeline(queues=m.QueueDepths(mte_aic=2),
                                    phases=m.PhaseRates(fix_bw_bytes_per_us=2.0e5)))
    c["pipeline_engine_queue2"] = lambda: run_api(
        sk(), 64, options=P.with_options(
            pipeline=m.PipelineConstraints(**split),
            engine_queue_depths=m.EngineQueueDepths(aic=2, vec0=2, aiv1=2)))
    c["pipeline_large_split"] = lambda: run_api(
        uniform_routing(4, 64, 8), 1024, p1=0, p2=0, options=_pipeline(**split))
    c["pipeline_layered_split"] = lambda: run_api(
        sk(), 64, kernel=kc(topo_urma=True), options=_pipeline(**split))
    c["serialize_dispatch_comm"] = lambda: run_api(
        sk(), 64, options=P.with_options(serialize_dispatch_comm=True))

    # ---- 运行时图重构 (钩子每次提交扫描全部未提交事件, 规模取小) ----
    c["stealing_gmm1"] = lambda: run_api(steal(), 256, restructure=m.idle_core_stealing())
    c["stealing_pipeline"] = lambda: run_api(
        steal(), 256, restructure=m.idle_core_stealing(),
        options=_pipeline(queues=m.QueueDepths(mte_aic=2)))
    return c


CASES = _cases()


# ---------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------

def _event_line(e) -> str:
    return "|".join((
        e.name, repr(e.start_us), repr(e.end_us),
        repr(e.dependency_ready_us), repr(e.resource_ready_us),
        repr(e.dependency_wait_us), repr(e.resource_queue_us),
        str(e.critical_parent), e.critical_reason,
        repr(e.capacity_wait_us),
        ",".join(e.resources), str(e.order),
        str(e.meta.get("stolen_from", "")),
    ))


def fingerprint(result: Dict[str, object]) -> Dict[str, object]:
    ranks = result["rank_results"]
    per_rank: List[Dict[str, object]] = []
    for r in sorted(ranks):
        rr = ranks[r]
        digest = hashlib.sha256()
        for e in rr["events"]:
            digest.update(_event_line(e).encode("utf-8"))
            digest.update(b"\n")
        per_rank.append({
            "rank": r,
            "total_us": rr["total_us"],
            "dag_end_us": rr["dag_end_us"],
            "events": len(rr["events"]),
            "wave_count": rr["wave_count"],
            "stage_busy_us": {k: rr["stage_busy_us"][k] for k in sorted(rr["stage_busy_us"])},
            "critical_path_len": len(rr["critical_path"]),
            "schedule_sha256": digest.hexdigest(),
        })
    return {
        "kernel_total_us": result["kernel_total_us"],
        "slowest_rank": result["slowest_rank"],
        "ranks": per_rank,
    }
