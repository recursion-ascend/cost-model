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


#: skewed_routing() 守恒所需的每 rank token 数: 每源 rank 发出 632 行 = 79 x topk 8。
SK_TOKENS = 79


def skewed_routing():
    """4 rank × 64 专家, 4 个非空专家 (每目的卡 300/64/12/256 行). 与 test_api_smoke 锚点同源.

    **守恒** (每源 rank 发出 SK_TOKENS x 8 = 632 行)。2026-10-05 之前这里是
    counts[dst][63] = [200, 20, 20, 16] 对全部目的卡相同, 于是源 rank 0 发出 1176 行、
    其余 440-460 行 —— 而 64 token x top-8 最多 512 行: 一组**物理上不可能**的输入,
    golden 一直在给它建图。现在专家 63 的重源按目的卡**轮转** (目的卡 d 的重源是
    src d), 专家 2 由 13 行改 12 行 (凑 8 的倍数): 保留"一个重专家、每卡一个重源"的
    偏斜, 而每个源 rank 发出的总行数相同。
    """
    world, local = 4, 64
    counts = [[[0] * world for _ in range(local)] for _ in range(world)]
    heavy = [200, 20, 20, 16]
    for dst in range(world):
        counts[dst][0] = [75, 75, 75, 75]
        counts[dst][1] = [16, 16, 16, 16]
        counts[dst][2] = [3, 3, 3, 3]
        counts[dst][63] = heavy[-dst:] + heavy[:-dst] if dst else list(heavy)
    return counts


def uniform_routing(world: int, local: int, per_src: int):
    return [[[per_src] * world for _ in range(local)] for _ in range(world)]


def manual_costs(cube_rate=CUBE_RATE, kernel=None):
    """手工拼 PrimitiveCosts (证明不经 build_analytical_costs 也能跑).

    **combine 的公式必须按 shape 声明的那个 kernel 建**: run_api 传的是
    P.shape_kw() (profile 的 KernelConfig(combine_meta_bytes_per_row=32)), 而这里原先
    裸构造 AnalyticalCombineCosts() —— meta 用缺省 16。于是同一个 case 里"形状说每行
    搬 32B、公式按 16B 算", 两条入口 (场景 vs API) 对同一个形状给出不同的 combine 字节
    与时长。2026-10-05 扩充 golden 指纹 (加 traffic_bytes) 才把它照出来 —— 原指纹只锁
    时长, 而夹具内部是自洽的 (时长也按 16 算), 所以这个不一致藏了很久。
    """
    km = kernel if kernel is not None else P.kernel
    comb = m.AnalyticalCombineCosts(
        meta_bytes_per_row=km.combine_meta_bytes_per_row)
    return m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=m.AnalyticalGmmCosts(cube_mac_per_us=cube_rate).gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts(cube_mac_per_us=cube_rate).gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=comb.tile,
        combine_write_bytes_per_row=comb.write_bytes_per_row,
        combine_read_bytes=comb.read_bytes,
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
    # 出处报告走和 api 同一个函数 (config.provenance.run_provenance) —— 这个低层入口
    # 原先不给出处, 于是 test_scenario_matches_golden 比的两份指纹天生不等。
    return {"provenance": m.run_provenance(costs, kernel),
            "kernel_total_us": float(ranks[slowest]["total_us"]),
            "kernel_dag_end_us": max(float(r["dag_end_us"]) for r in ranks.values()),
            "slowest_rank": slowest, "rank_results": ranks}


def _pipeline(**kw):
    return P.with_options(pipeline=m.PipelineConstraints(**kw))


# ---------------------------------------------------------------------------
# 用例表
# ---------------------------------------------------------------------------

def _cases() -> Dict[str, Callable[[], Dict[str, object]]]:
    sk = skewed_routing
    # 3 波: 6 专家 × 512 行。每源 rank 发出 2 卡 x 6 专家 x 256 = 3072 行 = 384 x top-8。
    # 2026-10-05 之前配的是 512 token (应发 4096 行) —— 不守恒, 现改为守恒的 384。
    w3 = lambda: uniform_routing(2, 6, 256)
    W3_TOKENS = 384
    steal = lambda: uniform_routing(2, 4, 256)       # 4 专家 × 512 行, 转移生效的最小规模
    off = m.StageWaveOffsets
    pol = P.with_policy          # 以 profile 的策略为底, 只改指定项
    kc = P.with_kernel
    split = dict(queues=m.QueueDepths(mte_aic=2, cube=2, fix=2))

    c: Dict[str, Callable[[], Dict[str, object]]] = {}

    # ---- MTE 路径: 默认与波偏移 ----
    c["mte_skewed_default"] = lambda: run_api(sk(), SK_TOKENS)
    c["mte_skewed_lag1"] = lambda: run_api(sk(), SK_TOKENS, policy=pol(gmm2_lag_waves=1))
    c["mte_uniform_bs64"] = lambda: run_api(
        uniform_routing(4, 64, 2), 64, topk=8, kernel=kc(),
        costs=analytical_costs(kc()), policy=pol())
    c["mte_3wave_lag0"] = lambda: run_api(w3(), W3_TOKENS, policy=pol(gmm2_lag_waves=0))
    c["mte_3wave_lag2"] = lambda: run_api(w3(), W3_TOKENS, policy=pol(gmm2_lag_waves=2))
    c["mte_3wave_lookahead3"] = lambda: run_api(w3(), W3_TOKENS, policy=pol(dispatch_lookahead=3))
    c["mte_3wave_offsets_2_m2"] = lambda: run_api(
        w3(), W3_TOKENS, policy=pol(wave_offsets=off(dispatch=2, gmm2=-2)))
    c["mte_3wave_p1_4"] = lambda: run_api(w3(), W3_TOKENS, p1=4)
    c["mte_lag_by_threshold"] = lambda: run_api(uniform_routing(2, 4, 4096), 4096)
    c["mte_shared_expert"] = lambda: run_api(sk(), SK_TOKENS, shared_expert_num=1)
    c["mte_aic16"] = lambda: run_api(w3(), W3_TOKENS, aic_num=16)

    # ---- MTE 路径: 流控与编译期旋钮 ----
    c["mte_act_depth2_credit2"] = lambda: run_api(
        w3(), W3_TOKENS, policy=pol(gmm2_combine_credit=2),
        options=P.with_options(links=(
            m.StageLink("gmm1", "activation", location="onchip", depth=2,
                        colocated_by_hardware=True),
            m.StageLink("activation", "gmm2", readiness=2))))
    c["mte_tile_m128"] = lambda: run_api(sk(), SK_TOKENS, kernel=kc(tile_m=128))
    c["mte_tile_n128"] = lambda: run_api(sk(), SK_TOKENS, kernel=kc(tile_n=128))
    c["mte_l1buf1_quant1"] = lambda: run_api(
        sk(), SK_TOKENS, kernel=kc(l1_buf_num=1, combine_quant_mode=1),
        costs=analytical_costs(kc(l1_buf_num=1, combine_quant_mode=1)))
    c["mte_l1_tile_k512"] = lambda: run_api(sk(), SK_TOKENS, kernel=kc(l1_tile_k=512))
    # TopkWeightsPrefetch: epilogue 行块 256->128 + GMM1 输出走 GM 往返 + 每行块一次
    # topk 权重读 (见 README「一个轴怎么才算建模了」)。costs 必须同编译点构造 ——
    # 手工 PrimitiveCosts 不描述读回, builder 会直接报错。
    c["mte_topk_prefetch"] = lambda: run_api(
        w3(), W3_TOKENS, kernel=kc(topk_weights_prefetch=True),
        costs=analytical_costs(kc(topk_weights_prefetch=True)))
    # B 复用比例 0.53: 由 bs128 (1 组) 与 bs8192 (12 组) 反解的那一个实测点
    c["mte_b_reuse"] = lambda: run_api(w3(), W3_TOKENS, kernel=kc(gmm1_b_reuse_frac=0.53))
    c["mte_gmm2_kl1_256"] = lambda: run_api(sk(), SK_TOKENS, options=P.with_options(gmm2_kl1=256))

    # ---- 策略旋钮 (经 MegaMoeShape) ----
    c["policy_priority_by_stage"] = lambda: run_shapes(
        w3(), W3_TOKENS, scheduling_policy=m.PriorityByStage())
    c["policy_critical_path_first"] = lambda: run_shapes(
        w3(), W3_TOKENS, scheduling_policy=m.CriticalPathFirst())
    c["core_greedy_least_busy"] = lambda: run_shapes(
        sk(), SK_TOKENS, core_assignment=m.GreedyLeastBusy())
    c["core_contiguous_block"] = lambda: run_shapes(
        sk(), SK_TOKENS, core_assignment=m.ContiguousBlock())
    c["packing_balanced"] = lambda: run_shapes(sk(), SK_TOKENS, wave_packing=m.BalancedWaves())
    c["packing_longest_first"] = lambda: run_shapes(
        sk(), SK_TOKENS, wave_packing=m.LongestExpertFirst())

    # ---- Layered 路径 ----
    c["layered_skewed"] = lambda: run_api(sk(), SK_TOKENS, kernel=kc(topo_urma=True))
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
    c["pipeline_neutral"] = lambda: run_api(sk(), SK_TOKENS, options=_pipeline())
    c["pipeline_queue_depth2"] = lambda: run_api(
        sk(), SK_TOKENS, options=_pipeline(queues=m.QueueDepths(mte_aic=2)))
    c["pipeline_sync_latency"] = lambda: run_api(
        sk(), SK_TOKENS, options=_pipeline(sync=m.SyncLatency(
            gmm1_act_handshake_us=0.5, act_gmm2_ready_us=0.3, gmm2_combine_ack_us=0.2)))
    c["pipeline_split"] = lambda: run_api(sk(), SK_TOKENS, options=_pipeline(**split))
    c["pipeline_compute_bound"] = lambda: run_api(
        sk(), SK_TOKENS, costs=manual_costs(cube_rate=6.75e6),
        options=_pipeline(queues=m.QueueDepths(mte_aic=2)))
    # 2026-10-05 审计: 这个 case 原先给 phases=PhaseRates(fix_bw_bytes_per_us=2.0e5),
    # 但那个参数**没有任何读者** (fix 相位时长恒为 0, 口径是"结果写出/数据释放事件忽略
    # 不计"), 所以它一直与 pipeline_split 逐位相同 —— 一个**什么都没测到**的 case,
    # 还占着"已覆盖 fix 相位"的名分。那个参数现在给了值会直接报错, 这里把它去掉。
    # case 保留: fix 相位的事件结构 (占 QUEUE:fix 与 FIXPIPE 单元) 仍被它覆盖, 而且
    # 哪天 fix 口径改成计时长, 差异会在这里显形。
    c["pipeline_fix_phase"] = lambda: run_api(
        sk(), SK_TOKENS, options=_pipeline(queues=m.QueueDepths(mte_aic=2)))
    # 2026-10-05 审计: 这里原有 pipeline_engine_queue2 = pipeline_split +
    # engine_queue_depths(aic/vec0/aiv1=2)。两者指纹**逐位相同** —— 每核引擎队列在
    # 持核事件独占该核时恒不起约束, 相位拆分又刻意不继承 Q:*, 所以那个旋钮任何取值
    # 都无后果。旋钮已删 (model.py 把容量写死 1), case 一并去掉: 留着也只是第二份
    # pipeline_split。
    # 1024 token x top-8 = 每源 rank 8192 行 = 4 卡 x 64 专家 x 32。2026-10-05 之前
    # 路由给的是 x 8 (2048 行) 配 1024 token —— 不守恒。p1=0 由 token 数自动取档,
    # 所以保 token 数、改路由 (真实 bs1024 每专家正是 4 x 32 = 128 行)。
    c["pipeline_large_split"] = lambda: run_api(
        uniform_routing(4, 64, 32), 1024, p1=0, p2=0, options=_pipeline(**split))
    c["pipeline_layered_split"] = lambda: run_api(
        sk(), SK_TOKENS, kernel=kc(topo_urma=True), options=_pipeline(**split))
    c["serialize_dispatch_comm"] = lambda: run_api(
        sk(), SK_TOKENS, options=P.with_options(serialize_dispatch_comm=True))

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


def _provenance_digest(prov: Dict[str, object]) -> str:
    """全部常数的 (名字, 取值, 完整出处文本) 的 sha256.

    entries 是 名字 -> (取值, 出处) 的全集; 逐名排序后入摘要, 所以任何一处数值或标签
    文本的改动都会改哈希。只锁类别计数抓不到同类别内的文本改动 (实测踩过)。
    """
    entries = prov.get("entries") or {}
    d = hashlib.sha256()
    for name in sorted(entries):
        val = entries[name]
        d.update(f"{name}|{val!r}\n".encode("utf-8"))
    return d.hexdigest()


def fingerprint(result: Dict[str, object]) -> Dict[str, object]:
    ranks = result["rank_results"]
    per_rank: List[Dict[str, object]] = []
    for r in sorted(ranks):
        rr = ranks[r]
        digest = hashlib.sha256()
        for e in rr["events"]:
            digest.update(_event_line(e).encode("utf-8"))
            digest.update(b"\n")
        bounds = rr.get("bounds") or {}
        per_rank.append({
            "rank": r,
            "total_us": rr["total_us"],
            "dag_end_us": rr["dag_end_us"],
            "events": len(rr["events"]),
            "wave_count": rr["wave_count"],
            "stage_busy_us": {k: rr["stage_busy_us"][k] for k in sorted(rr["stage_busy_us"])},
            "critical_path_len": len(rr["critical_path"]),
            # 2026-10-05 加入: 访存量与下界。
            # 为什么: 原指纹只锁**时长**, 不含字节申报, 也不含下界 —— 于是 2026-10-05
            # 连着两次拿 "40 个 golden 零 diff" 当提交依据, 却都漏掉了同样两条失败
            # (test_onchip_declares_no_act_gm_write 查的是 traffic_bytes,
            #  test_provenance_report 查的是出处标签)。字节口径的改动必须在 30 秒的
            # golden 里显形, 而不是等 18 分钟的全套。
            "traffic_bytes": {k: rr["traffic_bytes"][k]
                              for k in sorted(rr.get("traffic_bytes") or {})},
            "bounds": {k: bounds.get(k) for k in
                       ("compute_us", "bandwidth_us", "dependency_us", "lower_us",
                        "binding", "total_mac", "gm_to_l1_bytes", "violation")},
            "schedule_sha256": digest.hexdigest(),
        })
    return {
        "kernel_total_us": result["kernel_total_us"],
        "slowest_rank": result["slowest_rank"],
        # 出处也锁。两层:
        #   summary  每个类别的常数个数 —— 改分类 (spec/algo/impl/measured/...) 会变。
        #   sha256   **全部常数的 (名字, 取值, 完整出处文本)** 的哈希 —— 改标签文本也会变。
        # 为什么要第二层: 2026-10-05 我改 BW_SCATTER 的出处时丢了"域受限"三个字,
        # 被 test_provenance_report 抓住, 而当时的 golden 零 diff —— 只锁类别计数抓不到
        # 同类别内的文本改动。标签文本是承诺 (域限制、待重标), 和数值一样该被保护。
        "provenance_summary": {k: v for k, v in
                              sorted((result.get("provenance") or {})
                                     .get("summary", {}).items())},
        "provenance_sha256": _provenance_digest(result.get("provenance") or {}),
        "ranks": per_rank,
    }
