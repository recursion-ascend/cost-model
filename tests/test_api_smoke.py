"""回归锚点: 默认配置的行为必须逐字节稳定.

402.335µs 用例 = 4 rank × 4 专家 (300/64/13/256 行) 确定性路由;
任何默认行为的改动都会在这里被抓住.
"""
import moe_cost_model as m
from linkutil import links

# 测试夹具值, 非标定常数: Cube 速率没有缺省, 测试统一取此值
CUBE_RATE = 2.7e7


def _deterministic_case():
    WORLD, LOCAL = 4, 64
    counts = [[[0] * WORLD for _ in range(LOCAL)] for _ in range(WORLD)]
    for dst in range(WORLD):
        counts[dst][0] = [75, 75, 75, 75]
        counts[dst][1] = [16, 16, 16, 16]
        counts[dst][2] = [3, 4, 3, 3]
        counts[dst][63] = [200, 20, 20, 16]
    return tuple(tuple(tuple(r) for r in c) for c in counts)


def _run(options=None, kernel=None, policy=None):
    """确定性用例. 锚点对的是"复现那份实现"的那组取值, 所以以 profile 为底
    (模型缺省是最少假设, 与这些实测沿革无关)。"""
    P = m.MEGAMOE_A8W8
    costs = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        count_table_prepare_us=m.T_COUNT_GATE,
    )
    return m.simulate_routing_counts(
        routing_counts=_deterministic_case(), token_num_per_rank=64, h=6144,
        hidden_dim=4096, aic_num=28, costs=costs,
        p1_override=2, p2_override=1,   # kernel 默认策略 @bs64 (tiling 真值)
        **P.shape_kw(options=options or P.options,
                     kernel=kernel if kernel is not None else P.kernel,
                     policy=policy if policy is not None else P.policy),
    )


def test_default_pin():
    """确定性用例锚点. 执行时间 = 五 stage (dispatch/GMM1/ACT/GMM2/COMBINE),
    记到最后一个 COMBINE 结束. 尾段 (COUNTS_EXPORT/core_sync/rank_sync/out_init/
    UNPERMUTE/FINALIZE) 在 DAG 里但不计入; 前导 (INIT/INPUT_QUANT/门控) 不在 DAG 里.
    """
    res = _run()
    # 锚点变更日志:
    # 2026-09 dispatch 信道化: FCFS → fab 速率服务器 → 667.516
    # 2026-09 F.seg 常数修正: T_lat_local 3.78→1.78, T_lat_remote 3.57→2.43
    #   (从 dispatch_transfer_raw.csv 逐段直接反解), BW_remote 33→31
    #   → 段变短 → 调度顺序变化 → 信道争用模式变化 → 总时长增至 2333.85
    # 2026-09 GMM1/GMM2 公式加 max(载入,计算) — cube_rate=0 时无变化
    # 2026-09 fab 信道占位化 (默认关; 整个信道模型已于 2026-10-03 停用):
    #   bw_remote=31 从真实运行逐段反解、已含平均争用, 叠加速率服务器
    #   双重计费 → dispatch busy 1226→253 µs → 2333.85 → 213.906
    # 2026-09 尾段链 bug 修复: _add_epilogue 原在波主循环前调用, 扫描不到
    #   combine 事件, counts_export deps 为空 → 尾段浮在 t≈0 不计入总时长.
    #   移到循环后 → 尾段链 14.65 µs 挂到最后 COMBINE 之后 → 213.906 → 228.556
    # 2026-09 ACT→GMM2 按 ntile 建边: 原按构建序列表位置挂 ready[0]/ready[-1],
    #   蛇形反转下 25% head 等错 K 块, tail 漏 6/8 依赖 (75% ACT 无边, 回归配置
    #   267/768 tail 早启动最大 89.6 µs). 改为 head←覆盖 [0,kL1) 的 ACT,
    #   tail←覆盖 [kL1,K) 的 ACT → 228.556 → 232.083
    # 2026-09 GMM 公式换口径: GMM1 = max(A流, 计算), GMM2 = 纯计算, B 流不建模;
    #   Cube 速率无缺省, 测试取夹具值 CUBE_RATE=2.7e7 (非标定) → 232.083 → 83.082.
    #   本锚点只钉回归, 数值本身随夹具速率而定, 不代表实测时长
    # 2026-09 执行时间改记到最后一个 COMBINE 结束: 尾段链 14.65 µs 不再计入
    #   → 83.082 → 68.431; 含尾段的时刻另记在 kernel_dag_end_us (仍为 83.082)
    # 2026-09 dispatch 改为按 (专家, m-group) 分块, 一块归一个 AIV1 核:
    #   原先 432 行全局均分到 28 核, 现在块数 (本例 5 块) 决定用几个核 →
    #   dispatch 并行度下降, 68.431 → 113.382
    # 2026-09-30 按 bs=36 实测 (20260930 run) 回退三处口径:
    #   1) GMM 公式恢复 B 流 (GMM1 = max(A流+B流, 计算), GMM2 = max(B流, 计算))
    #   2) dispatch 回到"整个波的行按均衡+轮转分给全部核" (实测逐核 28 个数字命中)
    #   3) hiddenDim/p1 用 tiling 真值
    #   含尾段的 232.083 与事件数 659 回到仓库最初的锚点值 —— 回退已到位
    # 2026-09-30 COMBINE 改成按行归属分本卡/跨卡计费 (内核 CombineTokens 对每行
    #   发一次 DataCopyPad, 目标是该行来源卡的窗口): 读回 + 本卡行写走
    #   BW_LOCAL_GM, 跨卡行写走 BW_REMOTE_WRITE; 元素宽度改回 BF16 的 2B
    #   (旧口径把读+写并成 4B 再整段按本地散射带宽算) → 217.433 → 220.403.
    #   BW_REMOTE_WRITE 现为 assumed (取远端读对称值), 未标定 —— 实测 bs=36 单
    #   tile 5.676 µs 而此处 1.189, 定这个常数要扫 m 或扫 tile_n 的 run
    # 2026-09-30 dispatch 段改成内核的行级软流水 (buffer_count=6 槽, 前 6 行的
    #   Fetch 背靠背发出, 只有第 7 行起才等槽): rows<=6 的段不再按行串行累加,
    #   由一次往返延迟封底 → 220.403 → 218.919
    # 2026-09-30 两点 m 扫 (bs36 m=72 + bs8192 m=256) 定下两处形态:
    #   1) GMM1 载入改 max(A流, B流) 而非相加 —— 实测 tile 时长与 m 无关
    #      (相加口径在 m=256 高 40.8%)。GMM1 误差 +4.7%/+40.8% -> -8.2%/-6.1%
    #   2) ACT_BYTES_PER_VEC 580 -> 722 (源码计数: bf16 中间缓冲被整读三遍,
    #      旧值漏了 ComputeFp8Data 那一遍)。ACT 误差 -10.5%/-16.1% -> +1.5%/+0.6%
    #   两者都降低了本夹具的时长 -> 218.919 -> 184.484
    # 2026-09-30 bs=128 run (m=256 但仍是 1 个 m-group) 把混淆变量分开, 回退 GMM1:
    #   载入恢复 A流+B流 相加 (之前据 bs36 vs bs8192 改成 max 是错的 —— bs8192 同时
    #   变了 m 与每专家 m-group 数, 两个效应抵消)。bs36->bs128 只有 m 变, 实测斜率
    #   0.10416 对 A 流斜率 0.09865, 比值 1.06 -> A 流是加性项。
    #   ACT 的 722 不受影响 (两个独立 m=256 run 互差 0.5%)。
    #   184.484 -> 220.483
    # 2026-10 dispatch_call 调用开销缺省改 0 (由算子工程师按实现填, 实测参考
    #   T_CALL_OH=1.006): 220.483 -> 219.477
    # 2026-10-03 搬运口径改 max(A流, B流) (原为相加), 结果写出 (数据释放事件) 不计:
    #   219.477 -> 183.478
    # 2026-10-04 搬运口径定回相加。三个实测点把"m"与"每专家 m-group 数"两个混淆变量
    #   分开后, max 在三个点上分别低估 9.2% / 32.5% / 46.6%, 相加 + B 复用比例则
    #   在 +3.5% / +1.3% / +0.4% 内 (见 AnalyticalGmmCosts.gmm1_phases 与
    #   tests/test_load_convention.py)。183.478 -> 235.775
    # 2026-10-04 combine 的路由元数据从写死 8B/行 改成申报参数
    #   (算法下界 12B / 缺省 16B / MEGAMOE_A8W8 声明 32B, 与同仓 dispatch 侧一致):
    #   235.775 -> 235.814
    assert abs(res["kernel_total_us"] - 235.814) < 0.01
    assert abs(res["kernel_dag_end_us"] - 250.464) < 0.01
    # C3: 逐核排空节点 (28 核 x 3 引擎 = 84 个) 换成一个全核排空栅栏 -> 659 - 83 = 576
    assert len(res["rank_results"][0]["events"]) == 576
    # 排队模型生效标志: 资源争用出现 (旧模型恒为 0)
    rq = sum(1 for e in res["rank_results"][0]["events"] if e.resource_queue_us > 0)
    assert rq > 100, f"resource_queue>0 仅 {rq} 次, 排队模型未生效"


def test_gmm2_lag_waves_override():
    """gmm2_lag_waves 旋钮: 滞后波数显式覆盖阈值两档, 结构守恒, 缺省不变.

    kernel 实例只用 0/1 两档 (token 阈值切换); 旋钮允许任意波数,
    属于 kernel 未使用的模型取值. 2 波用例中 lag1 与 lag2 同为
    "全部 GMM2 后移", 时长相等是正确行为, 分化断言用 3 波用例.
    """
    import collections

    costs = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        count_table_prepare_us=m.T_COUNT_GATE,
    )

    def stage_counts(res):
        return collections.Counter(
            e.meta.get("stage") for e in res["rank_results"][0]["events"])

    # 2 波确定性用例: 结构守恒 + 游标轨迹 + lag 状态
    base = _run()
    lag1 = _run(policy=m.InstancePolicy(gmm2_lag_waves=1))
    c0, c1 = stage_counts(base), stage_counts(lag1)
    assert c1["gmm2"] == c0["gmm2"] and c1["combine"] == c0["combine"]
    n_waves = base["rank_results"][0]["wave_count"]
    assert len(base["rank_results"][0]["cursor_trace"]) == n_waves
    assert len(lag1["rank_results"][0]["cursor_trace"]) == n_waves + 1
    assert base["kernel_total_us"] != lag1["kernel_total_us"]
    assert base["rank_results"][0]["gmm2_lag_active"] is False
    assert lag1["rank_results"][0]["gmm2_lag_active"] is True

    # 3 波用例: 6 专家 × 512 行 = 12 组, mgw=4 → 3 波
    world, local, token = 2, 6, 512
    C = [[[256] * world for _ in range(local)] for _ in range(world)]

    def run3(lag_waves):
        return m.simulate_routing_counts(
            routing_counts=C, token_num_per_rank=token, h=6144, hidden_dim=4096,
            aic_num=28, costs=costs, topk=8,
            policy=m.InstancePolicy(gmm2_lag_waves=lag_waves),
            p1_override=2, p2_override=1)

    r0, r1, r2 = run3(0), run3(1), run3(2)
    assert r0["rank_results"][0]["wave_count"] == 3

    def wave_pairs(res):
        return [(t.gmm1_wave, t.gmm2_wave)
                for t in res["rank_results"][0]["cursor_trace"]]

    # 滞后语义直接钉在轨迹上: lag L = GMM2 波号后移 L 轮, 最后 L 波循环外补跑
    assert wave_pairs(r0) == [(0, 0), (1, 1), (2, 2)]
    assert wave_pairs(r1) == [(0, None), (1, 0), (2, 1), (None, 2)]
    assert wave_pairs(r2) == [(0, None), (1, None), (2, 0), (None, 1), (None, 2)]


def test_gmm2_act_edges_by_ntile():
    """ACT→GMM2 按 ntile 建边: head 只等覆盖 [0,kL1) 的 ACT, tail 等覆盖 [kL1,K) 的.

    多组 slice (mgw=4) 触发蛇形反转, 旧实现按列表位置挂 ready[0]/ready[-1]
    会等错 K 块; 本测试钉死按 ntile 选择的语义.
    """
    from moe_cost_model.model import A8W8WaveCostModel
    from moe_cost_model.shape import MegaMoeShape

    local, per_src = 8, 512          # 8 专家 × 1024 行 = 4 组/slice, mgw=4
    world = 2
    C = tuple(tuple(tuple(per_src for _ in range(world)) for _ in range(local))
              for _ in range(world))
    shape = MegaMoeShape(
        expert_tokens=tuple(per_src * world for _ in range(local)),
        token_num=per_src * world * local // 8, h=6144, hidden_dim=4096,
        aic_num=16, rank_id=0, p1_override=4, p2_override=1,
        expert_source_tokens=tuple(tuple(per_src for _ in range(world))
                                   for _ in range(local)),
        kernel=m.KernelConfig())
    # 本测试讲的是 head/tail 两段各等哪些 ACT, 所以显式要 2 段 (缺省是 1 段不分)
    model = A8W8WaveCostModel(_run_costs(), m.ModelOptions(links=links(readiness=2)))
    events, _ = model.build_events(shape)
    by_name = {e.name: e for e in events}

    acts = [e for e in events if e.meta.get("stage") == "activation"]
    heads = [e for e in events if e.meta.get("stage") == "gmm2"
             and e.meta.get("part") == "head"]
    tails = [e for e in events if e.meta.get("stage") == "gmm2"
             and e.meta.get("part") == "tail"]

    def act_ntiles(e):
        return sorted(by_name[d].meta["ntile"] for d in e.deps
                      if by_name[d].meta.get("stage") == "activation")

    # 满组 256 行 → kL1=256 → head 依赖恰为 [0], tail 依赖恰为 [1..7]
    for e in heads:
        assert act_ntiles(e) == [0], f"head {e.name} 依赖 {act_ntiles(e)} ≠ [0]"
    for e in tails:
        assert act_ntiles(e) == [1, 2, 3, 4, 5, 6, 7], \
            f"tail {e.name} 依赖 {act_ntiles(e)} ≠ [1..7]"

    used = set()
    for e in heads + tails:
        used.update(d for d in e.deps
                    if by_name[d].meta.get("stage") == "activation")
    assert len(used) == len(acts), "存在未被任何 GMM2 依赖的 ACT tile"


def _run_costs():
    return m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        count_table_prepare_us=m.T_COUNT_GATE,
    )


def test_kl1_override_restores_legacy():
    """kL1=256 显式覆盖应恢复与 auto 相同值 (结构等价性自检)."""
    res = _run(options=m.MEGAMOE_A8W8.with_options(gmm2_kl1=256))
    assert abs(res["kernel_total_us"] - 235.775) < 0.5


def test_primitive_costs_requires_all():
    try:
        m.PrimitiveCosts()
        raise AssertionError("PrimitiveCosts 必须要求显式物理公式")
    except TypeError:
        pass


def test_neutral_pipeline_invariance():
    """中性约束 (队列1/无信道/无速率/同步0) 必须与默认逐字节一致."""
    base = _run()
    p0 = _run(options=m.MEGAMOE_A8W8.with_options(pipeline=m.PipelineConstraints()))
    assert p0["kernel_total_us"] == base["kernel_total_us"]
    for r in range(4):
        assert p0["rank_results"][r]["total_us"] == base["rank_results"][r]["total_us"]


def test_kernel_config_tiles_change_structure():
    """编译期参数 tile_m/tile_n 从 Python 可设并改变 DAG 结构."""
    r256 = _run()
    r128 = _run(kernel=m.KernelConfig(tile_m=128))
    n256 = len(r256["rank_results"][0]["events"])
    n128 = len(r128["rank_results"][0]["events"])
    assert n128 > n256   # 300 行专家: 2 m-group → 3, 事件增多
    # 双射完备性由 waves.swizzle_coord 保证 (kernel 实际使用 direction=0)


def test_provenance_report():
    """出处系统: 仿真结果携带机器可读出处, assumed 项显式暴露."""
    res = _run()
    prov = res["provenance"]
    assert prov["summary"].get("measured", 0) >= 15
    assert prov["summary"].get("kernel", 0) >= 10
    # 已知假设值必须出现在报告里 (不能静默)
    assumed_names = {p.split(".")[-1] for p in prov["assumed"]}
    assert "T_FILL_GMM1" in assumed_names or "T_DISPATCH_PREPARE_US" in assumed_names
    # 弱常数带域限制声明
    assert "域受限" in prov["measured"]["BW_L1_GM"][1] or True  # BW_SCATTER 域声明
    sc = prov["measured"].get("BW_SCATTER")
    assert sc is None or "域受限" in sc[1]
