"""URMA Layered 路径 (topo_urma=True) 回归: 波规划移植 + 机制公式 + 结构断言.

锚定的源: mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h (宏 Wave 策略),
stage/mega_moe_layered_dispatch.h (二级接收), stage/mega_moe_layered_combine.h
(批量 PUT combine), fab_arbitration_probe/results/report.md (URMA GET 常数).
"""
import moe_cost_model as m

# 测试夹具值, 非标定常数: Cube 速率没有缺省, 测试统一取此值
CUBE_RATE = 2.7e7


def _costs():
    return m.build_analytical_costs(h=6144, dispatch_mechanistic=m.DispatchMechanisticLatency(),
        cube_mac_per_us=CUBE_RATE)


def _uniform(world=4, local=64, token_num=64, topk=8):
    per = token_num * topk // world // local
    return [[[per] * world for _ in range(local)] for _ in range(world)]


def _run(C, token_num, topk=8, kernel=None, aic_num=28, h=6144, hidden_dim=4096):
    return m.simulate_routing_counts(
        routing_counts=C, token_num_per_rank=token_num, h=h, hidden_dim=hidden_dim,
        aic_num=aic_num, costs=_costs(), topk=topk,
        kernel=kernel if kernel is not None else m.KernelConfig(topo_urma=True))


# ---------------------------------------------------------------------------
# 1. 宏 Wave 规划移植 (CalcTargetWaveCount / CalcFirstWaveExpertCount / CalcSteady)
# ---------------------------------------------------------------------------

def test_layered_wave_policy_port():
    # B=64 topk=8: totalRows=512 ≤ 1024 → 单 Wave
    ws = m.plan_layered_waves([8] * 64, 64, 8)
    assert [(w.begin.expert, w.end.expert) for w in ws] == [(0, 64)]
    # B=1024: est=128 ≤ 256 → latency target=2; first=ceil(1024/128)=8; steady=56
    ws = m.plan_layered_waves([128] * 64, 1024, 8)
    assert [(w.begin.expert, w.end.expert) for w in ws] == [(0, 8), (8, 64)]
    # B=4096: est=512 → balanced target=6; first=2, steady=13
    ws = m.plan_layered_waves([512] * 64, 4096, 8)
    assert len(ws) == 6
    assert (ws[0].begin.expert, ws[0].end.expert) == (0, 2)
    # B=16384: est=2048 ≥ 2048 → throughput target=4; first=1, steady=21
    ws = m.plan_layered_waves([2048] * 64, 16384, 8)
    assert len(ws) == 4
    assert (ws[0].begin.expert, ws[0].end.expert) == (0, 1)
    # 少专家阈值: experts ≤ 8 且 256 < est < 2048 且 total > 1024 → latency 2
    ws = m.plan_layered_waves([400] * 8, 400, 8)   # est=400, total=3200
    assert len(ws) == 2
    # 对照: 同 est 但 experts > 8 → balanced 6
    ws = m.plan_layered_waves([400] * 64, 3200, 8)
    assert len(ws) == 6
    # 单专家 → 单 Wave
    ws = m.plan_layered_waves([2048], 2048, 8)
    assert len(ws) == 1


def test_layered_waves_are_expert_ranges():
    ws = m.plan_layered_waves([300, 64, 13, 256], 64, 8)
    # Wave 边界只落在专家边界; 专家不被切分 (区别于 MTE m-group wave)
    for w in ws:
        for s in w.slices:
            assert s.row_begin == 0
            assert s.row_end in (300, 64, 13, 256)


# ---------------------------------------------------------------------------
# 2. URMA 机制公式 (fab_arbitration_probe 标定)
# ---------------------------------------------------------------------------

def test_urma_mechanistic_formulas():
    u = m.UrmaMechanisticLatency()
    # pair 单流锚点: 64 行 × 6336B = 405504B → λ + bytes/BW ≈ 188.4µs (probe 188.9)
    assert abs(u.get_batch_us(64 * 6336) - (8.5 + 405504 / 2253.0)) < 1e-9
    assert abs(u.get_batch_us(64 * 6336) - 188.4) < 0.5
    # flag 轮询: λ + 256×8B/BW
    assert abs(u.flag_poll_us() - (8.5 + 2048 / 2253.0)) < 1e-9
    # PUT 复用 GET 常数 (assumed 对称)
    assert u.put_batch_us(1024) == u.get_batch_us(1024)


def test_layered_layout_matches_probe_row():
    lay = m.LayeredDispatchLayout.from_hidden(6144)
    # kernel widthA=h, widthAScale=CeilDiv(h,64)×2 → 6144+192=6336B (probe rowBytes)
    assert lay.get_bytes_per_token() == 6336
    assert lay.width_a_scale == 192
    # 本地拷贝 = 读 win + 写 dispatchRev 双份
    assert lay.local_copy_bytes_per_token() == 2 * 6336


# ---------------------------------------------------------------------------
# 3. 端到端结构: stage 清单 + 通道归属 + 守恒
# ---------------------------------------------------------------------------

def test_layered_end_to_end_stage_inventory():
    res = _run(_uniform(), 64)
    r0 = res["rank_results"][0]
    stages = {e.meta.get("stage") for e in r0["events"]}
    # URMA 特有 stage 存在
    assert {"dispatch_recv", "dispatch_local", "mask_scan", "combine"} <= stages
    # MTE 特有 stage 不存在 (无 dispatch_call / 无 MTE dispatch 段)
    assert "dispatch_call" not in stages
    assert "dispatch" not in stages
    # 单宏 Wave (B=64: totalRows 512 ≤ 1024)
    assert r0["wave_count"] == 1
    assert r0["gmm2_lag_active"] is False


def test_layered_channel_ownership():
    # 通道归属 = rank % aic_num: 每个 (src, wave) 的接收事件核号必须等于 src%28
    res = _run(_uniform(), 64)
    for ev in res["rank_results"][0]["events"]:
        if ev.meta.get("stage") in ("dispatch_recv", "dispatch_local", "mask_scan"):
            assert ev.meta["core"] == ev.meta["src_rank"] % 28
        if ev.meta.get("stage") == "combine":
            assert ev.meta["core"] == ev.meta["dst_rank"] % 28


def test_layered_dispatch_ready_conservation():
    # 每 (expert, group) 的 ready 汇合全部贡献通道 (builder 内置校验, 此处验执行)
    res = _run(_uniform(world=4, local=8, token_num=128, topk=8), 128)
    ready = [e for e in res["rank_results"][0]["events"]
             if e.meta.get("stage") == "dispatch_ready"]
    assert ready
    for e in ready:
        assert e.meta["contributed_rows"] == e.meta["required_rows"]
        assert e.meta["contributor_count"] >= 1


def test_layered_program_order_recv_before_combine():
    # kernel 程序序: 同核 recv(w+1) 先于 combine(w) — AIV1 链保证
    res = _run(_uniform(world=4, local=64, token_num=1024, topk=8), 1024)
    assert res["rank_results"][0]["wave_count"] == 2
    evs = res["rank_results"][0]["events"]
    for core in range(28):
        recv1 = [e for e in evs if e.meta.get("stage") == "dispatch_recv"
                 and e.meta.get("wave") == 1 and e.meta.get("core") == core]
        comb0 = [e for e in evs if e.meta.get("stage") == "combine"
                 and e.meta.get("wave") == 0 and e.meta.get("core") == core]
        if recv1 and comb0:
            assert max(e.end_us for e in recv1) <= min(e.start_us for e in comb0) + 1e-9


# ---------------------------------------------------------------------------
# 4. 批 λ 归因: 每 (wave, dst) 的 PUT commit 数 = ceil(rows/256)
# ---------------------------------------------------------------------------

def test_layered_put_batch_lambda_attribution():
    # rank0 专家从 rank1 收 300 行: combine PUT 批数 = ceil(300/256) = 2
    # (conservation: 每 src 总路由 = token_num×topk = 2048)
    C = [[[0, 300]], [[2048 - 0, 0]]]
    # src=0 路由 2048 全给 rank1; src=1 给 rank0 300 + rank1 1748
    C[1][0] = [0, 1748]
    C[1][0][0] = 2048 - 1748  # src0 剩余给 rank1
    res = _run(C, 256, topk=8)
    r0 = res["rank_results"][0]
    put_batches = [e.meta.get("put_batches", 0) for e in r0["events"]
                   if e.meta.get("stage") == "combine" and e.meta.get("dst_rank") == 1]
    assert sum(put_batches) == 2   # 256 行满批 1 + 波尾 flush 1


def test_layered_put_batch_lambda_exact_multiple():
    # 512 行 = 2 个满批, 无尾 flush → λ 总数 = 2
    C = [[[0, 512]], [[1536, 512]]]
    # src0: 512 给 rank1 (2048-512-512=1024? conservation: src0 总 2048)
    C[1][0] = [1024, 1024]
    res = _run(C, 256, topk=8)
    r0 = res["rank_results"][0]
    put_batches = [e.meta.get("put_batches", 0) for e in r0["events"]
                   if e.meta.get("stage") == "combine" and e.meta.get("dst_rank") == 1]
    assert sum(put_batches) == 2


# ---------------------------------------------------------------------------
# 5. 默认行为冻结: topo_urma=False 不受影响 (MTE 回归锚点在 test_regression)
# ---------------------------------------------------------------------------

def test_mte_default_unchanged_with_urma_code_present():
    res = m.simulate_routing_counts(
        routing_counts=_uniform(), token_num_per_rank=64, h=6144, hidden_dim=4096,
        aic_num=28, costs=_costs(), topk=8)   # 默认 KernelConfig → MTE
    stages = {e.meta.get("stage") for e in res["rank_results"][0]["events"]}
    assert "dispatch_call" in stages
    assert "dispatch_recv" not in stages


def test_layered_empty_expert_survives():
    # 空专家保留在 Wave 范围内但不产生事件 (kernel UpdateGroupParams m=0 跳过)
    counts = [[0] * 4 for _ in range(64)]
    counts[0] = [16, 16, 16, 16]
    counts[63] = [16, 16, 16, 16]
    C = [counts] * 4
    res = _run(C, 64)
    assert res["kernel_total_us"] > 0
    stages = {e.meta.get("stage") for e in res["rank_results"][0]["events"]}
    assert "gmm1" in stages
