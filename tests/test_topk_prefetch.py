"""TopkWeightsPrefetch: 建模而不是拒绝.

kernel 里这个编译期模板参数有三个确定的后果 (都不是时长系数):

  1. `EPILOGUE_TILE_M = TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256`
     (op_kernel/arch35/mega_moe_arch35.h:161) —— epilogue 的行块高度减半, 一个
     GMM1 tile 的 epilogue 变成两个前后相继的行块;
  2. GMM1 的输出改走 GM: AIC 落 GM 并置 `gmm1TileStatus`, AIV 等这个 GM 标志再
     `CopyGM2UB` 读回 (stage/mega_moe_gmm1_activation.h:618-645, 405-460), 而不是
     Fixpipe L0C->UB 直给配对 AIV0 —— 于是 gmm1->activation 从片上变成 GM 往返,
     UB ping-pong 槽位约束不再适用;
  3. 每个行块多一次 topk 权重的 GM->UB 读 (m x META_INFO_SIZE x int32)。

本文件逐条钉住这三点, 并钉住"不开 prefetch 时一切不变"。
"""
import pytest

import moe_cost_model as m
from moe_cost_model.config.hardware import (EPILOGUE_TILE_M_PREFETCH,
                                            GMM1_OUT_ELEM_BYTES,
                                            META_BYTES_PER_ROW, KernelConfig,
                                            epilogue_tile_m)
from moe_cost_model.config.links import effective_gmm1_act_link
from moe_cost_model.implementations.compile import CompileConfig
from moe_cost_model.implementations.megamoe import A8W8WaveV1
from moe_cost_model.model import A8W8WaveCostModel
from moe_cost_model.shape import MegaMoeShape, ModelOptions

CUBE_RATE = 2.7e7


def _costs(kernel):
    return m.build_analytical_costs(
        h=6144, dispatch_mechanistic=m.DispatchMechanisticLatency(),
        kernel=kernel, cube_mac_per_us=CUBE_RATE)


def _shape(kernel):
    world, local, per = 2, 4, 512
    # 守恒: 每个源 rank 发出 local x per 行 = token_num x topk (topk = 8)
    return MegaMoeShape(
        expert_tokens=tuple(per * world for _ in range(local)),
        token_num=local * per // 8,
        h=6144, hidden_dim=4096, aic_num=8, rank_id=0,
        p1_override=2, p2_override=1,
        expert_source_tokens=tuple(tuple(per for _ in range(world))
                                   for _ in range(local)),
        kernel=kernel)


def _acts(kernel, options=None):
    shape = _shape(kernel)
    model = A8W8WaveCostModel(_costs(kernel), options or ModelOptions())
    events, _ = model.build_events(shape)
    return [ev for ev in events if ev.meta.get("stage") == "activation"]


# ---- 1. 几何 ----

def test_epilogue_tile_m_follows_the_kernels_ternary():
    """tile_m = 256 时与 kernel 的两个字面常量逐值相同."""
    assert epilogue_tile_m(256, False) == 256
    assert epilogue_tile_m(256, True) == EPILOGUE_TILE_M_PREFETCH == 128
    assert KernelConfig().epilogue_tile_m == 256
    assert KernelConfig(topk_weights_prefetch=True).epilogue_tile_m == 128


def test_epilogue_block_never_exceeds_the_gmm1_block():
    """tile_m < 128 的编译点上行块不会比 GMM1 的块还高 (模型外推, 非 kernel 事实)."""
    assert epilogue_tile_m(64, True) == 64


# ---- 2. 编译轴只有一个出处 ----

def test_the_axis_lives_on_the_compile_point_not_on_the_orchestration():
    """ModelOptions 上不再有这个字段: 它是 MEGAMOE_TOPK_PREFETCH, 不是编排选择."""
    assert not hasattr(ModelOptions(), "topk_weights_prefetch")
    assert CompileConfig.from_kernel_config(
        KernelConfig(topk_weights_prefetch=True)).topk_weights_prefetch is True
    assert CompileConfig.from_kernel_config(
        KernelConfig()).topk_weights_prefetch is False


def test_the_axis_moves_the_compile_fingerprint():
    a = CompileConfig.from_kernel_config(KernelConfig())
    b = CompileConfig.from_kernel_config(KernelConfig(topk_weights_prefetch=True))
    assert a.fingerprint != b.fingerprint


def test_it_is_no_longer_refused():
    """原先这里抛 Unsupported / NotImplementedError."""
    A8W8WaveV1().accepts(CompileConfig(topk_weights_prefetch=True), ModelOptions())
    km = KernelConfig(topk_weights_prefetch=True)
    A8W8WaveCostModel(_costs(km), ModelOptions()).build_events(_shape(km))


# ---- 3. 那条边 ----

def test_prefetch_turns_the_gmm1_act_edge_into_a_gm_round_trip():
    base = effective_gmm1_act_link((), KernelConfig())
    pf = effective_gmm1_act_link((), KernelConfig(topk_weights_prefetch=True))
    assert (base.location, base.depth, base.colocated_by_hardware) == ("onchip", 1, True)
    assert (pf.location, pf.depth, pf.colocated_by_hardware) == ("gm", 0, False)


def test_the_ub_slot_constraint_disappears_under_prefetch():
    """槽位是 Fixpipe 交接的容量; GM 往返下没有这个交接, 留着等于凭空多一条约束."""
    km = KernelConfig(topk_weights_prefetch=True)
    shape = _shape(km)
    model = A8W8WaveCostModel(_costs(km), ModelOptions())
    events, _ = model.build_events(shape)
    slots = [q for ev in events for q, _ in ev.acquires if "UB:gmm1act" in q]
    assert slots == []
    plain = KernelConfig()
    events2, _ = A8W8WaveCostModel(_costs(plain), ModelOptions()).build_events(_shape(plain))
    assert [q for ev in events2 for q, _ in ev.acquires if "UB:gmm1act" in q]


# ---- 4. 行块拆分 ----

def test_prefetch_halves_the_epilogue_row_block_and_doubles_the_act_events():
    plain = _acts(KernelConfig())
    pf = _acts(KernelConfig(topk_weights_prefetch=True))
    assert len(pf) == 2 * len(plain)
    assert {ev.meta["m_rows"] for ev in plain} == {256}
    assert {ev.meta["m_rows"] for ev in pf} == {128}
    # 行范围无缝覆盖: 每个 256 行组被切成 [0,128) 与 [128,256)
    assert {(ev.meta["row_begin"] % 256, ev.meta["row_end"] % 256 or 256)
            for ev in pf} == {(0, 128), (128, 256)}


def test_the_row_blocks_keep_the_same_dependency_key():
    """GMM2 还是按 (专家, m-group) 取依赖 —— kernel 的 flag 下标仍是 subMLoc/256."""
    km = KernelConfig(topk_weights_prefetch=True)
    model = A8W8WaveCostModel(_costs(km), ModelOptions())
    events, _ = model.build_events(_shape(km))
    by_name = {ev.name: ev for ev in events}
    acts = {ev.name for ev in events if ev.meta.get("stage") == "activation"}
    g2 = [ev for ev in events if ev.meta.get("stage") == "gmm2"]
    assert g2
    for ev in g2:
        mine = [d for d in ev.deps if d in acts]
        # 两个行块都要到齐
        assert len(mine) >= 2, ev.name
        assert {by_name[d].meta["mgroup"] for d in mine} == {ev.meta["mgroup"]}


def test_no_split_when_the_row_range_already_fits_the_block():
    """行数 <= 行块高度时不拆, 名字也不带 .e 后缀 (非 prefetch 路径逐字不变)."""
    assert all(".e" not in ev.name for ev in _acts(KernelConfig()))


# ---- 5. 读回的字节与时长 ----

def test_the_readback_bytes_are_declared_on_their_own_channel():
    km = KernelConfig(topk_weights_prefetch=True)
    acts = _acts(km)
    one = acts[0]
    chans = {name: b for name, b, _ in one.channel_bytes}
    assert "act_readback" in chans
    rows, cols = one.meta["m_rows"], one.meta["logical_n"]
    expect = (rows * cols * km.activation_n_half * GMM1_OUT_ELEM_BYTES
              + rows * META_BYTES_PER_ROW)
    assert chans["act_readback"] == expect == one.meta["readback_bytes"]


def test_the_gmm1_side_declares_the_matching_write():
    """同一股字节两边对称: GMM1 写 GM, ACT 读回来."""
    km = KernelConfig(topk_weights_prefetch=True)
    model = A8W8WaveCostModel(_costs(km), ModelOptions())
    events, _ = model.build_events(_shape(km))
    wrote = sum(b for ev in events if ev.meta.get("stage") == "gmm1"
                for name, b, _ in ev.channel_bytes if name == "hbm_write")
    read = sum(ev.meta["readback_bytes"] - ev.meta["m_rows"] * META_BYTES_PER_ROW
               for ev in events if ev.meta.get("stage") == "activation")
    assert wrote == pytest.approx(read)


def test_the_readback_time_is_serial_with_the_vector_work():
    """kernel 用 MTE2_V 标志把搬运与计算隔开, 所以是相加而不是取大."""
    km = KernelConfig(topk_weights_prefetch=True)
    c = _costs(km)
    act = _acts(km)[0]
    rows, cols = act.meta["m_rows"], act.meta["logical_n"]
    assert act.duration_us == pytest.approx(
        c.activation_tile(rows, cols) + c.activation_ready_publish_us
        + c.activation_readback_us(rows, cols))
    assert c.activation_readback_us(rows, cols) > 0


def test_a_costs_object_without_the_prefetch_path_fails_loudly():
    """手工 PrimitiveCosts + prefetch: 报错, 不按"读回免费"悄悄算完."""
    km = KernelConfig(topk_weights_prefetch=True)
    bare = m.PrimitiveCosts(
        dispatch_mechanistic=m.DispatchMechanisticLatency(),
        gmm1_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm1_tile,
        gmm2_tile=m.AnalyticalGmmCosts(cube_mac_per_us=CUBE_RATE).gmm2_tile,
        activation_tile=m.AnalyticalActCosts().tile,
        activation_store_bytes=m.AnalyticalActCosts().store_bytes,
        combine_tile=m.AnalyticalCombineCosts().tile,
        combine_write_bytes_per_row=m.AnalyticalCombineCosts().write_bytes_per_row,
        combine_read_bytes=m.AnalyticalCombineCosts().read_bytes,
    )
    with pytest.raises(ValueError, match="prefetch"):
        A8W8WaveCostModel(bare, ModelOptions()).build_events(_shape(km))


# ---- 6. 它必须真的改变评估结果 ----

def test_turning_it_on_changes_the_estimate():
    """多搬一遍 GMM1 的输出不可能免费 —— 否则这个轴就是个无法生效的旋钮."""
    plain, pf = KernelConfig(), KernelConfig(topk_weights_prefetch=True)
    a = A8W8WaveCostModel(_costs(plain), ModelOptions()).simulate(_shape(plain))
    b = A8W8WaveCostModel(_costs(pf), ModelOptions()).simulate(_shape(pf))
    assert b["total_us"] != a["total_us"]
