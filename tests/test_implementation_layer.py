"""实现层: 身份 / 编译指纹 / 适配器选择.

这一层是四层架构 (Workload -> Runtime -> Compile -> Lowering -> DAG -> Scheduler) 里
中间两层的落地。它要守住的事:

  1. 一份实现有**机器可核对的名字**, 不是自由文本。换 kernel 变体是换适配器, 不是翻布尔。
  2. 编译点有**指纹**, 因为 kernel 自己的 tiling key 只编码 5 个轴, TILE_M/TILE_N/
     L1_BUF_NUM/IsGmm1Interleaved 都在 key 之外 —— 同一个 key 可以对应多个二进制。
  3. 指纹**只盖编译轴**: 形状与拓扑每次运行都变, 混进去就不再是"同一个二进制"的标识。
  4. 标定值的适用域能被问出来 (形状域 / 拓扑 / 指纹), 这是第 8 步把全局常数换成
     keyed 标定的前提。
  5. 归位**不改行为**: 适配器包住现有建图器, golden 39 个指纹逐位不变 (另见
     tools/gen_golden.py --check --explain)。
"""
import dataclasses

import pytest

import moe_cost_model as m
from moe_cost_model.implementations import (CalibrationDomain, CompileConfig,
                                            FINGERPRINT_AXES, ImplementationId,
                                            RuntimeTopology, ShapeDomain, Unsupported,
                                            A8W8WaveV1, LayeredV1, adapter_for, resolve)


# --------------------------------------------------------------------------- 身份

def test_implementation_id_round_trips_through_its_string():
    """名字是三段式且可还原 —— 它要能当表键、文件名与报告里的列名."""
    i = ImplementationId("ascend950", "megamoe.a8w8_wave", "v1")
    assert i.key == "ascend950.megamoe.a8w8_wave.v1"
    assert ImplementationId.parse(i.key) == dataclasses.replace(i, source_refs=())


def test_implementation_id_rejects_names_that_cannot_be_keys():
    """大写/空格/斜杠都挡掉: 这个字符串要直接做键与文件名, 不该再转义一层."""
    for bad in ("Ascend950", "ascend 950", "ascend/950", ""):
        with pytest.raises(ValueError):
            ImplementationId(bad, "megamoe", "v1")
    with pytest.raises(ValueError, match="至少要三段"):
        ImplementationId.parse("ascend950.megamoe")


def test_both_repo_implementations_are_named_and_cite_their_source():
    """仓内两份实现都有身份, 且 source_refs 指向真实存在的文件.

    自由文本的"出处"是这一层要消灭的东西 (ReferenceProfile.source 就是自由文本),
    所以这里核对路径真的在仓里。
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for adapter in (A8W8WaveV1(), LayeredV1()):
        ident = adapter.identity()
        assert ident.hardware_id == "ascend950"
        assert ident.source_refs, f"{ident.key} 没有源码依据"
        missing = [r for r in ident.source_refs if not (root / r.split(":")[0]).exists()]
        assert not missing, f"{ident.key} 的 source_refs 指向不存在的文件: {missing}"


# --------------------------------------------------------------------------- 编译指纹

def test_fingerprint_is_stable_and_covers_every_declared_axis():
    """同一组取值 -> 同一个指纹; 任一编译轴改动 -> 指纹变.

    逐轴扫而不是抽查: 新增一个轴却忘了让它进指纹, 这里会红 —— 那正是"同一个 key 对应
    多个二进制"的老问题换个地方重演。
    """
    base = CompileConfig()
    assert base.fingerprint == CompileConfig().fingerprint
    for axis in FINGERPRINT_AXES:
        cur = getattr(base, axis)
        other = (not cur) if isinstance(cur, bool) else \
                (cur + 1 if isinstance(cur, int) else cur + "_x")
        moved = dataclasses.replace(base, **{axis: other})
        assert moved.fingerprint != base.fingerprint, f"{axis} 不进指纹"


def test_fingerprint_ignores_shape_and_topology_and_bookkeeping():
    """形状/拓扑/记录项不进指纹.

    为什么重要: 指纹的含义是"同一个二进制"。h 变了不该换指纹, 否则标定表会按形状分裂,
    而形状域本来由 ShapeDomain 表达。provenance 是记录从哪儿读来的, 同样不是编译轴。
    """
    a = CompileConfig()
    assert a.fingerprint == dataclasses.replace(a, provenance="manifest").fingerprint
    assert a.fingerprint == dataclasses.replace(
        a, runtime_selected=("weight_nz",)).fingerprint
    assert "h" not in FINGERPRINT_AXES and "aic_num" not in FINGERPRINT_AXES


def test_compile_config_default_matches_the_in_tree_kernel():
    """缺省编译点 = 仓内 kernel 的编译点.

    逐项对 C++: TILE_M/TILE_N (include/CMakeLists.txt:29-30),
    L1_BUF_NUM (:31), TOPK_PREFETCH 缺省 0 (include/kernel.cpp:24-26),
    MEGA_MOE_WEIGHT1_INTERLEAVED 缺省 0 (mega_moe_apt.cpp:51-53),
    BlockSchedulerSwizzle<3, 0> (common/mega_moe_gmm_common.h:33),
    ACTIVATION_N_HALF=2 (common/mega_moe_constants.h:92),
    L1_TILE_K=256 (common/mega_moe_gmm_common.h:30)。
    """
    c = CompileConfig()
    assert (c.tile_m, c.tile_n, c.l1_tile_k, c.l1_buf_num) == (256, 256, 256, 2)
    assert (c.topk_weights_prefetch, c.gmm1_interleaved) == (False, False)
    assert (c.swizzle_offset, c.swizzle_direction) == (3, 0)
    assert c.activation_n_half == 2 and c.comm_mode == "mte"
    assert "仓内缺省编译点" in c.describe()


def test_compile_config_reads_the_axes_out_of_kernel_config():
    """从 KernelConfig 取**编译轴**, 不是逐字段搬.

    KernelConfig 混着三类东西: 编译轴 (tile_m)、建模参数 (gmm1_b_reuse_frac)、
    硬件容量 (l1_size)。只有第一类属于编译点, 所以后两类不得影响指纹。
    """
    k = m.KernelConfig()
    assert CompileConfig.from_kernel_config(k).fingerprint == CompileConfig().fingerprint
    nz = CompileConfig.from_kernel_config(m.KernelConfig(tile_n=128))
    assert nz.fingerprint != CompileConfig().fingerprint and nz.tile_n == 128
    for field, value in (("gmm1_b_reuse_frac", 0.53), ("l1_size", 256 * 1024)):
        same = CompileConfig.from_kernel_config(m.KernelConfig(**{field: value}))
        assert same.fingerprint == CompileConfig().fingerprint, f"{field} 不该进指纹"


def test_runtime_selected_axes_are_declared_as_such():
    """weight_nz / combine_quant_mode 实际是运行期选的, 要说出来.

    kernel 把 IsWeightNZ 的两个特化都编进二进制, 按 groupedMatmulMode 在运行期选
    (stage/mega_moe_gmm1_activation.h:1074-1088); combineQuantMode 是 attr。模型按编译轴
    记它们, 所以必须标明"改它不需要重编", 否则会误导使用者。
    """
    assert set(CompileConfig().runtime_selected) == {"weight_nz", "combine_quant_mode"}


# --------------------------------------------------------------------------- 适配器

def test_comm_mode_picks_the_implementation():
    """选哪份实现由编译点的 comm_mode 定 (对应 kernel 的 TILINGKEY_COMM_MODE)."""
    assert adapter_for(CompileConfig()).identity().key.endswith("a8w8_wave.v1")
    assert adapter_for(CompileConfig(comm_mode="urma")).identity().key.endswith("layered.v1")


def test_old_orchestration_names_still_resolve():
    """"mte"/"layered" 这两个旧名字继续可用 —— 场景文件里写着它们."""
    assert resolve("mte").identity() == A8W8WaveV1().identity()
    assert resolve("layered").identity() == LayeredV1().identity()
    assert resolve("ascend950.megamoe.layered.v1").identity() == LayeredV1().identity()
    with pytest.raises(ValueError, match="未知实现"):
        resolve("no_such_impl")


def test_the_adapter_refuses_nothing_on_the_wave_path_any_more():
    """原先这里拒 TopkWeightsPrefetch=true, 理由是"没建模"。

    2026-10-06 它建模了 (epilogue 行块 256->128、GMM1 输出走 GM 往返、每行块多一次
    topk 权重读 —— 见 tests/test_topk_prefetch.py)。钩子留着: 适配器接受的是一个
    编译点集合, "拒绝"必须有地方说。
    """
    A8W8WaveV1().accepts(CompileConfig(topk_weights_prefetch=True), m.ModelOptions())
    A8W8WaveV1().accepts(CompileConfig(), m.ModelOptions())


def test_wave_plan_is_computed_once_per_shape():
    """波计划只算一次: 后处理要用同一份 (wave_count 进 golden 指纹).

    原先 model.waves(shape) 被调用两次 (建图 + 后处理), 两条路径都能漂。
    """
    from moe_cost_model.shape import MegaMoeShape
    costs = m.build_analytical_costs(
        h=6144, dispatch_mechanistic=m.DispatchMechanisticLatency())
    model = m.A8W8WaveCostModel(costs, m.ModelOptions())
    shape = MegaMoeShape(expert_tokens=(256, 256), token_num=64, h=6144, hidden_dim=4096,
                         aic_num=28, p1_override=2, p2_override=1,
                         expert_source_tokens=((128, 128), (128, 128)))
    first = model.wave_plan(shape)
    assert model.wave_plan(shape) is first
    assert model.waves(shape) == list(first.waves)
    assert model.m_groups_per_wave(shape) == first.m_groups_per_wave


def test_layered_plan_declares_no_m_group_wave_width():
    """Layered 的宏波没有 m-group 波宽概念, 返回 0 表示"这个量不存在".

    0 不是"未知": plan_waves 会拒绝 0 (planning/waves.py 要求为正), 所以这条路径必须
    走 plan_layered_waves, 不能回落到按波宽规划。
    """
    from moe_cost_model.shape import MegaMoeShape
    shape = MegaMoeShape(expert_tokens=(256,), token_num=64, h=6144, hidden_dim=4096,
                         aic_num=28, topk=8,
                         expert_source_tokens=((256,),))
    plan = LayeredV1().plan(shape, CompileConfig(comm_mode="urma"), m.ModelOptions())
    assert plan.m_groups_per_wave == 0 and len(plan) > 0


# --------------------------------------------------------------------------- 标定域

def test_calibration_domain_separates_out_of_range_from_undeclared():
    """越界与"这一维没声明过"必须分开报.

    为什么: 一个标定值在 h=5120 量的, 拿到 h=8192 用, 与它**从没说过** hidden_dim 的范围,
    是两种不同的不确定性。config/hardware.py 里的注释已经区分了 (例如 BW_L1_GM 写明
    "B=64 域单点, 并发未扫"), 以前没有地方存这个区分。
    """
    dom = CalibrationDomain(
        implementation=A8W8WaveV1().identity(),
        compile_fingerprint=CompileConfig().fingerprint,
        shape=ShapeDomain(token_num=(36, 8192), h=(5120, 6144)),
        topology=RuntimeTopology(world_size=4, active_cores=28))
    ok, bad = dom.shape.covers(token_num=64, h=5120)
    assert ok and bad == ()
    ok, bad = dom.shape.covers(token_num=64, h=8192)
    assert not ok and bad == ("h",)
    assert dom.shape.undeclared(hidden_dim=4096, topk=8) == ("hidden_dim", "topk")
    assert dom.topology.mismatch(world_size=2, active_cores=28) == ("world_size",)
    assert dom.key == (A8W8WaveV1().identity().key, CompileConfig().fingerprint)


def test_empty_fingerprint_is_a_gap_not_a_wildcard():
    """没记录编译点 = 一条要暴露的缺口, 不是"适用于所有编译点"."""
    dom = CalibrationDomain(implementation=A8W8WaveV1().identity())
    assert dom.compile_fingerprint == ""
    assert dom.key[1] == "", "空指纹不得被当成通配符悄悄匹配"


def test_every_result_carries_the_identity_that_produced_it():
    """结果自带身份: 实现 id + 编译指纹 + 拓扑.

    为什么必须在结果里: 一个时长数字离开 Python 之后就无从知道它对应哪份 kernel、
    哪个编译点、几张卡几个核 —— 而 config/hardware.py 的标定注释恰恰说明这些数换了
    编排或拓扑未必还成立。放在 simulate_multi 而不是 api 层, 是因为 run_shapes 这类
    入口直达前者 (bounds 与 provenance 当初就是这么漏掉四个 golden case 的)。
    """
    rc = [[[32] * 2 for _ in range(4)] for _ in range(2)]
    res = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=32, h=1024, hidden_dim=1024, aic_num=4,
        costs=m.build_analytical_costs(
            h=1024, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        topk=8, p1_override=1, p2_override=1)
    info = res["rank_results"][0]["implementation"]
    assert info["id"] == "ascend950.megamoe.a8w8_wave.v1"
    assert info["compile_fingerprint"] == CompileConfig().fingerprint
    assert info["measured_end_stage"] == "combine"
    assert info["topology"]["world_size"] == 2 and info["topology"]["active_cores"] == 4
    assert info["source_refs"], "身份必须带源码依据"


def test_layered_results_report_the_layered_implementation():
    """换 comm_mode 就换实现 id —— 结果里要看得出来换的是哪一份."""
    rc = [[[32] * 2 for _ in range(4)] for _ in range(2)]
    res = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=32, h=1024, hidden_dim=1024, aic_num=4,
        costs=m.build_analytical_costs(
            h=1024, dispatch_mechanistic=m.DispatchMechanisticLatency(),
            urma_mechanistic=m.UrmaMechanisticLatency()),
        topk=8, p1_override=1, p2_override=1, kernel=m.KernelConfig(topo_urma=True))
    info = res["rank_results"][0]["implementation"]
    assert info["id"] == "ascend950.megamoe.layered.v1"
    assert "comm_mode=urma" in info["compile_point"]


def test_reference_profile_now_has_an_identity_and_a_compile_point():
    """profile 不再只有自由文本 source: 它能说出实现 id 与编译指纹.

    MEGAMOE_A8W8 的编译点与仓内缺省只差 combine_meta_bytes_per_row=32 —— 那正是它声明的
    那份实现的取值 (META_INFO_SIZE 8 x int32, common/mega_moe_constants.h:73), 而模型缺省
    16 是"四个具名字段"的下界口径。指纹把这个差别记下来。
    """
    prof = m.MEGAMOE_A8W8
    assert prof.implementation.key == "ascend950.megamoe.a8w8_wave.v1"
    assert prof.compile_config.combine_meta_bytes_per_row == 32
    assert prof.compile_config.fingerprint != CompileConfig().fingerprint
    assert "combine_meta_bytes_per_row=32" in prof.compile_config.describe()


def test_orchestration_name_selects_the_implementation_not_just_the_builder():
    """orchestration = "layered" 要选**那份实现** (含它的宏波规划), 不只是换建图器类.

    2026-10-05 之前这个名字只换建图器, 而波计划仍按 kernel.topo_urma 分支 —— 于是
    orchestration="layered" 配 topo_urma=False 得到"Layered 建图器 + m-group 波宽"这种
    错配组合。仓内没有用例依赖它 (test_tiling 的两个用例都把两者配对), 所以改成
    名字直接选适配器, 两种拼法同值。
    """
    base = m.load_scenario(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples" / "scenario_basic.toml")
    small = {"workload.tokens": 32, "workload.world": 2, "workload.local_experts": 4,
             "aic_num": 4, "h": 1024, "hidden_dim": 1024}

    def run(**extra):
        rr = m.simulate(base.with_overrides({**small, **extra}))["rank_results"][0]
        return rr["implementation"]["id"], round(rr["total_us"], 6)

    by_kernel = run(**{"kernel.topo_urma": True})
    by_name = run(orchestration="layered")
    by_id = run(orchestration="ascend950.megamoe.layered.v1")
    assert by_kernel == by_name == by_id
    assert by_name[0] == "ascend950.megamoe.layered.v1"
    assert run(orchestration="mte")[0] == "ascend950.megamoe.a8w8_wave.v1"


def test_a_custom_registered_builder_still_works_and_says_it_is_custom():
    """自定义建图器 (registry 注册的类) 仍然能用, 身份标成 custom.

    旧契约是 register("orchestration", name, 建图器类); examples 与 tests 都按它注册过。
    适配器接口是新加的, 不能让旧注册失效 —— 但也不能把自定义实现冒充成仓内那两份,
    所以身份里 implementation_id = "custom"。
    """
    from moe_cost_model.builders.mte import MteEventBuilder

    class _Probe(MteEventBuilder):
        pass

    m.register("orchestration", "probe_impl_layer", _Probe)
    rc = [[[32] * 2 for _ in range(4)] for _ in range(2)]
    res = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=32, h=1024, hidden_dim=1024, aic_num=4,
        costs=m.build_analytical_costs(
            h=1024, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        topk=8, p1_override=1, p2_override=1, orchestration="probe_impl_layer")
    ident = res["rank_results"][0]["implementation"]["id"]
    assert ident.startswith("ascend950.custom."), ident
