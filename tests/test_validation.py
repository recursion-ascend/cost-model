"""结构校验 (步骤 5) 与实测 trace 结构比对 (步骤 6).

步骤 5: 不变量用 IR 的词表写, 所以**与 kernel 无关** —— 换一份实现照样能用。每条规则都要
能在反例上报警, 又不能在正常图上误报; 一个永远不会失败的校验器等于没有校验器, 所以这里
逐条给反例。

步骤 6: 比对**先声明能比什么**。实测 trace 里没有搬运字节字段, 也没有缓冲槽的取/还记录,
所以那两项对不了 —— 这是数据的限制, 不是"还没做"。能比的是波数、逐专家分布的形状、核的
覆盖面, 以及条数比在各专家间是否一致。
"""
from pathlib import Path

import pytest

import moe_cost_model as m
from moe_cost_model.scheduler.events import Event
from moe_cost_model.validation import (check_graph, compare_run, read_run,
                                       read_run_config, read_trace)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BS128 = "20260930_154407_446172_bs128_h5120_i4608_k6_cyclic_noshared"
BS8192 = "20260930_145851_854111_bs8192_h5120_i4608_k6_cyclic_noshared"


# --------------------------------------------------------------- 步骤 5: 不变量

def _good_graph():
    """一张满足全部不变量的小图: 跨事件持有一个 UB 槽, 取还配对且同核."""
    return [
        Event("gmm1", ("AIC:c0",), 10.0, acquires=(("UB:slot:c0", 1),)),
        Event("act", ("AIV0:c0",), 5.0, deps=("gmm1",),
              releases=(("UB:slot:c0", 1),), colocate_with="gmm1"),
    ]


def test_a_valid_graph_passes_every_rule():
    assert check_graph(_good_graph(), capacities={"UB:slot:c0": 1}) == []


@pytest.mark.parametrize("rule, events, caps", [
    # C1 取了不还: 台账会漂, 约束悄悄失效 (表现是更快的排程, 不是报错)
    ("C1 缓冲槽取还配对",
     [Event("a", ("AIC:c0",), 10.0, acquires=(("UB:x:c0", 1),))], None),
    # C1 跨核取还: 片上缓冲是按核的
    ("C1 缓冲槽同核",
     [Event("b", ("AIC:c1",), 10.0, acquires=(("UB:x:c0", 1),)),
      Event("c", ("AIC:c0",), 10.0, releases=(("UB:x:c0", 1),))], None),
    # C2 执行单元容量 > 1: 等于声称硬件多了一条管道
    ("C2 执行单元容量为 1",
     [Event("d", ("AIC:c0",), 10.0, acquires=(("MTE2:c0", 1),),
            releases=(("MTE2:c0", 1),))], {"MTE2:c0": 2}),
    # C3 共位落在别的核: 共位表达的是硬件通路 (Fixpipe 只在绑定对内)
    ("C3 共位同核",
     [Event("e", ("AIC:c0",), 10.0),
      Event("f", ("AIC:c1",), 10.0, colocate_with="e")], None),
    # C4 自环: 永不就绪
    ("C4 无自环", [Event("g", ("AIC:c0",), 10.0, deps=("g",))], None),
    # C4 依赖不存在的事件
    ("C4 依赖存在", [Event("h", ("AIC:c0",), 10.0, deps=("ghost",))], None),
    # C5 新通路没登记: 字节照样累加, 但方向与两端丢了
    ("C5 搬运两端已知",
     [Event("i", ("AIC:c0",), 10.0,
            channel_bytes=(("brand_new_channel", 128, 0.0),))], None),
    # C6 零时长占执行单元: 让"这条管道忙了多久"算不平
    ("C6 零时长不占单元",
     [Event("j", ("AIC:c0",), 0.0, acquires=(("MTE2:c0", 1),),
            releases=(("MTE2:c0", 1),))], {"MTE2:c0": 1}),
])
def test_each_rule_fires_on_its_counterexample(rule, events, caps):
    got = [v.rule for v in check_graph(events, capacities=caps)]
    assert rule in got, f"{rule} 没有在反例上报警, 实际报的是 {got}"


def test_every_violation_says_what_goes_wrong():
    """违规必须带后果说明 —— 不然读到的人不知道该不该管."""
    bad = [Event("a", ("AIC:c0",), 10.0, acquires=(("UB:x:c0", 1),))]
    for v in check_graph(bad):
        assert len(v.consequence) > 20, f"{v.rule} 没说清后果"


def test_the_real_graphs_of_both_adapters_satisfy_the_invariants():
    """仓内两份实现的真实事件图都要过 —— 这是步骤 5 的验收."""
    from moe_cost_model.implementations import A8W8WaveV1, LayeredV1
    from moe_cost_model.shape import MegaMoeShape
    P = m.MEGAMOE_A8W8
    for adapter, urma in ((A8W8WaveV1(), False), (LayeredV1(), True)):
        costs = m.build_analytical_costs(
            h=1024, dispatch_mechanistic=m.DispatchMechanisticLatency(),
            **({"urma_mechanistic": m.UrmaMechanisticLatency()} if urma else {}))
        kw = P.shape_kw()
        kw["kernel"] = m.KernelConfig(topo_urma=urma, combine_meta_bytes_per_row=32)
        shape = MegaMoeShape(expert_tokens=(64, 64), token_num=16, h=1024,
                             hidden_dim=1024, aic_num=4, topk=8, p1_override=1,
                             p2_override=1,
                             expert_source_tokens=((32, 32), (32, 32)), **kw)
        model = m.A8W8WaveCostModel(costs, P.options)
        events, _ = model.build_events(shape)
        assert check_graph(events) == [], adapter.identity().key


# --------------------------------------------------------------- 步骤 6: trace

@pytest.mark.skipif(not (DATA / BS128).exists(), reason="缺打点数据")
def test_a_complete_trace_reads_cleanly():
    """完整文件不该被判成截断; 但 note 会说明"只读了哪个视图"——那是信息不是问题."""
    trace = read_trace(next((DATA / BS128).glob("*_trace_rank0.json")))
    assert not trace.truncated and trace.recovered_records == 0
    # 本仓的 trace 一个文件含同一次运行的两个视图 ("完整流水" / "隐藏 WAIT"), 读取器
    # 只取一个并把跳过了哪个写进 note —— 静默跳过会让人以为文件里只有这些事件。
    assert "只读了" in trace.note and "完整流水" in trace.note
    assert trace.rank == 0 and len(trace.events) > 1000
    stages = trace.by_stage()
    for stage in ("gmm1", "activation", "gmm2", "combine", "dispatch"):
        assert stages.get(stage, 0) > 0, stage
    assert trace.waves("gmm1") == (0, 1)


@pytest.mark.skipif(not (DATA / BS8192).exists(), reason="缺打点数据")
def test_a_truncated_trace_is_recovered_and_flagged():
    """两个 bs8192 run 的 8 个文件都在同一个字节数处断在记录中间.

    必须能救回前面的完整记录 (否则两个最大的 run 整个用不了), 同时**必须标记**它是截断的
    —— 不标就会把"事件数比模型少"当成模型的问题。
    """
    trace = read_trace(next((DATA / BS8192).glob("*_trace_rank0.json")))
    assert trace.truncated and "截断" in trace.note and "下界" in trace.note
    assert len(trace.events) > 20000, "截断文件也该救回大部分记录"
    assert trace.by_stage().get("gmm1", 0) > 0


@pytest.mark.skipif(not DATA.exists(), reason="缺打点数据")
def test_the_trace_corpus_state_is_what_the_code_assumes():
    """约束数据现状: 16 个文件完整, 8 个截断 (两个 bs8192 run 的全部 rank).

    哪天数据补齐或重采, 这里会红 —— 那是好事: validation/trace 的说明与比对工具的措辞都
    建立在这个事实上, 事实变了说明也要改。
    """
    complete = truncated = 0
    for run in sorted(p for p in DATA.iterdir() if p.is_dir()):
        for rank, trace in read_run(run).items():
            if trace.truncated:
                truncated += 1
            else:
                complete += 1
    assert (complete, truncated) == (16, 8), (complete, truncated)


@pytest.mark.skipif(not (DATA / BS128).exists(), reason="缺打点数据")
def test_run_config_maps_onto_the_model_vocabulary():
    """config.json5 的字段名与模型的不同, 映射要对.

    tokens/hidden/intermediate/ep/experts -> token 数 / h / hidden_dim / world /
    **全局**专家数。模型侧的形状从这里建, 不从 tiling 真值建: data/*/raw/ 整个被
    gitignore, 六个 run 的 tiling_rank0.bin 都不在仓里, 所以 examples/ 里那六个场景在
    干净克隆上跑不起来。
    """
    cfg = read_run_config(DATA / BS128)
    # hidden_dim 是 2I: 目录名与 config.json5 里的 i4608 是 I
    assert (cfg["tokens"], cfg["h"], cfg["intermediate"]) == (128, 5120, 4608)
    assert cfg["hidden_dim"] == 9216
    assert (cfg["topk"], cfg["world"], cfg["experts_total"]) == (6, 4, 12)
    assert cfg["local_experts"] == 3 and cfg["aic_cores"] == 28
    assert cfg["routing"] == "cyclic" and cfg["dtype"] == "fp8_e5m2"


@pytest.mark.skipif(not (DATA / BS128).exists(), reason="缺打点数据")
def test_structural_comparison_reports_ratios_and_flags_what_needs_explaining():
    """比对要报**事实**: 条数比、逐专家是否一致、波数、核覆盖.

    实测 108 个 GMM1 tile 对模型 27 个, 比值 4.00 且逐专家一致, 波数两边都是 2。
    比值不为 1 必须被标出来要解释 —— 两边都按 tile 计数 (kernel 的 MOE_PROFILE_BEGIN 带
    ProfileTile(mLoc,nLoc)), 所以它要么是采集含多轮 (config 里 warmup: 3), 要么是 tile
    网格真的不同。把它当成"口径不同, 没事"是自欺。
    """
    cfg = read_run_config(DATA / BS128)
    trace = read_trace(next((DATA / BS128).glob("*_trace_rank0.json")))
    scenario = m.Scenario(
        workload=m.Workload(tokens=cfg["tokens"], topk=cfg["topk"], world=cfg["world"],
                            local_experts=cfg["local_experts"], routing=cfg["routing"],
                            seed=cfg["seed"]),
        h=cfg["h"], hidden_dim=cfg["hidden_dim"], aic_num=cfg["aic_cores"],
        profile="megamoe-a8w8", calibration=m.Calibration(cube_mac_per_us=2.7e7))
    events = m.simulate(scenario)["rank_results"][0]["events"]
    report = compare_run(trace, events, BS128)
    by_stage = {s.stage: s for s in report.stages}

    # 2026-10-08: 这四个 stage 逐条对上了, 比值 1.00。此前报 4.00 (108 vs 27), 两个
    # 2 倍都在**比对工具侧**, 建模侧零差异:
    #   * read_run_config 把 config.json5 的 intermediate (I=4608) 当 hidden_dim 返回,
    #     而模型的 hidden_dim 是 2I=9216 -> 模型按一半宽度建图, 27 而不是 54 个 GMM1 tile;
    #   * read_trace 把文件里**同一次运行的两个视图** (pid "完整流水" / "隐藏 WAIT",
    #     ts 与 dur 逐位相同) 都收了 -> 实测条数翻倍。
    for st in ("gmm1", "activation", "gmm2", "combine"):
        s_ = by_stage[st]
        assert s_.trace_count == s_.model_count, (st, s_.trace_count, s_.model_count)
        assert s_.ratio == pytest.approx(1.0), st
        assert s_.consistent, f"{st}: 逐专家比值应当一致"
        assert len(s_.trace_waves) == len(s_.model_waves) == 2, st
        assert s_.trace_cores == s_.model_cores == 28, st
    assert (by_stage["gmm1"].trace_count, by_stage["gmm1"].model_count) == (54, 54)
    assert (by_stage["gmm2"].trace_count, by_stage["gmm2"].model_count) == (60, 60)

    # dispatch 仍不为 1: 那是**粒度**口径 (模型成批 vs kernel 行级软流水), 另一个问题。
    issues = "\n".join(report.issues())
    assert "dispatch: 条数比" in issues
    assert "warmup" not in issues, "采集含多轮这个假设已被证伪, 不该再出现在诊断里"

    # dispatch 的实测事件不带专家号, 所以"逐专家"这一项不可比 —— 不该报成问题
    assert not by_stage["dispatch"].comparable_by_expert
    assert "专家集合不同" not in issues


def test_time_group_count_is_only_evidence_not_a_normaliser():
    """时间段数只作证据, 不拿来归一化.

    同一个 run 的不同 stage 用间隔启发式切出来是 7/10/3/5/14 段, 彼此矛盾 —— 把这种推断
    写进比值, 比不写更糟 (会得出一个看着精确的错数)。
    """
    from moe_cost_model.validation.compare import StageComparison, time_groups
    assert time_groups([0.0, 1.0, 2.0]) == 1                   # 样本太少: 不猜
    assert time_groups([0, 1, 2, 3, 100, 101, 102, 103]) == 2  # 一个明显空档
    stage = StageComparison(stage="gmm1", trace_count=108, model_count=27,
                            trace_time_groups=7)
    assert stage.ratio == pytest.approx(4.0)       # 原始比值, 没被段数除过
    assert not hasattr(stage, "per_iteration_ratio")
