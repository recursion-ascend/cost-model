"""下界: 墙钟在物理上不可能低于 算法事实 + 硬件规格 推出的那条线.

这组测试有两个职责:
  1. 钉住下界本身只用算法事实 (换编排不变, 换形状才变);
  2. **把两处已知漏账钉成测试** —— 它们现在确实会穿透带宽下界, 修好之前这两条
     测试断言"能被检出", 修好之后要改成断言"不再穿透"。
"""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import moe_cost_model as m
from moe_cost_model.analysis.bounds import (TOL, BoundViolation, Bounds,
                                            bandwidth_bound_us,
                                            check_wall_clock,
                                            compute_bound_us,
                                            dependency_bound_us,
                                            workload_facts)

SCENARIO = Path("examples/scenario_basic.toml")
PIPE = {"options.pipeline": {"queues": {"mte_aic": 2, "cube": 2}}}


def _shape(h=6144, hidden=4096, rows=(100, 200, 0), half=2):
    return SimpleNamespace(h=h, hidden_dim=hidden, expert_tokens=rows,
                           kernel=SimpleNamespace(activation_n_half=half))


# ------------------------------------------------------- 算法事实

def test_mac_counts_match_the_matmul_definition():
    """GMM1 = A(m x h) x B(h x hidden); GMM2 = A(m x I) x B(I x h). 不含 tile 口径."""
    f = workload_facts(_shape(rows=(100, 200, 0)))
    m_total, h, hidden, inter = 300, 6144, 4096, 2048
    assert f.gmm1_mac == m_total * h * hidden
    assert f.gmm2_mac == m_total * inter * h
    assert f.total_mac == f.gmm1_mac + f.gmm2_mac
    assert f.rows_total == m_total
    assert f.experts_active == 2          # 0 行的专家不算


def test_weight_bytes_counted_once_per_active_expert():
    """权重每专家至少搬一次 —— 这是下界; L2 复用只能让实际更接近它, 不能更少."""
    f = workload_facts(_shape(rows=(100, 200, 0)))
    assert f.gmm1_b_bytes == 2 * 6144 * 4096
    assert f.gmm2_b_bytes == 2 * 2048 * 6144
    # 行数为 0 的专家不搬权重
    assert workload_facts(_shape(rows=(100, 0, 0))).gmm1_b_bytes == 6144 * 4096


def test_facts_do_not_depend_on_orchestration():
    """同一形状, 换 activation_n_half 之外的任何编排都不该改算法事实."""
    a = workload_facts(_shape())
    b = workload_facts(_shape())
    assert a == b


def test_materialised_link_adds_gmm2_a_bytes():
    """ACT 物化到 GM 再被 GMM2 读回是编排选择, 它确实多搬字节 —— 要算进去."""
    on = workload_facts(_shape(), gmm2_a_from_gm=True)
    off = workload_facts(_shape(), gmm2_a_from_gm=False)
    assert on.gmm2_a_bytes > 0 and off.gmm2_a_bytes == 0
    assert on.gm_to_l1_bytes > off.gm_to_l1_bytes


# ------------------------------------------------------- 三个下界

def test_compute_bound_is_mac_over_cores_times_rate():
    f = workload_facts(_shape())
    got = compute_bound_us(f, cube_mac_per_us=1e7, active_cores=28)
    assert got == pytest.approx(f.total_mac / (1e7 * 28))
    # 速率未标定 (<=0) 时给 0, 不参与取最大
    assert compute_bound_us(f, cube_mac_per_us=0, active_cores=28) == 0.0


def test_bandwidth_bound_takes_the_smaller_of_two_hardware_limits():
    f = workload_facts(_shape())
    # 聚合带宽更紧
    us, rate, who = bandwidth_bound_us(f, bw_per_core_bytes_per_us=51900,
                                       active_cores=28,
                                       aggregate_bytes_per_us=1.0e6)
    assert who == "aggregate" and rate == 1.0e6
    assert us == pytest.approx(f.gm_to_l1_bytes / 1.0e6)
    # 每核合计更紧
    us2, rate2, who2 = bandwidth_bound_us(f, bw_per_core_bytes_per_us=10_000,
                                          active_cores=28,
                                          aggregate_bytes_per_us=1.0e9)
    assert who2 == "per_core" and rate2 == 10_000 * 28


def test_dependency_bound_sums_the_chain():
    assert dependency_bound_us({"dispatch": 1.0, "gmm1": 2.0, "activation": 0.5,
                                "gmm2": 3.0, "combine": 1.5}) == pytest.approx(8.0)


def test_binding_names_the_largest_bound():
    f = workload_facts(_shape())
    b = Bounds(compute_us=100.0, bandwidth_us=10.0, dependency_us=1.0, facts=f)
    assert b.binding == "compute" and b.lower_us == 100.0
    b2 = Bounds(compute_us=10.0, bandwidth_us=100.0, dependency_us=1.0, facts=f)
    assert b2.binding == "bandwidth"


def test_check_wall_clock_rejects_below_bound_and_accepts_above():
    f = workload_facts(_shape())
    b = Bounds(compute_us=100.0, bandwidth_us=0.0, dependency_us=0.0, facts=f)
    check_wall_clock(b, 100.0)              # 相等放过
    check_wall_clock(b, 120.0)
    with pytest.raises(BoundViolation, match="低于物理下界"):
        check_wall_clock(b, 80.0)


# ------------------------------- 端到端: 下界进结果, 且两处漏账被检出

@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_bounds_are_attached_to_every_rank():
    r = m.simulate(m.load_scenario(SCENARIO))
    for rr in r["rank_results"].values():
        b = rr["bounds"]
        for k in ("compute_us", "bandwidth_us", "dependency_us", "lower_us",
                  "binding", "total_mac", "gm_to_l1_bytes"):
            assert k in b
        assert b["lower_us"] == max(b["compute_us"], b["bandwidth_us"],
                                    b["dependency_us"])


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_declared_traffic_is_at_least_the_algorithmic_minimum():
    """申报的 GM->L1 字节不得少于算法必搬的字节.

    2026-10-05 修掉的漏账: builders/gmm2.py 只申报 a_gm (激活), 不申报 B 流
    (k2 x cols 的权重); GMM1 一直两股都申报。当时 scenario_basic 上申报 1660.9MB <
    算法必搬 2420.1MB (差 759.2MB ≈ GMM2 权重 805.3MB)。申报量低于算法下界在物理上
    不可能 —— 那是漏账, 不是口径差异。
    申报量**可以高于**下界 (模型按 tile 重复读权重, 多个 m-group 各读一次)。
    """
    r = m.simulate(m.load_scenario(SCENARIO))
    rr = r["rank_results"][0]
    declared = rr["traffic_bytes"]["R0.gm_to_l1"]
    algorithmic = rr["bounds"]["gm_to_l1_bytes"]
    assert declared >= algorithmic, (
        f"申报 {declared / 1e6:.1f}MB 少于算法必搬 {algorithmic / 1e6:.1f}MB")


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_phase_pipelining_respects_the_bandwidth_bound():
    """相位流水不得穿透带宽下界: 载入必须占住本核那条 MTE2 管道.

    2026-10-05 修掉的漏账: .ld 相位 resources=() 不占任何资源, GMM2 的载入又整段裹在
    AIC 事件里 —— 等于每个核有两条载入管道, 聚合载入带宽翻倍。当时墙钟 1221.80us
    低于带宽下界 1665.37us 达 26.6%, 被当成"相位流水省了 30.24%"。
    修法是硬件事实: 一个 AI Core 只有一条 MTE2, 所以 GMM1 与 GMM2 的载入都占
    MTE2:c{core}; 双缓冲 (queues.mte_aic = L1 槽数) 只决定能提前多少发起, 不决定
    能同时搬几笔。修后真实收益是 0.00% (1751.48, 与基线逐位相同), 不是 -30%。
    """
    base = m.load_scenario(SCENARIO)
    r = m.simulate(base.with_overrides(PIPE))
    rr = r["rank_results"][0]
    assert rr["bounds"]["violation"] is None
    assert r["kernel_total_us"] >= rr["bounds"]["lower_us"]


def test_load_phase_holds_the_cores_mte2_pipe():
    """硬件事实: 一个 AI Core 一条 MTE2, 所以同一个核上两笔载入不得重叠.

    不占的话载入并发只受 L1 槽数限制 (28 核 x d 笔同时满带宽), 聚合载入带宽会超过
    核数 x BW_L1_GM 这条硬件规格。
    写成计数信号量 (容量 1) 而不是独占资源, 是为了走 late-bind 的 "c*" 占位重映射 ——
    独占资源会把 .ld 钉在建图时的占位核号上, 与它所在相位组绑定的核冲突。
    """
    base = m.load_scenario(SCENARIO)
    r = m.simulate(base.with_overrides(PIPE))
    lds = [e for e in r["rank_results"][0]["events"] if e.name.endswith(".ld")]
    assert lds, "开了相位流水却没有载入相位事件"
    holding = [e for e in lds
               if any("MTE2:" in t for t, _ in getattr(e, "acquires", ()))]
    assert holding, "载入相位没有占住 MTE2 管道"
    # 同一个核上两笔载入不得重叠 —— 容量 1 的直接推论, 这里直接验时间线。
    # meta["core"] 由引擎改写成**实际绑定**的核号 (晚绑定下建图时的占位号不作数)。
    by_core = {}
    for e in holding:
        by_core.setdefault(e.meta.get("core"), []).append((e.start_us, e.end_us))
    for core, spans in by_core.items():
        spans.sort()
        for (a0, a1), (b0, _) in zip(spans, spans[1:]):
            assert b0 >= a1 - 1e-9, f"核 {core} 上两笔载入重叠: {a1} > {b0}"


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_check_bounds_raises_by_default_and_records_when_switched_off():
    """断言开关本身要有效, 且缺省是**抛**而不是静默.

    用一个带宽低得荒谬的平台把带宽下界顶到墙钟之上来触发 —— 不依赖任何现存漏账,
    所以漏账修好之后这条测试照样有效。
    """
    tiny = m.PlatformSpec(name="tiny-bw", source="test fixture",
                          hbm_bytes_per_us=1.0e3, fabric_bytes_per_us=1.0e3)
    sc = m.load_scenario(SCENARIO)
    with pytest.raises(BoundViolation, match="低于物理下界"):
        m.simulate(sc, platform=tiny)
    r = m.simulate(sc, platform=tiny, check_bounds=False)
    rr = r["rank_results"][0]
    assert rr["bounds"]["violation"], "降级之后也要把诊断记进结果, 不能悄悄丢掉"
    assert rr["bounds"]["bandwidth_limited_by"] == "aggregate"


# ------------------------------- 执行单元数 != 队列深度

#: 每核每种**执行单元**恒为 1 条 (硬件事实). 名字见 builders/pipeline_expand.py。
EXEC_UNITS = ("MTE2", "FIXPIPE", "MTE_AIV")


def _unit_concurrency(events):
    """每个执行单元实例上的最大同时占用数 (按信号量名分组).

    必须按**信号量名**分组而不是按 meta["core"]: AIC:c7 与 AIV0:c7 是两个物理核,
    各有自己的搬运单元, 它们重叠是对的。第一次查这个问题时按核分组, 把
    "AIC 的载入与 AIV 的载入重叠"误报成了违规。
    """
    import collections
    spans = collections.defaultdict(list)
    for e in events:
        for tok, _ in getattr(e, "acquires", ()):
            kind = tok.split(".")[-1].split(":")[0]
            if kind in EXEC_UNITS:
                spans[tok].append((e.start_us, e.end_us))
    out = {}
    for tok, ss in spans.items():
        pts = []
        for a, b in ss:
            pts += [(a, 1), (b, -1)]
        pts.sort()
        cur = mx = 0
        for _, d in pts:
            cur += d
            mx = max(mx, cur)
        kind = tok.split(".")[-1].split(":")[0]
        out[kind] = max(out.get(kind, 0), mx)
    return out


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_execution_units_stay_at_one_even_when_queue_depths_are_raised():
    """队列深度是"能攒几笔", 执行单元是"同时能跑几笔" —— 后者恒为 1.

    这两件事混在一起就是 2026-10-05 那个 bug 的根源: 只有队列深度、没有执行单元约束,
    等于给每个核凭空多出几条管道 (载入可无限并行, 墙钟低于带宽下界 26.6%)。
    当时只补了 mte_aic 的 MTE2; fix 与 mte_aiv 的单元是后补的 —— 这条测试把三个
    一起钉住, 且**把深度都调到 >1**, 否则深度 1 下两者重合, 测不出区别。
    """
    sc = m.load_scenario(SCENARIO).with_overrides({
        "options.pipeline": {
            "queues": {"mte_aic": 2, "cube": 2, "fix": 4, "mte_aiv": 3},
            "phases": {"act_load_bw_bytes_per_us": 157000.0},
        }})
    r = m.simulate(sc)
    got = _unit_concurrency(r["rank_results"][0]["events"])
    assert got, "没有任何执行单元被占用 —— 相位没拆?"
    for kind, mx in got.items():
        assert mx <= 1, f"{kind} 同时跑了 {mx} 笔, 但每核只有一条"
    # MTE2 与 MTE_AIV 必须真的被用上 (不是空操作); FIXPIPE 见下一条
    assert got.get("MTE2") == 1
    assert got.get("MTE_AIV") == 1


@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
def test_fixpipe_unit_is_declared_but_currently_dormant():
    """FIXPIPE 这条约束现在**量不出来**: fix 相位时长恒为 0.

    口径是"结果写出 (数据释放事件) 忽略不计" (见 builders/pipeline_expand.py 的 fix
    相位与 PhaseRates.fix_bw_bytes_per_us 的说明), 所以给了 fix_bw_bytes_per_us 也不
    影响时长 —— 那个字段现在是"保留以备改口径"。
    于是 FIXPIPE 是一条**预置的护栏**: 不花代价, 等口径改了自动生效。
    这条测试钉住"它确实被申报了"与"它现在确实是空操作"两件事, 避免把它当成已验证的约束。
    """
    sc = m.load_scenario(SCENARIO).with_overrides({
        "options.pipeline": {"queues": {"mte_aic": 2, "cube": 2, "fix": 4},
                             "phases": {"fix_bw_bytes_per_us": 157000.0}}})
    r = m.simulate(sc)
    ev = r["rank_results"][0]["events"]
    fx = [e for e in ev if (e.meta or {}).get("phase") == "fix"]
    assert fx, "没有 fix 相位事件"
    assert all(e.acquires for e in fx)
    assert any(any("FIXPIPE:" in t for t, _ in e.acquires) for e in fx), \
        "fix 相位没有申报 FIXPIPE 单元"
    assert sum(e.end_us - e.start_us for e in fx) == 0.0, \
        "fix 相位有了非零时长 —— 口径变了, 把这条测试翻成断言 FIXPIPE 并发 <= 1"


# ------------------------- 字节申报不得从时长倒推

@pytest.mark.skipif(not SCENARIO.exists(), reason="需要场景文件")
@pytest.mark.parametrize("extra,label", [
    ({}, "缺省"),
    ({"calibration.gmm1_tile_restart_us": 0.5}, "serial 有 chunk_restart"),
])
def test_declared_bytes_do_not_depend_on_phase_pipelining(extra, label):
    """开不开相位流水, 申报的 GM→L1 字节必须**逐位相同**.

    相位流水是编排旋钮, 它改变的是"什么时候搬", 不是"搬多少"。
    2026-10-05 之前三处都从时长倒推字节 (时长 x 名义带宽), 于是拆相位会改变申报量:
      COMBINE  base_dur x BW_SCATTER   —— 连常数都标着"已不用" (已删)
      GMM1     load_us x BW_L1_GM      —— max 口径下只拿到较大那一股 (已改为透传)
      GMM2     load_us x BW_L1_GM      —— serial 口径下还把 chunk_restart 当成字节 (已改为透传)
    现在建图器按算法逐项申报, 相位展开只做重新分配。
    """
    base = m.load_scenario(SCENARIO)
    flat = m.simulate(base.with_overrides(dict(extra)))
    split = m.simulate(base.with_overrides(
        {**extra, "options.pipeline": {"queues": {"mte_aic": 2, "cube": 2}}}))
    a = flat["rank_results"][0]["traffic_bytes"]
    b = split["rank_results"][0]["traffic_bytes"]
    assert set(a) == set(b), f"{label}: 拆相位改变了申报的通路集合"
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-12), (
            f"{label}: 拆相位把 {k} 从 {a[k]:.0f} 改成了 {b[k]:.0f} —— "
            "字节不该随编排旋钮变")


def test_per_core_bandwidth_is_capped_by_aggregate_spec():
    """单核带宽常数乘核数不得超过整卡聚合规格 —— 包括 NZ 布局的 B 流.

    2026-10-05 之前只给 A 流加帽: 一个够大的 bw_l1_gm_b_nz 能让 28 个核合起来抽出
    2.24 TB/s, 超过 950PR 的 1.60 TB/s 规格。下界断言会抓住它 (墙钟低于带宽下界
    30.8%), 但更该在源头收敛。
    """
    kw = dict(h=6144, dispatch_mechanistic=m.DispatchMechanisticLatency(),
              kernel=m.KernelConfig(weight_nz=True), bw_l1_gm_b_nz=80000.0)
    free = m.build_analytical_costs(**kw).gmm1_tile.__self__
    capped = m.build_analytical_costs(platform=m.ASCEND_950PR, active_cores=28,
                                      **kw).gmm1_tile.__self__
    assert free.bw_b * 28 > m.ASCEND_950PR.hbm_bytes_per_us      # 不加帽会超规格
    assert capped.bw_b * 28 == pytest.approx(m.ASCEND_950PR.hbm_bytes_per_us)
