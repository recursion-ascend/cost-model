"""设计空间扫描: 模型对算子工程师的直接交付物.

一行不只给时长, 还要回答"收益在哪个 stage、最大的等待是什么、少搬了多少字节、
这个方案下模型自己的不变量守住了没有"。
"""
import moe_cost_model as m
from linkutil import links

OPT = m.ModelOptions


def _runner(local=3, hd=9216):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6

    def run(options):
        return m.simulate_routing_counts(
            routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hd, aic_num=28,
            costs=m.build_analytical_costs(
                h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
            p1_override=1, p2_override=1, topk=6, options=options)
    return run


def _rows():
    return m.design_space(_runner(), {
        "基线": OPT(),
        "UB 深度 2": OPT(links=links(2)),
        "ACT 不物化": OPT(links=links(location="onchip")),
        "那份实现": m.MEGAMOE_A8W8.options,
    })


def test_baseline_is_the_reference_point():
    rows = _rows()
    assert rows[0]["is_baseline"] and rows[0]["delta_us"] == 0.0
    assert all(not r["is_baseline"] for r in rows[1:])


def test_delta_is_measured_against_the_baseline():
    rows = {r["name"]: r for r in _rows()}
    for name, r in rows.items():
        assert abs(r["delta_us"] - (r["total_us"] - rows["基线"]["total_us"])) < 1e-9


def test_onchip_shows_the_trade_it_makes():
    """不物化: 少搬字节 (访存量差为负) + 多花墙钟, 两边都要在行里看得见."""
    r = {x["name"]: x for x in _rows()}["ACT 不物化"]
    assert r["delta_us"] > 0
    assert any(v < 0 for v in r["traffic_delta_bytes"].values())


def test_path_attribution_explains_a_change():
    """时长变了就要说得出变在关键路径的哪一段 —— 否则这一行帮不了工程师."""
    r = {x["name"]: x for x in _rows()}["UB 深度 2"]
    assert abs(r["delta_us"]) > 1.0
    assert r["path_stage_delta_us"], "时长变了却说不出变在哪个 stage"


def test_invariant_guard_flags_the_static_implementation():
    """护栏列: 那份实现是静态分核, 有可避免空闲 -> 该行的收益不能当准数看."""
    rows = {r["name"]: r for r in _rows()}
    assert rows["基线"]["invariant_ok"]
    assert not rows["那份实现"]["invariant_ok"]
    assert rows["那份实现"]["invariant_violations"]


def test_format_is_one_line_per_point():
    rows = _rows()
    text = m.format_design_space(rows)
    lines = text.splitlines()
    assert len(lines) == len(rows) + 1          # 表头 + 每个方案一行
    for r in rows:
        assert str(r["name"]) in text


# ---- compare_variants: 换得了 tiling 几何与策略, 不只是 ModelOptions ----

def _scenario():
    """一个不依赖实测 tiling 真值的小场景 (Scenario 是方案描述的载体)."""
    W, PER, LOCAL = 5, 64, 3
    rc = tuple(tuple(tuple(0 if s == d else PER for s in range(W))
                     for _ in range(LOCAL)) for d in range(W))
    tok = sum(rc[d][e][1] for d in range(W) for e in range(LOCAL)) // 6
    return m.Scenario(
        workload=m.Workload(tokens=tok, topk=6, routing="explicit", counts=rc),
        h=5120, hidden_dim=9216, aic_num=28, p1_override=1, p2_override=1)


def test_compare_variants_can_change_tiling_and_strategies():
    """design_space 的入口只换 ModelOptions, 所以这些维度**换不了**:
    tile 几何在 kernel 上, 分核/打包/调度策略在 Scenario 上。
    compare_variants 以 Scenario 的点分路径为方案描述, 覆盖面与场景文件一致。
    """
    rows = m.compare_variants(_scenario(), {
        "基线": {},
        "tile_n 128": {"kernel.tile_n": 128},
        "GMM2 粒度 2": {"options.granularity": {"gmm2": 2}},
        "静态分核 + 最闲核优先": {"options.late_bind_pools": [],
                                  "core_assignment": "greedy_least_busy"},
    }, check_bounds=False)
    by = {r["name"]: r for r in rows}
    assert by["基线"]["is_baseline"]
    # tile_n 减半 -> n 方向的 tile 数翻倍 -> 事件数必须变 (几何真的进了事件图)
    assert by["tile_n 128"]["events"] > by["基线"]["events"]
    # 粒度变粗 -> 事件数必须变少
    assert by["GMM2 粒度 2"]["events"] < by["基线"]["events"]
    # 策略换了也要有后果: 静态分核下 tile->核 的分配变了
    assert by["静态分核 + 最闲核优先"]["total_us"] != by["基线"]["total_us"]


def test_every_row_reports_utilization_and_the_binding_bound():
    """"资源利用率"与"性能瓶颈"要逐方案给, 否则比较只剩一个时长数字."""
    rows = m.compare_variants(_scenario(), {"基线": {}, "tile_n 128":
                                            {"kernel.tile_n": 128}},
                              check_bounds=False)
    for r in rows:
        util = r["utilization"]
        assert util, "利用率表不能为空"
        # 角色名从资源名取, 不写死: 至少 Cube 要在里面
        assert any(k.startswith("AIC") for k in util)
        assert all(0.0 <= v <= 1.0 for v in util.values())
        # 三条下界里哪条绑定 + 离它多远
        assert r["binding_bound"] in ("compute", "bandwidth", "dependency", "-")
        assert r["lower_us"] > 0 and r["over_bound_pct"] >= 0
    text = m.format_design_space(rows)
    assert "利用率" in text and "瓶颈" in text


def test_avoidable_idle_is_read_differently_under_static_pinning():
    """"有就绪的活却有核空闲"这个量的判读取决于绑定方式 (analysis/idle.py 的规则).

    晚绑定的池: avoidable 必须 0, 不为 0 是不变量没守住 -> 那一行收益不可信。
    静态钉核的池: avoidable 是**那种分核方式留下的可回收空闲** —— 是结论, 不是缺陷。
    2026-10-08 定位过一次: examples/scenario_basic.toml 用 profile="megamoe-a8w8",
    那份 profile 把 late_bind_pools 设成 (), 于是 AIC 上有 1022.8 核·us 的 avoidable。
    两种虚报成因 (共位、相位组) 都不在场 (AIV0 恒 0, pipeline 未开), 只把 AIC 入池
    avoidable 立刻为 0、时长 1751.48 -> 1727.97。所以它是真帐。
    把两者混在一个"违反"里, 会把一个结论读成一个 bug。
    """
    base = _scenario()
    rows = {r["name"]: r for r in m.compare_variants(base, {
        "静态钉核": {"options.late_bind_pools": []},
        "AIC 入池": {"options.late_bind_pools": ["AIC"]},
    }, check_bounds=False)}

    static = rows["静态钉核"]
    assert static["late_bind_pools"] == ()
    # 静态钉核下的 avoidable 不计入"违反", 而是单列出来
    assert static["invariant_ok"], static["invariant_violations"]
    assert not static["invariant_violations"]

    late = rows["AIC 入池"]
    assert late["late_bind_pools"] == ("AIC",)
    # 入池的那个角色必须守住不变量
    assert late["avoidable_idle_us"].get("AIC", 0.0) <= 1e-6
    assert late["invariant_ok"]
