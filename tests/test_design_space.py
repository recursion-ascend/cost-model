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
    """护栏列: 那份实现是静态发牌, 有可避免空闲 -> 该行的收益不能当准数看."""
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
