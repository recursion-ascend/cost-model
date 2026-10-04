"""combine 的写出落点布局 (缺口 10 的结构部分).

    "token_scatter"     落点 = (tokenIdx·topK + topkIdx)·n, 跨度 = token 数 x topk
                        好处: UNPERMUTE 顺序读。代价: 写侧散射。
    "expert_contiguous" 按专家连续写, 跨度 = 本窗行数。读侧改 gather。

本文件钉的是**结构**: 布局 -> 跨度 -> (可填系数的) 代价。系数缺省 0, 所以缺省下换布局
不改时长 —— 这是有意的, 因为定系数要"固定 m 与 n、只扫 token 数"的 run。
两处偏置也写成断言, 免得有人拿这个旋钮当免费收益。
"""
import dataclasses

import pytest

import moe_cost_model as m


def _costs(scatter=0.0, exponent=0.0):
    base = m.build_analytical_costs(
        h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(),
        cube_mac_per_us=m.cube_mac_per_us("fp8"))
    if not scatter:
        return base
    c = m.AnalyticalCombineCosts(scatter_us_per_row=scatter, scatter_exponent=exponent)
    return dataclasses.replace(base, combine_tile=c.tile,
                               combine_write_bytes_per_row=c.write_bytes_per_row)


def _run(layout="token_scatter", scatter=0.0, exponent=0.0, **kw):
    W, LOCAL, PER = 5, 3, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(LOCAL)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(LOCAL)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=9216, aic_num=28,
        topk=6, p1_override=1, p2_override=1, costs=_costs(scatter, exponent),
        options=m.ModelOptions(combine_layout=layout, **kw))["rank_results"][0]


def _combines(rr):
    return [e for e in rr["events"] if e.meta.get("stage") == "combine"]


def test_token_scatter_is_the_default():
    assert m.ModelOptions().combine_layout == "token_scatter"


def test_unknown_layout_is_refused():
    with pytest.raises(ValueError, match="combine_layout"):
        _run(layout="row_major")


def test_layout_decides_the_declared_spread():
    """跨度是布局的推论: 散射 = token 数 x topk; 连续 = 本窗行数."""
    sc = _combines(_run("token_scatter"))
    ct = _combines(_run("expert_contiguous"))
    # 散射: 整卡落点空间 = token 数 x topk, 与本窗行数无关 -> 所有事件同一个值
    (one,) = {e.meta["spread_slots"] for e in sc}
    assert one == 768.0                        # 本夹具 token 数 128 x topk 6
    # 连续: 跨度就是本窗行数 (不散开)
    assert all(e.meta["spread_slots"] == e.meta["m_rows"] for e in ct)
    assert {e.meta["spread_slots"] for e in ct} != {one}


def test_spread_costs_nothing_until_the_coefficient_is_filled():
    """缺省系数 0: 换布局只改申报的跨度, 不改时长 (定系数要扫 token 数的 run)."""
    assert _run("token_scatter")["total_us"] == pytest.approx(
        _run("expert_contiguous")["total_us"])


def test_with_a_coefficient_contiguous_becomes_cheaper():
    """填了系数, 连续布局的写侧才显出便宜 —— 结构是通的."""
    sc = _run("token_scatter", scatter=0.02, exponent=0.5)["total_us"]
    ct = _run("expert_contiguous", scatter=0.02, exponent=0.5)["total_us"]
    assert ct < sc


def test_the_read_side_of_the_trade_is_not_modelled():
    """偏置声明: UNPERMUTE 从顺序读变 gather 的代价**没建模**.

    UNPERMUTE 现在是"字节量 / BW_UNPERMUTE_AGG"一个除法, 与落点布局无关。所以填了
    scatter 系数后 expert_contiguous 会单方面变好 —— 那是模型的偏置, 不是结论。
    """
    for layout in ("token_scatter", "expert_contiguous"):
        ev = [e for e in _run(layout, scatter=0.02, exponent=0.5)["events"]
              if e.meta.get("part") == "unpermute"]
        assert len(ev) == 1
        dur = ev[0].end_us - ev[0].start_us
        if layout == "token_scatter":
            base = dur
    assert dur == pytest.approx(base)          # 两种布局下 UNPERMUTE 完全一样


def test_the_scatter_term_is_superlinear_only_if_asked():
    """缺省指数 0 -> 每行开销与跨度无关; 给了指数才超线性 (两点实测与 0.5 相容)."""
    flat = m.AnalyticalCombineCosts(scatter_us_per_row=0.02)
    assert flat.scatter_us(256, 768) == pytest.approx(flat.scatter_us(256, 256))
    curved = m.AnalyticalCombineCosts(scatter_us_per_row=0.02, scatter_exponent=0.5)
    assert curved.scatter_us(256, 768) > curved.scatter_us(256, 256)
    # 跨度小于行数时不该变成"折扣"
    assert curved.scatter_us(256, 1) == pytest.approx(curved.scatter_us(256, 256))


def test_negative_coefficients_are_refused():
    with pytest.raises(ValueError):
        m.AnalyticalCombineCosts(scatter_us_per_row=-1)
    with pytest.raises(ValueError):
        m.AnalyticalCombineCosts(scatter_exponent=-0.5)
