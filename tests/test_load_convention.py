"""搬运口径: A 流与 B 流相加, 而不是取较慢的一股.

三个实测点 (20260930 的 run, h=5120 hidden=9216 -> K=5120, cols=256, wb=2) 把两个
混淆变量分开了:
    bs36    m= 72,  1 个 m-group, 28 核
    bs128   m=256,  1 个 m-group, 28 核   <- 与 bs36 只差 m
    bs8192  m=256, 12 个 m-group, 28 核   <- 与 bs128 只差 m-group 数

max 口径下 B 流 (2.62MB) 恒大于 A 流 (<=1.31MB), 所以 tile 时长不随 m 变 —— 而 bs36 到
bs128 实测从 55.6 升到 74.8。相加口径 + B 复用比例能同时解释三个点。
详见 AnalyticalGmmCosts.gmm1_phases 的口径沿革。
"""
import moe_cost_model as m

K, COLS = 5120, 256
#: (m, 每专家 m-group 数, 实测每 tile us)
MEASURED = {
    "bs36": (72, 1, 55.645),
    "bs128": (256, 1, 74.810),
    "bs8192": (256, 12, 53.810),
}
#: 由 bs128 与 bs8192 反解的 B 复用比例 (一个点, 不是规律)
REUSE_FRAC = 0.53


def _b_share(groups: int, frac: float = REUSE_FRAC) -> float:
    """每 tile 平均付多少份 B: 首个 m-group 整份, 其余各 frac."""
    return 1.0 if groups == 1 else (1 + (groups - 1) * frac) / groups


def _tile(mode, rows, groups):
    return m.AnalyticalGmmCosts(load_overlap=mode).gmm1_tile(
        rows, K, COLS, _b_share(groups))


def test_default_is_additive():
    assert m.AnalyticalGmmCosts().load_overlap == "sum"


def test_unknown_mode_is_refused():
    import pytest
    with pytest.raises(ValueError, match="load_overlap"):
        m.AnalyticalGmmCosts(load_overlap="overlap")


def test_additive_fits_all_three_measured_points_within_4pct():
    for tag, (rows, groups, meas) in MEASURED.items():
        got = _tile("sum", rows, groups)
        assert abs(got - meas) / meas < 0.04, (tag, got, meas)


def test_max_cannot_fit_them():
    """max 在三个点上分别低估 9.2% / 32.5% / 46.6% —— 记录它错在哪, 不只是"不用它"."""
    errs = {tag: (_tile("max", rows, g) - meas) / meas
            for tag, (rows, g, meas) in MEASURED.items()}
    assert errs["bs36"] < -0.05
    assert errs["bs128"] < -0.30
    assert errs["bs8192"] < -0.40


def test_max_makes_the_tile_independent_of_m():
    """max 口径的要害: B 流恒大时时长完全不随 m 变, 而实测 bs36->bs128 涨了 34%."""
    flat = {_tile("max", rows, 1) for rows in (72, 128, 256)}
    assert len(flat) == 1
    grows = [_tile("sum", rows, 1) for rows in (72, 128, 256)]
    assert grows[0] < grows[1] < grows[2]
    # 实测斜率 (全部 tile 的 median): (74.810 - 55.645) / (256 - 72) = 0.1041 us/行
    # 模型斜率 = K / BW_L1_GM = 5120 / 51900 = 0.09865 —— 低 5.2%, 即这条斜率反解出的
    # 带宽是 49.2 GB/s 而不是 51.9。
    #
    # 为什么不就把 BW_L1_GM 改成 49.2: 按波分开看, 斜率跟并发核数有关 ——
    #   w0 (28 核并发): (77.92 - 57.12) / 184 = 0.1130 -> 45.3 GB/s
    #   w1 (18 核并发): (55.20 - 29.73) / 184 = 0.1384 -> 37.0 GB/s
    # 一个常数满足不了两档并发。这条斜率是目前最干净的带宽标定手柄 (几何、并发都固定,
    # 只有 m 变), 但它说明 BW_L1_GM 不是一个常数 —— 要定它得扫并发数。
    model_slope = (grows[2] - grows[0]) / (256 - 72)
    assert abs(model_slope - 0.1041) / 0.1041 < 0.06


def test_b_reuse_frac_is_a_fraction_not_a_switch():
    """复用是"付多少比例", 不是"付/不付": 全复用会把 12 组的 tile 算得太便宜."""
    full_reuse = m.AnalyticalGmmCosts().gmm1_tile(256, K, COLS, _b_share(12, 0.0))
    measured = MEASURED["bs8192"][2]
    assert full_reuse < measured * 0.75          # 全复用严重低估
    assert m.KernelConfig().gmm1_b_reuse_frac == 1.0      # 缺省不声称有复用
