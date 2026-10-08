"""出处分类: 每个常数要能回答"这个数是谁定的".

原来的 kernel: 一个标签同时盖着三类东西 —— 硬件容量、算法定义、某实现的取值。
读到一个 kernel: 常数时就分不出"物理上只能这样"还是"那份实现这么选的", 这正是把实现
取值当成"应该的值"的来源 (本仓两次踩过)。现在拆成 spec / algo / impl。
"""
import pytest

from moe_cost_model.config import hardware as H
from moe_cost_model.config import platform as P
from moe_cost_model.config.provenance import (CATEGORIES, SourcedInt,
                                              SourcedValue,
                                              unknown_categories)


def _constants(mod):
    out = {}
    for name in dir(mod):
        v = getattr(mod, name)
        if isinstance(v, (SourcedValue, SourcedInt)):
            out[name] = (float(v), v.source)
    return out


def test_no_constant_uses_the_retired_kernel_category():
    """kernel: 已废 —— 它不回答"谁定的"."""
    bad = {n: s for n, (_v, s) in _constants(H).items() if s.startswith("kernel:")}
    assert not bad, f"还有 kernel: 标签: {sorted(bad)}"


def test_every_constant_has_a_known_category():
    for mod in (H, P):
        assert unknown_categories(_constants(mod)) == {}


def test_hardware_capacities_are_spec_not_impl():
    """容量/位宽/格式标准换一份实现不会变 -> spec."""
    for name in ("TOTAL_L1_SIZE", "TOTAL_UB_SIZE", "TOTAL_L0C_SIZE", "VEC_REG_WIDTH",
                 "MXFP_DIVISOR_SIZE", "MXFP_MULTI_BASE_SIZE"):
        assert getattr(H, name).source.startswith("spec:"), name


def test_algorithm_definitions_are_algo():
    """SwiGLU 的 gate+up 两投影是算法定义, 换实现不变, 换算法才变."""
    assert H.ACTIVATION_N_HALF.source.startswith("algo:")


def test_implementation_choices_are_impl():
    """tile 几何、缓冲槽数、档位阈值、元数据字节 —— 都是某份实现选的."""
    for name in ("TILE_M", "TILE_N", "L1_TILE_K", "DISPATCH_BUFFER_COUNT",
                 "LAYERED_META_BYTES_PER_ROW", "GMM2_LAG_MIN_TOKEN_NUM",
                 "URMA_FLAG_WINDOW_TOKENS"):
        assert getattr(H, name).source.startswith("impl:"), name


def test_impl_constants_reachable_as_parameters():
    """impl: 类的数必须能被参数覆盖, 否则"那份实现"就焊死在公式里了.

    这里只覆盖本轮接出来的两个 (原先一个写死在公式、一个是复制出来的派生值);
    其余 impl: 常数的参数路径见 KernelConfig / InstancePolicy / ModelOptions。
    """
    import moe_cost_model as m
    u = m.UrmaMechanisticLatency()
    assert u.meta_bytes_per_row == float(H.LAYERED_META_BYTES_PER_ROW)
    assert u.flag_window_bytes == float(
        int(H.URMA_FLAG_WINDOW_TOKENS) * int(H.URMA_FLAG_BYTES))
    lean = m.UrmaMechanisticLatency(meta_bytes_per_row=12, flag_window_bytes=1024)
    assert lean.flag_poll_us() < u.flag_poll_us()
    # combine 侧同一处理 (KernelConfig 申报)
    assert m.KernelConfig().combine_meta_bytes_per_row == 16
    assert m.MEGAMOE_A8W8.kernel.combine_meta_bytes_per_row == 32


def test_report_separates_the_three():
    from moe_cost_model.config.provenance import provenance_report
    rep = provenance_report(_constants(H))
    for cat in ("spec", "algo", "impl", "measured", "assumed"):
        assert cat in rep["summary"], cat
    assert rep["impl"] and rep["spec"] and rep["algo"]


def test_categories_are_declared_once():
    assert set(CATEGORIES) == {"spec", "algo", "impl", "measured", "derived", "assumed"}
    with pytest.raises(AssertionError):
        assert unknown_categories({"x": (1.0, "kernel:旧标签")}) == {}


def test_model_default_has_zero_epilogue_overheads_profile_carries_them():
    """实测残留不得藏在模型缺省里: 缺省 literal=True 取字面 0, profile 才回落到实测常数.

    2026-10-05: EpilogueOverheads 的 docstring 原先写"缺省沿用实测值, 这样默认结果不变"
    —— 那句描述的是本类自己 (literal=False) 的行为, 却会被读成"模型缺省沿用实测值"。
    这条测试把两个缺省的区别固定, 免得文档再把人 (包括我) 带偏。
    """
    import moe_cost_model as m
    d = m.ModelOptions().epilogue_overheads
    assert d.literal is True, "模型缺省必须按字面取 0 = 不引用任何实现"
    assert (d.counts_export_us, d.core_sync_us, d.rank_sync_us,
            d.output_init_us, d.finalize_us) == (0.0, 0.0, 0.0, 0.0, 0.0)
    p = m.MEGAMOE_A8W8.options.epilogue_overheads
    assert p.literal is False, "profile 要回落到模块实测常数 (复现那份实现)"
    # 回落确实发生: 同一形状下 profile 的尾段比缺省长
    assert float(m.T_CORE_SYNC_BARRIER_US) > 0 and float(m.T_FINALIZE_US) > 0
