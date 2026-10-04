"""硬件规格 (spec) 与实测 (measured) 分开记.

规格是峰值 -> 用它算出的是时间**下界**, 该配效率系数; 实测已含争用, 不该再乘。
本文件钉住两件事: 单位换算 (FLOPS -> MAC) 与聚合带宽上限。
"""
import pytest

import moe_cost_model as m
from moe_cost_model.config.platform import (CUBE_TFLOPS, SPEC_CUBE_CORES,
                                            resolve_platform)


def test_fp8_peak_is_half_the_flops_figure():
    """每核 MAC/µs = 每核 FLOP/µs / 2 —— 一次 MAC 两个 FLOP.

    本模型的计算量分子是 MAC 数 (GMM1 = 2·m·cols·K, 那个 2 是 gate/up 两个投影)。
    把规格的 FLOPS 直接当 MAC/µs 用会让计算时间整整差一倍 —— 仓库里原来的占位值
    2.7e7 就是这么来的 (它既等于 fp8 的 FLOP/µs, 也等于 mxfp4 的 MAC/µs)。
    """
    per_core_flops_per_us = CUBE_TFLOPS["fp8"] * 1e12 / int(SPEC_CUBE_CORES) / 1e6
    assert m.cube_mac_per_us("fp8") == pytest.approx(per_core_flops_per_us / 2)
    assert m.cube_mac_per_us("fp8") == pytest.approx(1.35e7, rel=1e-3)
    assert m.cube_mac_per_us("mxfp4") == pytest.approx(2.7e7, rel=1e-3)


def test_whitepaper_totals_are_self_consistent():
    """Cube 部分 = 合计 - Vector 部分, 且 fp8 = 2x fp16 / mxfp4 = 4x fp16 自洽."""
    from moe_cost_model.config import platform as P
    assert 2 * CUBE_TFLOPS["fp16"] + float(P.VECTOR_TFLOPS_FP16) == pytest.approx(
        float(P.TOTAL_TFLOPS_FP8), abs=2.0)          # 918 vs 919
    assert 4 * CUBE_TFLOPS["fp16"] + float(P.VECTOR_TFLOPS_FP16) == pytest.approx(
        float(P.TOTAL_TFLOPS_MXFP4), abs=3.0)        # 1782 vs 1784


def test_efficiency_scales_and_is_validated():
    assert m.cube_mac_per_us("fp8", 0.5) == pytest.approx(m.cube_mac_per_us("fp8") / 2)
    with pytest.raises(ValueError):
        m.cube_mac_per_us("fp8", 0.0)
    with pytest.raises(ValueError, match="未知 dtype"):
        m.cube_mac_per_us("fp6")


def test_aggregate_hbm_caps_the_per_core_bandwidth():
    """单核带宽 x 活跃核数 不得超过聚合 HBM —— 一个常数在核数多时会突破物理上界."""
    per_core = 51900.0
    pr, dt = m.ASCEND_950PR, m.ASCEND_950DT
    # 28 核 (本卡真实可用核数): 两档都不受限
    assert pr.gm_bw_per_core(28, per_core) == per_core
    assert dt.gm_bw_per_core(28, per_core) == per_core
    # 但 950PR 在 28 核已经吃掉聚合的 91%, 32 核就超了
    assert 0.90 < pr.hbm_utilisation(28, per_core) < 0.92
    assert pr.hbm_utilisation(32, per_core) > 1.0
    assert pr.gm_bw_per_core(32, per_core) < per_core          # 被压到 50000
    # 950DT 的聚合是 2.5 倍, 同样核数下只占 36%
    assert dt.hbm_utilisation(28, per_core) < 0.40


def test_cost_factory_applies_the_cap():
    def tile(platform, cores):
        c = m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(),
            cube_mac_per_us=m.cube_mac_per_us("fp8"),
            platform=platform, active_cores=cores)
        return c.gmm1_tile(256, 5120, 256)
    # 950PR 32 核: 带宽被聚合压低 -> tile 变慢; 950DT 不受限
    assert tile(m.ASCEND_950PR, 32) > tile(m.ASCEND_950DT, 32)
    # 28 核时两档相同 (都没撞上限)
    assert tile(m.ASCEND_950PR, 28) == tile(m.ASCEND_950DT, 28)
    # 不给 platform = 不加上限 (向后兼容)
    plain = m.build_analytical_costs(
        h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(),
        cube_mac_per_us=m.cube_mac_per_us("fp8")).gmm1_tile(256, 5120, 256)
    assert plain == tile(m.ASCEND_950DT, 32)


def test_platform_names_resolve():
    assert resolve_platform("950PR") is m.ASCEND_950PR
    assert resolve_platform("Ascend 950DT") is m.ASCEND_950DT
    with pytest.raises(ValueError, match="未知平台"):
        resolve_platform("910B")


def test_spec_is_its_own_provenance_category():
    from moe_cost_model.config import platform as P
    from moe_cost_model.config.provenance import provenance_report
    rep = provenance_report({"fp8": (float(P.TOTAL_TFLOPS_FP8), P.TOTAL_TFLOPS_FP8.source)})
    assert "spec" in rep["summary"] and rep["spec"]
