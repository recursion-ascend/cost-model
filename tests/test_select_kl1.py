"""kL1 自适应选择: 移植自 kernel 的 CalcAdaptiveL1Params, 两件事不能混成一件.

kernel 的 `CalcAdaptiveL1Params` (common/mega_moe_gmm_common.h) 里:
  * `constexpr uint64_t maxKL1Units = 2U` —— **固定**倍数, 问的是"两个基础 K 窗放不放得下";
  * `MEGAMOE_L1_BUF_NUM` (l1BufNum) 在那三个容量比较式里**一次都没出现**, 它只进
    `L1Params{.l1BufNum = ...}`, 管 ping-pong 的缓冲块数。

2026-10-06 之前 Python 把 `KernelConfig.l1_buf_num` 当成那个倍数传进容量判据
(`select_kl1(n_windows=km.l1_buf_num)`)。缺省值恰好都是 2, 所以缺省配置下两种写法同值 ——
这也是它一直没被 golden 抓到的原因 (唯一一个 l1_buf_num=1 的 case 走的是提前返回那条分支)。
一旦扫这个旋钮, 旧写法会改 kL1 而 kernel 不会: 模型里那个旋钮多了一份 kernel 没有的后果。
"""
import pytest

from moe_cost_model.config.hardware import (MAX_KL1_UNITS, MXFP_DIVISOR_SIZE,
                                            MXFP_MULTI_BASE_SIZE, SCALE_TRANSFER_BYTES,
                                            TOTAL_L1_SIZE, select_kl1)

#: 能走到容量判据的形状: 部分 tile (m < tile_m) 且 K > 基线
PARTIAL_ROWS, BIG_K = 128, 512


def _capacity_fits(units: int, m_rows: int = PARTIAL_ROWS, base: int = 256,
                   tile_n: int = 256, l1_size: int = TOTAL_L1_SIZE) -> bool:
    """逐字复算 kernel 的三个比较式, 用来证明测试里的断言不是凭空写的."""
    block_m = ((m_rows + 15) // 16) * 16
    data_per_unit = block_m * base + tile_n * base
    scale_k = ((base + MXFP_DIVISOR_SIZE - 1) // MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE
    scale_a, scale_b = block_m * scale_k, tile_n * scale_k
    return (units * data_per_unit + units * (scale_a + scale_b) <= l1_size // 2
            and units * scale_a <= SCALE_TRANSFER_BYTES
            and units * scale_b <= SCALE_TRANSFER_BYTES)


def test_the_doubling_factor_is_fixed_at_the_kernels_value():
    assert MAX_KL1_UNITS == 2, "kernel 的 maxKL1Units 是固定的 2U"


def test_select_kl1_no_longer_takes_a_window_count():
    """n_windows 参数已删 —— 留着它就会有人再把 l1_buf_num 传进来."""
    import inspect
    assert "n_windows" not in inspect.signature(select_kl1).parameters


def test_early_returns_match_the_kernels_guards():
    """整 tile 或 K<=基线 直接用基线 (kernel 同样先过这两道门)."""
    assert select_kl1(256, BIG_K) == 256          # m >= tile_m: 整 tile
    assert select_kl1(0, BIG_K) == 256            # 空 tile
    assert select_kl1(PARTIAL_ROWS, 256) == 256   # K <= 基线
    assert select_kl1(PARTIAL_ROWS, BIG_K, override=128) == 128   # 显式覆盖优先


def test_partial_tile_doubles_when_two_k_windows_fit():
    """部分 tile 且两个 K 窗放得下 -> kL1 翻倍. 判据与 kernel 逐字一致."""
    assert _capacity_fits(MAX_KL1_UNITS)
    assert select_kl1(PARTIAL_ROWS, BIG_K) == 512


def test_a_smaller_l1_falls_back_to_the_base():
    """半片 L1 装不下两个窗就退回基线 —— 容量判据真的在起作用."""
    assert not _capacity_fits(MAX_KL1_UNITS, l1_size=256 * 1024)
    assert select_kl1(PARTIAL_ROWS, BIG_K, l1_size=256 * 1024) == 256


def test_the_buffer_count_no_longer_changes_kl1():
    """这是修掉的那个分歧: l1_buf_num=3 时旧写法给 256, kernel 给 512.

    算术: m=128 -> blockM=128, base=256, tile_n=256
      每个窗 = 128*256 + 256*256 = 98304 B 数据 + 3072 B scale
      units=2 -> 202752 <= 半片 L1 262144 -> 翻倍 (kernel 的答案)
      units=3 -> 304128 >  262144      -> 不翻倍 (旧 Python 的答案)
    现在 select_kl1 不再接受窗数, 所以这个分歧在接口上就不可能再出现;
    这里把两个数都算出来, 证明当年那个差异是真的, 不是理论上的。
    """
    assert _capacity_fits(2) and not _capacity_fits(3)
    assert select_kl1(PARTIAL_ROWS, BIG_K) == 512


def test_l1_buf_num_still_has_its_real_effect_elsewhere():
    """l1_buf_num 仍然有后果, 只是在别处: 单缓冲 (=1) 让 GMM 走串行换块.

    把它从 kL1 判据里拿掉, 不等于把这个旋钮变成装饰 —— 它在 AnalyticalGmmCosts 里控制
    serial, 那才是 kernel 里 l1BufNum 真正管的事。
    """
    import moe_cost_model as m

    single = m.AnalyticalGmmCosts(l1_buf_num=1, cube_mac_per_us=2.7e7)
    double = m.AnalyticalGmmCosts(l1_buf_num=2, cube_mac_per_us=2.7e7)
    assert single.serial and not double.serial
    assert single.gmm1_tile(256, 1024, 256) != double.gmm1_tile(256, 1024, 256)
