"""COMBINE 建模核对 (2026-10-04): 物理对不对, 以及表达得出哪些编排.

参考实现是**验证点**, 不是标准 —— 所以这里分两类断言:
  物理/算法: 读回的是 GM 上的 GMM2 输出 (BF16), 每行写到它来源卡的窗口, 本卡与跨卡
            两种带宽。这些不随编排变。
  编排选择: 每行搬几个路由字段 (combine_meta_bytes_per_row)、写侧落点布局、combine
            跑在哪个角色/什么粒度。前者已是申报参数; 后两者是缺口 10 / 11。
"""
import pytest

import moe_cost_model as m

#: 算法下界: combine 只需 route 三项 (dstRankId / tokenIdx / topkIdx)
ALGO_MIN_META_BYTES = 12
#: 某实现的取值: 搬满 META_INFO_SIZE=8 个 int32 槽
ONE_IMPL_META_BYTES = 32
#: 20260930 run 的 COMBINE 单 tile 中位时长 (rank0, n=256, 3/4 行跨卡)
MEASURED = {72: 5.553, 256: 36.751}


def test_meta_bytes_is_declared_not_hardcoded():
    """每行搬几个字段是编排选择: 缺省 16B, 实现可声明 32B, 算法下界 12B."""
    assert m.KernelConfig().combine_meta_bytes_per_row == 16
    assert m.MEGAMOE_A8W8.kernel.combine_meta_bytes_per_row == ONE_IMPL_META_BYTES
    lean = m.AnalyticalCombineCosts(meta_bytes_per_row=ALGO_MIN_META_BYTES)
    fat = m.AnalyticalCombineCosts(meta_bytes_per_row=ONE_IMPL_META_BYTES)
    assert lean.tile(256, 256, 192) < fat.tile(256, 256, 192)


def test_kernel_declaration_reaches_the_formula():
    """KernelConfig 的申报要真的进到公式里 (否则这个旋钮是装饰)."""
    def tile(meta):
        c = m.build_analytical_costs(
            h=5120, kernel=m.KernelConfig(combine_meta_bytes_per_row=meta),
            dispatch_mechanistic=m.DispatchMechanisticLatency())
        return c.combine_tile(256, 256, 192)
    assert tile(32) > tile(16) > tile(12)


def test_read_term_is_tile_bytes_plus_meta():
    C = m.AnalyticalCombineCosts()
    m_rows, n = 256, 256
    want = m_rows * (2.0 * n + C.meta_bytes) / C.bw_local      # BF16 tile + 路由元数据
    assert C.read_us(m_rows, n) == pytest.approx(want)


def test_write_splits_local_and_remote_by_destination_rank():
    """每行写到 route.dstRankId 的窗口 -> 本卡行走本卡带宽, 跨卡行走片间带宽."""
    C = m.AnalyticalCombineCosts()
    m_rows, n = 256, 256
    row = C.write_bytes_per_row(n)
    all_local = C.tile(m_rows, n, 0)
    all_remote = C.tile(m_rows, n, m_rows)
    assert all_remote > all_local
    assert (all_remote - all_local) == pytest.approx(
        m_rows * row * (1 / C.bw_remote - 1 / C.bw_local))


def test_no_quant_writes_bf16_quant_writes_fp8_plus_scale():
    assert m.AnalyticalCombineCosts(combine_quant_mode=0).write_bytes_per_elem == 2.0
    q = m.AnalyticalCombineCosts(combine_quant_mode=1)
    assert q.write_bytes_per_elem == pytest.approx(1 + 1 / 32)


def test_model_is_optimistic_and_misses_the_superlinear_scaling():
    """记录缺口 10: 实测对 m 超线性, 而按字节/按每行固定开销都是线性的.

    实测 m 比 3.56x -> 时长比 6.62x; 模型两个口径都给线性。所以这不是调带宽能修的,
    缺的是"写落点跨度"这个量 (见 docs/design_space_gaps.md 缺口 10)。
    """
    C = m.AnalyticalCombineCosts()
    model = {rows: C.tile(rows, 256, rows * 3 // 4) for rows in MEASURED}
    # 模型在两个点上都偏快, 且大 m 上更偏
    ratio = {rows: MEASURED[rows] / model[rows] for rows in MEASURED}
    assert ratio[72] > 4.0
    assert ratio[256] > ratio[72]            # 偏差随 m 变大 = 超线性没被建模
    # 模型自己是线性的: 时长比应当接近字节比
    assert model[256] / model[72] == pytest.approx(256 / 72, rel=0.02)
    assert MEASURED[256] / MEASURED[72] > 1.5 * (256 / 72)


def test_only_one_combine_orchestration_is_expressible():
    """记录缺口 11: 换数据格式不该换编排, 而模型只有一种编排.

    combine 现在恒是"AIV1 与 GMM2 tile 1:1 配对同核"; 另一种 (放另一个向量角色、
    逐专家独立一遍、挂在整波之后) 表达不出来。所以 combine_quant_mode 只改字节,
    事件图不变 —— 这是**对的**(格式与编排无因果), 缺的是那个独立的编排旋钮。
    """
    W, PER, LOCAL = 4, 18, 3
    rc = [[[PER] * W for _ in range(LOCAL)] for _ in range(W)]

    def run(mode):
        return m.simulate_routing_counts(
            routing_counts=rc, token_num_per_rank=36, h=5120, hidden_dim=9216,
            aic_num=28, topk=6, p1_override=1, p2_override=1,
            kernel=m.KernelConfig(combine_quant_mode=mode),
            costs=m.build_analytical_costs(
                h=5120, kernel=m.KernelConfig(combine_quant_mode=mode),
                dispatch_mechanistic=m.DispatchMechanisticLatency()),
        )["rank_results"][0]["events"]

    def combines(evs):
        return [e for e in evs if e.meta.get("stage") == "combine"]

    a, b = run(0), run(1)
    assert len(combines(a)) == len(combines(b))            # 换格式不动结构
    assert all(e.resources[0].startswith("R0.AIV1") for e in combines(b))
    # 字节宽度确实换了 -> 每个 combine 事件更快
    assert sum(e.end_us - e.start_us for e in combines(b)) < \
        sum(e.end_us - e.start_us for e in combines(a))
