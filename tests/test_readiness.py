"""就绪粒度: 取值的编码、哪条边能分段、段界与单调性.

分三块对应三件事:
  * Readiness 本身是个纯函数 (档 -> 每段吃几个块), 所以编码与段界可以硬断言;
  * 哪条边允许非 "whole" 由 EDGE_AXES 说, 不可分段的边必须**报错**而不是静默忽略;
  * 单调性: 段数一定随"更细"不减; 墙钟只在静态钉核下断言, 晚绑定下只记事实
    (贪心表调度对事件数不单调, 见 test_scheduler 的 Graham 异常先例)。
"""
import pytest

import moe_cost_model as m
from moe_cost_model.config.links import EDGE_AXES, validate_links
from moe_cost_model.config.readiness import (Readiness, chunk_count,
                                             parse_readiness, segment_spans)
from linkutil import links
from golden_cases import run_api, uniform_routing


# --------------------------------------------------------------- 编码: 无哨兵

@pytest.mark.parametrize("wrote, want", [
    ("whole", Readiness.whole()),
    ("per_chunk", Readiness.per_chunk()),
    ("first_chunk", Readiness.first_chunk()),
    (2, Readiness.n(2)),
    (7, Readiness.n(7)),
])
def test_every_spelling_parses_and_round_trips(wrote, want):
    got = parse_readiness(wrote)
    assert got == want
    assert got.spelling == wrote          # to_dict() 要喂得回 load_scenario
    assert parse_readiness(got) is got


@pytest.mark.parametrize("bad, fragment", [
    (0, 'per_chunk'),            # 旧的"最细"
    (1, 'whole'),                # 旧的"等齐"
    (-1, "不能为负"),
    (True, "布尔"),
    (2.0, "应为"),
    ("finest", "不认识"),
    ("even", "不认识"),          # 均分要给段数, 光说"均分"不是取值
])
def test_ambiguous_values_are_refused(bad, fragment):
    """0 与 1 都报错: 它们在旧编码里分别是"最细"与"等齐", 不单调, 而且 0 的含义
    与 granularity 的 0=整片、dispatch 的 0=沿用 tiling 三处相反。"""
    with pytest.raises(ValueError, match=fragment):
        parse_readiness(bad)


def test_even_needs_at_least_two_segments():
    """"均分 1 段"就是 whole —— 不留两种写法."""
    with pytest.raises(ValueError, match="至少 2 段"):
        Readiness.n(1)
    with pytest.raises(ValueError, match="不带段数"):
        Readiness("whole", 3)


# --------------------------------------------------------------- 段界与余数

def test_segments_land_on_chunk_boundaries():
    """段界一律落在自然块边界上: 半块就绪在硬件上没有对应物."""
    for extent, chunk in ((4608, 256), (4608, 512), (5120, 256), (300, 256)):
        n = chunk_count(extent, chunk)
        for spell in ("whole", "per_chunk", "first_chunk", 2, 3, 5, 99):
            spans = segment_spans(parse_readiness(spell), extent, chunk)
            assert spans[0][0] == 0 and spans[-1][1] == extent
            for (lo, hi), (nxt, _) in zip(spans, spans[1:]):
                assert hi == nxt                      # 无缺口无重叠
            for lo, hi in spans:
                assert lo % chunk == 0                # 起点在块界上
                assert hi % chunk == 0 or hi == extent  # 末块可以不满
            assert len(spans) <= n


def test_remainder_goes_to_the_front_segments():
    """除不尽时余数给**前面**的段 —— 写死, 不是隐式行为.

    9 块分 4 段 = 3,2,2,2 (不是 2,2,2,3)。
    """
    assert Readiness.n(4).chunk_counts(9) == (3, 2, 2, 2)
    assert Readiness.n(4).chunk_counts(8) == (2, 2, 2, 2)
    assert Readiness.n(2).chunk_counts(5) == (3, 2)
    assert Readiness.n(3).chunk_counts(4) == (2, 1, 1)


def test_more_segments_than_chunks_is_per_chunk():
    """块是最小单位: 要的段数超过块数就是每块一段 (不是报错, 也不是空段)."""
    assert Readiness.n(99).chunk_counts(4) == (1, 1, 1, 1)
    assert Readiness.n(99).segment_count(4) == Readiness.per_chunk().segment_count(4)
    assert Readiness.first_chunk().chunk_counts(1) == (1,)   # 只有一块: 退化成等齐


@pytest.mark.parametrize("n_chunks", [1, 2, 3, 4, 9, 18, 64])
def test_segment_count_is_monotone_in_fineness(n_chunks):
    """"更细"就是"段数更多", 单调 —— 旧编码 (1/2/0/>=n) 做不到这一点."""
    counts = [parse_readiness(s).segment_count(n_chunks)
              for s in ("whole", 2, 3, 4, 8, "per_chunk")]
    assert counts == sorted(counts)
    assert counts[0] == 1 and counts[-1] == n_chunks
    for s in ("whole", 2, 3, 4, 8, "per_chunk", "first_chunk"):
        r = parse_readiness(s)
        assert sum(r.chunk_counts(n_chunks)) == n_chunks


# --------------------------------------------------------- 哪条边能分段

def test_every_declared_edge_says_who_consumes_it():
    """可分段 = 有人读。EDGE_AXES 里 segmentable 的边必须写出消费者,
    否则就是"写得出但没人看"(SharedAxis.__post_init__ 也拦)。"""
    for (producer, consumer), axis in EDGE_AXES.items():
        assert axis.axis and axis.chunk and axis.note, (producer, consumer)
        if axis.segmentable:
            assert axis.consumed_by, (producer, consumer)
        else:
            assert not axis.consumed_by, (producer, consumer)
    assert EDGE_AXES[("activation", "gmm2")].segmentable
    assert sum(1 for a in EDGE_AXES.values() if a.segmentable) == 1


@pytest.mark.parametrize("producer, consumer", [
    ("dispatch", "gmm1"), ("gmm1", "activation"), ("gmm2", "combine")])
def test_non_whole_readiness_on_an_unsegmentable_edge_is_refused(producer, consumer):
    """这是本次改动的要点: 在没有消费者的边上写非缺省值, 原先**静默无效**
    (算子工程师在那儿扫一圈得到"0 收益", 还以为硬件上也没收益)。"""
    with pytest.raises(ValueError, match="readiness"):
        m.ModelOptions(links=(m.StageLink(producer, consumer, readiness="per_chunk"),))


def test_gmm2_combine_points_at_granularity():
    """这条边的共享轴就是 combine 的打包单元, 报错要指向正确的参数."""
    with pytest.raises(ValueError, match=r'granularity\["combine"\]'):
        m.ModelOptions(links=(m.StageLink("gmm2", "combine", readiness=4),))


def test_unknown_edge_is_refused():
    """模型只有这几条边; 别的边名多半是拼错, 而在那儿写什么都不会被读."""
    with pytest.raises(ValueError, match="模型没有的边"):
        m.ModelOptions(links=(m.StageLink("activation", "gmm1"),))


def test_segment_sync_us_needs_a_segmentable_edge():
    with pytest.raises(ValueError, match="segment_sync_us 没有作用对象"):
        validate_links((m.StageLink("gmm1", "activation", segment_sync_us=0.1),))
    with pytest.raises(ValueError, match="不能为负"):
        m.StageLink("activation", "gmm2", segment_sync_us=-1.0)


def test_gmm1_act_link_under_prefetch_keeps_readiness_whole():
    """prefetch 改的是落点/深度/共位, 不是就绪 —— 而这条边上就绪只能是 whole,
    所以"照抄"不会把一个没人读的取值带进图里 (links.effective_gmm1_act_link)。"""
    opts = m.ModelOptions()
    link = opts.gmm1_act_link(m.KernelConfig(topk_weights_prefetch=True))
    assert link.location == "gm" and link.depth == 0
    assert not link.colocated_by_hardware
    assert link.readiness.is_whole


# ------------------------------------------------------------- 接到图上

def _run(readiness, late=(), **kw):
    return run_api(uniform_routing(2, 4, 128), 128, topk=8, aic_num=28,
                   options=m.ModelOptions(late_bind_pools=late,
                                          links=links(readiness=readiness), **kw))


def _gmm2(res):
    return [e for e in res["rank_results"][0]["events"]
            if e.meta.get("stage") == "gmm2"]


FINER = ("whole", 2, 4, "per_chunk")


def _built(readiness):
    """排程前的事件图 —— 依赖边只在这里看得到 (ScheduledEvent 不带 deps)."""
    from moe_cost_model.model import A8W8WaveCostModel
    from moe_cost_model.shape import MegaMoeShape
    from golden_cases import H, HIDDEN, manual_costs
    rows = uniform_routing(2, 4, 128)[0]
    shape = MegaMoeShape(
        expert_tokens=tuple(sum(r) for r in rows), token_num=128, h=H,
        hidden_dim=HIDDEN, aic_num=28, expert_source_tokens=rows,
        p1_override=2, p2_override=1, topk=8, kernel=m.KernelConfig())
    events, _ = A8W8WaveCostModel(
        manual_costs(),
        m.ModelOptions(links=links(readiness=readiness))).build_events(shape)
    return events


def test_finer_readiness_never_makes_the_first_segment_wait_for_more():
    """结构断言 (与调度无关): 越细, 每个 tile 的**首段**等的 ACT 越少 —— 这就是
    "开工更早"在图上的样子。首段吃的块数 = ceil(块数/段数), 对段数不增。"""
    waits = []
    for spell in FINER:
        g2 = [e for e in _built(spell) if (e.meta or {}).get("stage") == "gmm2"]
        firsts = [e for e in g2 if (e.meta or {}).get("part") in ("head", "k0")] or g2
        waits.append(max(len([d for d in e.deps if ".act." in d]) for e in firsts))
    assert waits == sorted(waits, reverse=True), waits
    assert waits[0] > waits[-1], waits        # 最粗与最细必须真的不同


def test_static_pinning_wall_clock_does_not_increase_with_finer_readiness():
    """静态钉核下墙钟不增.

    本夹具上四个档**完全相等** (289.065us): 静态分核的墙钟由各核的工作量定,
    GMM2 的等待不在关键路径上, 所以分段既不赚也不亏。断言取"不增"而不是"严格
    下降" —— 严格下降不是这个机制的承诺。
    """
    totals = [_run(s)["kernel_total_us"] for s in FINER]
    for a, b in zip(totals, totals[1:]):
        assert b <= a * (1 + 1e-9), totals


def test_late_binding_is_not_monotone_in_readiness():
    """晚绑定下墙钟对段数**不单调**, 这是记录下来的事实而不是容差:
    本夹具上均分 3 段 (250.511us) 比均分 2 段 (247.986us) 更差, 而 4 段又回到
    247.986。成因是贪心表调度对事件集合的 Graham 异常 (test_scheduler 已有先例),
    不是分段本身有代价 —— 分段的代价是 segment_sync_us, 缺省 0 未标定。

    所以: 用 readiness 扫出来的几个百分点, 只有在同一绑定方式下、且差值大于这类
    抖动时才可读。
    """
    late = ("AIC", "AIV1")
    two, three, four = (_run(s, late)["kernel_total_us"] for s in (2, 3, 4))
    assert three > two and four < three, (
        f"2 段 {two:.3f} / 3 段 {three:.3f} / 4 段 {four:.3f} —— 如果现在单调了, "
        "那是调度器变了 (比如缺省策略换成 WorkConservingCriticalPath): "
        "改这个测试, 同时更新 docs/design_space_gaps.md 里那张表")


def test_segment_sync_us_charges_the_extra_segments():
    """分段的代价要能表达出来: 给了 segment_sync_us, 细分段就不再免费.

    每个 tile 多 (段数-1) 段, 所以总时长增量 = tile 数 x (段数-1) x 开销。
    """
    free = _run("per_chunk")
    charged = run_api(
        uniform_routing(2, 4, 128), 128, topk=8, aic_num=28,
        options=m.ModelOptions(late_bind_pools=(), links=links(
            readiness="per_chunk", segment_sync_us=0.05)))
    n_seg = len(_gmm2(free))
    n_tile = len(_gmm2(_run("whole")))
    assert n_seg > n_tile
    busy = sum(e.end_us - e.start_us for e in _gmm2(charged))
    busy0 = sum(e.end_us - e.start_us for e in _gmm2(free))
    assert busy == pytest.approx(busy0 + (n_seg - n_tile) * 0.05, rel=1e-6)
