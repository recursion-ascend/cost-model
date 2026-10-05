"""未标定输入 -> 结论的区间. 不能判定的比较必须说出来, 不能报点值."""
import pytest

from moe_cost_model.analysis.sensitivity import (RANGED, UNCERTAIN_INPUTS,
                                                 UNKNOWNS, Interval, Ranged,
                                                 Unknown, propagate, report)


def test_ranged_requires_nominal_inside_and_a_source():
    Ranged(name="x", nominal=5.0, low=1.0, high=9.0, source="实测")
    with pytest.raises(ValueError, match="不在"):
        Ranged(name="x", nominal=99.0, low=1.0, high=9.0, source="实测")
    with pytest.raises(ValueError, match="出处"):
        Ranged(name="x", nominal=5.0, low=1.0, high=9.0, source="")


def test_every_registered_input_says_where_its_range_came_from():
    """区间没有出处就不可信; 没有区间的必须说明为什么没有、怎么定."""
    assert UNCERTAIN_INPUTS, "这张表空了 = 声称模型全部标定到点, 那不是事实"
    for r in RANGED:
        assert r.source and len(r.source) > 10
    for u in UNKNOWNS:
        assert u.reason and len(u.reason) > 10


def test_unknown_inputs_are_not_given_a_fabricated_range():
    """连范围都没有的输入**不能**参与区间传播 —— 那是把无知包装成精度."""
    names = {u.name for u in UNKNOWNS}
    assert names, "至少 late_bind_fetch_us 是没测过的"
    assert not (names & {r.name for r in RANGED}), "同一个输入不能既有区间又无区间"


def test_propagate_widens_the_interval_around_the_nominal():
    r = Ranged(name="k", nominal=2.0, low=1.0, high=4.0, source="实测区间")

    def metric(shifts):
        return 10.0 * shifts.get("k", 2.0)        # 单调

    iv = propagate(metric, ranged=(r,), unknowns=())
    assert iv.nominal == pytest.approx(20.0)
    assert iv.low == pytest.approx(10.0)
    assert iv.high == pytest.approx(40.0)
    assert iv.driver == "k"


def test_interval_that_straddles_zero_is_not_decidable():
    r = Ranged(name="k", nominal=1.0, low=0.0, high=2.0, source="实测区间")

    def metric(shifts):
        return shifts.get("k", 1.0) - 1.0         # 在区间内变号

    iv = propagate(metric, ranged=(r,), unknowns=())
    assert iv.straddles_zero and not iv.decidable
    assert "不可判定" in iv.format()


def test_depending_on_an_unmeasured_input_makes_it_undecidable_too():
    """符号确定也不够: 结论若依赖一个连范围都没有的输入, 区间本身不完整."""
    u = Unknown(name="calibration.late_bind_fetch_us", reason="从未测过", calibration="R7")
    iv = propagate(lambda shifts: -1.34, ranged=(), unknowns=(u,),
                   depends_on_unknown=("calibration.late_bind_fetch_us",))
    assert not iv.straddles_zero          # 符号是确定的
    assert not iv.decidable               # 但仍然不可判定
    assert "依赖未测量" in iv.format()


def test_report_lists_both_kinds():
    txt = report()
    assert "有区间" in txt and "无区间" in txt
    assert "bw_combine_remote" in txt
    assert "late_bind_fetch_us" in txt
