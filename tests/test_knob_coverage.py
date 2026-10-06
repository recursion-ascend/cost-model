"""旋钮覆盖: 每个可调参数都必须能动模型, 不能动的必须已声明理由.

这条护栏保的是**设计空间探索工具的可信度**。算法工程师改编排 / 编译期 / 运行期参数后,
一个 "0 收益" 有四种意思, 指示完全相反:

  生效    这个形状上就动 -> 结论可信
  生效*   换对形状才动 (只有一波谈不上超前几波, 只有一个 K 块谈不上逐块就绪)
  被拒    模型显式拒绝该取值 (缺标定 / 这条路径没实现) -> 诚实的拒绝
  动不了  模型里没有可表达的后果 -> **陷阱**: 扫出来的 0 会被当成 "硬件上也没收益"

tools/knob_audit.py 是扫描器 (五个互补形状 x 逐旋钮扰动, 比对墙钟 / 事件数 / 事件名 /
逐事件时长 / 逐信道字节), 它的全量判定钉在 knob_audit.EXPECTED 里。全量扫描是分钟级的,
所以本文件跑一个形状, 并用 EXPECTED 做交叉判据:

  * 旋钮树是**自动走**出来的 (dataclasses.fields + scenario._NESTED), 所以新加的字段
    自动进审计。新字段没进 EXPECTED -> 红: 要么接线了 (钉上判定), 要么是死旋钮。
  * 本形状上动了, 而 EXPECTED 说 "动不了" -> 红: 声明过期了, 该删。
  * EXPECTED 说 "生效" (与形状无关) 而本形状不动 -> 红: 接线退化了。

全量扫描要人跑: python tools/knob_audit.py (--emit 重新生成 EXPECTED)。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import knob_audit as ka                                            # noqa: E402
from moe_cost_model import load_scenario                           # noqa: E402

SHAPE = "多波核紧"


@pytest.fixture(scope="module")
def rows():
    scenario = load_scenario(ka.SCENARIO)
    _, out = ka.audit_shape(scenario, ka.SHAPES[SHAPE])
    return out


def test_knob_tree_matches_the_pinned_table(rows):
    """旋钮集合 = EXPECTED 的键集合.

    新加一个旋钮 (或删掉一个) 必须同时更新钉住的判定 —— 这一步就是"有没有人核过
    它到底接没接上线"的那道关。
    """
    assert set(rows) == set(ka.EXPECTED), (
        f"旋钮树与 EXPECTED 不一致: 多出来 {sorted(set(rows) - set(ka.EXPECTED))}, "
        f"少了 {sorted(set(ka.EXPECTED) - set(rows))}; "
        f"跑 python tools/knob_audit.py --emit 重新生成")


def test_every_knob_has_a_candidate_value(rows):
    """每个旋钮都要有可试的取值 —— 没有就等于没被审计过 (审计的盲区不是结论)."""
    skipped = {p: why for p, (kind, why) in rows.items() if kind == "跳过"}
    assert not skipped, f"这些旋钮没有候选取值, 给 CANDIDATES 补上: {skipped}"


def test_shape_independent_knobs_still_move_here(rows):
    """EXPECTED 里标"生效"的是与形状无关的, 在本形状上必须照样动."""
    broke = [p for p, v in ka.EXPECTED.items()
             if v == "生效" and rows[p][0] != "生效"]
    assert not broke, f"这些旋钮本该与形状无关地生效, 现在不动了: {broke}"


def test_declared_dead_knobs_are_still_dead(rows):
    """声明"动不了"的旋钮必须真的动不了 —— 哪天接上了, 声明要删.

    反向守: 声明会过期。combine_layout 一旦接上读侧代价 (缺口 10) 这里就红, 提醒把它
    从 DECLARED_UNREAD / EXPECTED 里改掉, 否则工程师会以为它仍是空白。
    """
    for path, verdict in ka.EXPECTED.items():
        if verdict != "动不了":
            continue
        assert path in ka.DECLARED_UNREAD, f"{path} 判为动不了但没写理由"
        assert rows[path][0] == "无动静", (
            f"{path} 已声明为模型里动不了, 但本形状上它 {rows[path][0]} —— 接上了就改声明")


def test_engine_queue_depth_is_not_a_knob():
    """引擎队列深度已不是旋钮: 这个事件代数里它没有可表达的后果.

    持核事件独占 AIC/AIV0/AIV1, 同核在途数恒 <= 1; 相位拆分后的 load 相位又刻意不继承
    Q:* (继承会让容量 1 的引擎信号量卡死 L1 缓冲深度)。所以任何深度都与 1 逐位相同 ——
    留着旋钮只会让扫描得出"深了也没用"的假结论。
    """
    import dataclasses

    import moe_cost_model as m

    assert not hasattr(m, "EngineQueueDepths")
    assert not any(f.name == "engine_queue_depths"
                   for f in dataclasses.fields(m.ModelOptions))


def test_roles_and_epilogue_are_reachable_from_a_scenario_file():
    """两个编排旋钮必须在**场景文件这条日常路径**上写得出来.

    在日常路径上写不出的旋钮等于没有。2026-10-05 之前 options.roles 与
    options.epilogue_overheads 走 with_overrides 会报"应为数值" —— 只能在 Python 里
    构造对象, 于是"哪个 stage 跑在哪个核上"这一类编排在场景扫描里根本到不了。
    """
    scenario = load_scenario(ka.SCENARIO)
    moved = scenario.with_overrides({"options.roles": {"combine": "AIV0"},
                                     "options.epilogue_overheads": {"literal": True}})
    assert moved.options.roles.role_of("combine") == "AIV0"
    assert moved.options.epilogue_overheads.literal is True


def test_readme_knob_counts_match_the_table():
    """README 不再抄旋钮表, 但它引用了项数 —— 引用也会过期, 所以钉住.

    2026-10-06 之前 README 里是一张**手写副本**: 列着早已删掉的 gmm1_activation_depth,
    把 gmm1_b_reuse 写成"不影响时长", 还少十几项。现在那段只说"权威清单在 knob_audit.EXPECTED,
    共 N 项", 并按类别给项数 —— 本测试核对那几个 N。
    """
    import collections
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert f"(**{len(ka.EXPECTED)}** 项)" in readme or f"({len(ka.EXPECTED)} 项)" in readme, (
        f"README 里的旋钮总数与 EXPECTED ({len(ka.EXPECTED)} 项) 不一致")
    by_cat = collections.Counter(
        k.split(".")[0] if "." in k else "(顶层)" for k in ka.EXPECTED)
    for cat, prefix in (("kernel", "`kernel.*`"), ("options", "`options.*`"),
                        ("policy", "`policy.*`")):
        row = re.search(re.escape(prefix) + r"\s*\|\s*(\d+)\s*\|", readme)
        assert row, f"README 的分类表里没有 {prefix} 这一行"
        assert int(row.group(1)) == by_cat[cat], (
            f"README 说 {prefix} 有 {row.group(1)} 项, 实际 {by_cat[cat]} 项")
    top = re.search(r"顶层策略名\s*\|\s*(\d+)\s*\|", readme)
    assert top and int(top.group(1)) == by_cat["(顶层)"]
