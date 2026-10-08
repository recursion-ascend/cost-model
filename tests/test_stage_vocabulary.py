"""stage 词汇表是核心与具体实现的边界.

两件事要有测试:
  1. **唯一出处**: 粒度 / 边 / 角色 / 必经链 / 搬运组 / 比对口径 / 排空归集 这七处
     读的是同一份声明 —— 原先它们各写一份字面量, 改一处对不上另一处。
  2. **另一个算子能用**: 换一份 StageVocabulary, 上面那些校验与缺省表照新的办,
     不必改 config/ 与 analysis/ 的代码。这是"框架化"这一步的判据。
"""
import pytest

from moe_cost_model.analysis.stealing import ALL_STEAL_GROUPS, DEFAULT_STEAL_GROUPS
from moe_cost_model.builders.base import EventBuilderBase
from moe_cost_model.config.granularity import (DEFAULT_GRANULARITY, STAGES, UNIT_OF,
                                               GranularityAssignment, StageGranularity,
                                               resolve_granularity)
from moe_cost_model.config.links import EDGE_AXES, StageLink, validate_links
from moe_cost_model.config.roles import (CUBE_ONLY_STAGES, DEFAULT_STAGE_ROLES,
                                         RoleAssignment)
from moe_cost_model.config.stages import (AIC, AIV0, AIV1, SharedAxis, StageVocabulary,
                                          default_vocabulary)
from moe_cost_model.implementations.megamoe import A8W8WaveV1, LayeredV1
from moe_cost_model.implementations.megamoe_stages import MEGAMOE
from moe_cost_model.validation.compare import COMPARED_STAGES


# ---- 1. 唯一出处 ----

def test_the_seven_sites_read_the_same_declaration():
    """七处缺省表/校验表都来自词汇表, 没有第二份字面量."""
    assert STAGES == MEGAMOE.pipeline
    assert UNIT_OF == MEGAMOE.unit_of
    assert EDGE_AXES == MEGAMOE.edges
    assert CUBE_ONLY_STAGES == MEGAMOE.cube_only
    assert DEFAULT_STAGE_ROLES == MEGAMOE.roles
    assert COMPARED_STAGES == MEGAMOE.compared
    assert EventBuilderBase.DRAIN_STAGES == MEGAMOE.drain
    assert ALL_STEAL_GROUPS == MEGAMOE.steal_groups
    assert DEFAULT_STEAL_GROUPS == MEGAMOE.default_steal_groups


def test_the_adapters_declare_the_vocabulary():
    """"哪些 stage"属于实现, 所以适配器要说得出来 (ImplementationAdapter.stages)."""
    for cls in (A8W8WaveV1, LayeredV1):
        assert cls().stages() is MEGAMOE
    assert default_vocabulary() is MEGAMOE


def test_megamoe_declaration_is_unchanged():
    """这一步只归位不改行为: MegaMoE 声明的还是原先那五个 stage 与四条边."""
    assert MEGAMOE.pipeline == ("dispatch", "gmm1", "activation", "gmm2", "combine")
    assert MEGAMOE.chain == MEGAMOE.pipeline
    assert set(MEGAMOE.edges) == {("dispatch", "gmm1"), ("gmm1", "activation"),
                                  ("activation", "gmm2"), ("gmm2", "combine")}
    assert MEGAMOE.items_default("dispatch") == 0
    assert all(g.items_per_event == 1 for g in DEFAULT_GRANULARITY
               if g.stage != "dispatch")
    assert MEGAMOE.default_steal_drivers == ("gmm1",)


# ---- 2. 另一个算子 ----

#: 一个与 MegaMoE 不同的划法: 三个 stage, 两条边, 多一个权重解压阶段。
#: 只为测"核心不认识 stage 名"而存在, 不声称对应任何真实 kernel。
OTHER = StageVocabulary(
    operator="toy_two_stage",
    pipeline=("load", "unpack", "matmul"),
    unit_of={"load": "row", "unpack": "block", "matmul": "tile"},
    roles={"load": AIV1, "unpack": AIV0, "matmul": AIC},
    cube_only=("matmul",),
    edges={
        ("load", "unpack"): SharedAxis(
            axis="行", chunk="行块", segmentable=False,
            note="一个 unpack 事件一次吃一个行块"),
        ("unpack", "matmul"): SharedAxis(
            axis="K", chunk="K 块", segmentable=True,
            consumed_by="测试: 只校验到此, 没有建图器"),
    },
    compared=("matmul",),
    drain=(("aic", ("matmul",)), ("aiv0", ("unpack",)), ("aiv1", ("load",))),
    steal_groups=(("matmul", ("unpack",)),),
    default_items={"load": 0},
)


def test_another_vocabulary_drives_granularity():
    g = GranularityAssignment(vocab=OTHER)
    assert g.items("load") == 0 and g.items("matmul") == 1
    assert g.coarser_than_default() == ()
    assert g.with_stage("matmul", 4).coarser_than_default() == ("matmul",)
    assert resolve_granularity({"unpack": 2}, vocab=OTHER).items("unpack") == 2
    with pytest.raises(ValueError, match="未知 stage"):
        StageGranularity("gmm1", 1, vocab=OTHER)
    with pytest.raises(ValueError, match="未知 stage"):
        g.of("gmm1")


def test_another_vocabulary_drives_roles():
    r = RoleAssignment(vocab=OTHER)
    assert r.role_of("matmul") == AIC and r.role_of("load") == AIV1
    assert r.resource("unpack", 3) == f"{AIV0}:3"
    assert r.stages_on(AIC) == ("matmul",)
    with pytest.raises(ValueError, match="只能跑在"):
        RoleAssignment(overrides={"matmul": AIV0}, vocab=OTHER)
    with pytest.raises(KeyError, match="未知 stage"):
        r.role_of("gmm1")


def test_another_vocabulary_drives_edge_validation():
    validate_links([StageLink("unpack", "matmul", readiness="per_chunk")],
                   vocab=OTHER)
    # MegaMoE 的边在这份词汇表里不存在 —— 报错而不是静默接受
    with pytest.raises(ValueError, match="这份词汇表里没有的边"):
        validate_links([StageLink("activation", "gmm2")], vocab=OTHER)
    # 不可分段的边上写分段就绪照样被拒
    with pytest.raises(ValueError, match="load->unpack"):
        validate_links([StageLink("load", "unpack", readiness=2)], vocab=OTHER)


# ---- 3. 词汇表自身的护栏 ----

@pytest.mark.parametrize("bad,match", [
    (dict(unit_of={"load": "row"}), "unit_of"),
    (dict(roles={"load": AIV1, "unpack": AIV0}), "没给执行角色"),
    (dict(cube_only=("load",)), "缺省角色"),
    (dict(edges={("load", "swiglu"): SharedAxis("x", "y", False)}), "不在 pipeline"),
    (dict(compared=("swiglu",)), "compared"),
    (dict(drain=(("aic", ("swiglu",)),)), "drain"),
    (dict(steal_groups=(("swiglu", ()),)), "steal_groups"),
    (dict(default_steal_drivers=("load",)), "default_steal_drivers"),
    (dict(default_items={"load": -1}), "非负整数"),
    (dict(pipeline=("load", "load", "matmul")), "重复"),
])
def test_a_vocabulary_must_be_self_consistent(bad, match):
    """写得出的声明必须自洽: 校验在构造时报错, 而不是在建图时才炸."""
    base = dict(operator="toy", pipeline=OTHER.pipeline, unit_of=dict(OTHER.unit_of),
                roles=dict(OTHER.roles), cube_only=OTHER.cube_only,
                edges=dict(OTHER.edges), compared=OTHER.compared, drain=OTHER.drain,
                steal_groups=OTHER.steal_groups)
    base.update(bad)
    with pytest.raises(ValueError, match=match):
        StageVocabulary(**base)
