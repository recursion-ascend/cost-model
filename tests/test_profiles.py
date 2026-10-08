"""实例层 profile: "某一份实现是设计空间里的哪个点".

要守住的性质: 模型的缺省值不引用任何实现; 要复现实现就显式引用 profile。
所以这里测的是**分层本身**, 不是某个数。
"""
import dataclasses

import pytest

import moe_cost_model as m


def test_profile_carries_every_implementation_specific_choice():
    """profile 里的编排取值必须是显式写出的, 不是"跟缺省一样"就算."""
    P = m.MEGAMOE_A8W8
    assert P.options.dispatch_partition  # 谁取哪些行
    assert P.options.dispatch_pacing     # 下一波 dispatch 等什么
    g2 = P.options.link("activation", "gmm2")
    assert g2.location == "gm"                         # 物化
    assert g2.readiness == m.Readiness.first_chunk()    # 首块一段 + 其余一段
    g1 = P.options.link("gmm1", "activation")
    assert g1.location == "onchip" and g1.depth == 1 and g1.colocated_by_hardware
    assert P.options.epilogue_overheads is not None
    assert P.kernel is not None and P.policy is not None
    assert type(P.tile_grid).__name__ == "SwizzledTileGrid"


def test_profile_has_a_source():
    """每个取值都要能回答"出处是什么" —— profile 自己带出处."""
    assert "mega_moe" in m.MEGAMOE_A8W8.source


def test_with_options_only_changes_what_is_named():
    P = m.MEGAMOE_A8W8
    changed = P.with_options(m_groups_per_wave=4)
    assert changed.m_groups_per_wave == 4
    assert changed.dispatch_partition == P.options.dispatch_partition
    assert changed.epilogue_overheads is P.options.epilogue_overheads


def test_unknown_profile_name_is_refused():
    from moe_cost_model.profiles import resolve_profile
    with pytest.raises(ValueError, match="未知 profile"):
        resolve_profile("no-such-kernel")


def test_scenario_profile_is_a_base_that_explicit_fields_override(tmp_path):
    """场景文件 profile= 给底, 文件里写出的字段覆盖它 (只写一项, 其余沿用)."""
    src = open("examples/scenario_basic.toml").read()
    f = tmp_path / "s.toml"
    f.write_text(src + "\n[options]\nm_groups_per_wave = 4\n")
    sc = m.load_scenario(str(f))
    assert sc.options.m_groups_per_wave == 4                     # 文件说的
    assert sc.options.dispatch_partition == \
        m.MEGAMOE_A8W8.options.dispatch_partition                # profile 说的
    assert type(sc.tile_grid).__name__ == "SwizzledTileGrid"     # profile 说的


def test_scenario_without_profile_takes_model_defaults():
    """不写 profile 的场景沿用模型缺省, 不沿用任何实现的取值."""
    sc = m.load_scenario("examples/scenario_basic.toml")
    bare = dataclasses.replace(sc, profile="", options=m.ModelOptions(), tile_grid=None)
    assert bare.options == m.ModelOptions()
