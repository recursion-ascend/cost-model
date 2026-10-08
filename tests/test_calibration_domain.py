"""按域登记的标定值: 一个数只在它量过的地方有效 (步骤 8).

要替换的现状: 标定常数是 `config/hardware.py` 的模块级全局量, 一套数覆盖所有实现 / 编译点 /
形状 / 拓扑 —— 而那些常数自己的注释早就写明不是这样: BW_L1_GM 按核数重拟 28 核 45300 /
18 核 37000; BW_REMOTE_WRITE 在三个形状上 9.5 / 7.8 / 4.5 GB/s; URMA_GET_LAT_US 的域是
"4 卡 3 条流, 超出未验证"。模型此前没有地方记这件事, 越域时也没有地方提醒。

这组测试守四件事:
  1. 域内 / 越界 / 从未声明 三种结果必须分开 —— 后两者是两种不同的不确定性;
  2. 换编译指纹要报 wrong_key, **不能**默默拿另一个二进制上量的值来用;
  3. 不做自动外推 (那些依赖只有两三个点);
  4. 打点语料的编译点必须与"复现那份实现"的 profile 一致 —— 否则正常场景全报 wrong_key。
"""
import pytest

import moe_cost_model as m
from moe_cost_model.implementations import (A8W4WaveV1Declared, A8W8WaveV1,
                                            CORPUS_COMPILE, CompileConfig, LayeredV1,
                                            Unsupported, default_table)
from moe_cost_model.implementations.identity import ImplementationId

#: 语料的代表性实际形状。**不在这里再写一份** —— 2026-10-08 之前这里、工具里、
#: calibration.py 的域里三处各写一遍, 三处都把 hidden_dim 写成 I (4608) 而不是 2I。
from moe_cost_model.implementations import CORPUS_POINT as CORPUS_SHAPE
CORPUS_TOPO = {"world_size": 4, "active_cores": 28}


@pytest.fixture(scope="module")
def table():
    return default_table()


def test_corpus_compile_point_matches_the_reference_profile():
    """打点语料的编译点 = profiles.MEGAMOE_A8W8 的编译点.

    打点跑的是 kernel, 而 kernel 搬满 META_INFO_SIZE = 8 个 int32 = 32B; 模型缺省 16 是
    算法下界口径。语料若按 16 登记, 指纹就与 profile 不同, 于是"复现那份实现"的场景查标定时
    全部报 wrong_key —— 那是种子数据的错, 不是真换了二进制 (2026-10-05 自查时正是如此)。
    """
    assert CORPUS_COMPILE.combine_meta_bytes_per_row == 32
    assert CORPUS_COMPILE.fingerprint == m.MEGAMOE_A8W8.compile_config.fingerprint


def test_inside_the_measured_domain_everything_is_usable(table):
    """语料自己的形状与拓扑上, 登记过的值都该是域内 (或明确标出未声明的维)."""
    rows = table.audit(implementation=A8W8WaveV1().identity(),
                       compile_fingerprint=CORPUS_COMPILE.fingerprint,
                       shape=CORPUS_SHAPE, topology=CORPUS_TOPO)
    assert rows, "语料形状下应当查到记录"
    assert not [n for n, got in rows.items() if got.verdict == "out_of_domain"]
    assert table.lookup("bw_l1_gm", implementation=A8W8WaveV1().identity(),
                        compile_fingerprint=CORPUS_COMPILE.fingerprint,
                        shape=CORPUS_SHAPE, topology=CORPUS_TOPO).usable


def test_out_of_range_and_undeclared_are_different_answers(table):
    """越界与"这一维从没声明过"必须分开.

    bw_local_gm 的注释只说了"单核大块 MTE, 无其它流量", 没有给任何形状范围 —— 所以它在
    任何形状上都是 undeclared, 而不是 in_domain (那会谎称量过) 也不是 out_of_domain
    (那会谎称量过且超了)。
    """
    ident = A8W8WaveV1().identity()
    local = table.lookup("bw_local_gm", implementation=ident,
                         compile_fingerprint=CORPUS_COMPILE.fingerprint,
                         shape=CORPUS_SHAPE, topology=CORPUS_TOPO)
    assert local.verdict == "undeclared" and local.undeclared
    assert not local.usable
    wide = table.lookup("bw_l1_gm", implementation=ident,
                        compile_fingerprint=CORPUS_COMPILE.fingerprint,
                        shape={**CORPUS_SHAPE, "h": 6144}, topology=CORPUS_TOPO)
    assert wide.verdict == "out_of_domain" and wide.out_of_range == ("h",)


def test_topology_mismatch_is_caught_separately(table):
    """核数不同要报出来: BW_L1_GM 按核数重拟差 1.40 倍, 那是拓扑依赖不是形状依赖."""
    got = table.lookup("bw_l1_gm", implementation=A8W8WaveV1().identity(),
                       compile_fingerprint=CORPUS_COMPILE.fingerprint,
                       shape=CORPUS_SHAPE, topology={"world_size": 4, "active_cores": 18})
    assert got.verdict == "out_of_domain"
    assert got.topology_mismatch == ("active_cores",)
    assert got.record.spread == pytest.approx(51900 / 37000, rel=1e-6)


def test_another_compile_point_is_refused_not_silently_reused(table):
    """换了编译指纹就是换了二进制: 报 wrong_key, 不拿别处量的值顶上.

    这正是"不能继续用一套全局常数覆盖所有编排"的那一条 —— 以前没有键, 所以必然顶上。
    """
    got = table.lookup("bw_l1_gm", implementation=A8W8WaveV1().identity(),
                       compile_fingerprint=CompileConfig(tile_n=128).fingerprint,
                       shape=CORPUS_SHAPE, topology=CORPUS_TOPO)
    assert got is not None and got.verdict == "wrong_key"
    assert "不同的二进制" in got.note
    assert not got.usable


def test_an_unknown_implementation_returns_nothing_rather_than_a_default(table):
    """没登记就返回 None —— 不能悄悄回落到模块全局量."""
    other = ImplementationId("ascend950", "someone.elses_kernel", "v1")
    assert table.lookup("bw_l1_gm", implementation=other,
                        compile_fingerprint=CORPUS_COMPILE.fingerprint) is None


def test_layered_constants_live_under_the_layered_implementation(table):
    """URMA 的常数挂在 layered 实现上, 不在 a8w8 上 —— 它们是两条不同的通信路径."""
    layered_fp = CompileConfig(comm_mode="urma", provenance="URMA 探针").fingerprint
    got = table.lookup("urma_get_lat_us", implementation=LayeredV1().identity(),
                       compile_fingerprint=layered_fp,
                       shape={"h": 5120}, topology={"world_size": 4,
                                                    "concurrent_streams": 3})
    assert got is not None and got.usable
    # 同一个名字在 a8w8 下查不到 (而不是查到一个"通用"值)
    assert table.lookup("urma_get_lat_us", implementation=A8W8WaveV1().identity(),
                        compile_fingerprint=CORPUS_COMPILE.fingerprint) is None


def test_spread_quantifies_how_unstable_a_constant_is(table):
    """离散度要能直接读出来: BW_REMOTE_WRITE 在三个形状上差 2.11 倍.

    比模型平时争论的差异大得多 —— 这就是"一套全局常数"最贵的地方。
    """
    ident = A8W8WaveV1().identity()
    got = table.lookup("bw_remote_write", implementation=ident,
                       compile_fingerprint=CORPUS_COMPILE.fingerprint,
                       shape=CORPUS_SHAPE, topology=CORPUS_TOPO)
    assert got.record.spread == pytest.approx(9500 / 4500, rel=1e-6)
    assert len(got.record.observations) == 3
    assert all(evidence for evidence in got.record.domain.evidence)


def test_no_automatic_extrapolation_is_offered(table):
    """越域时只报判定, 不给"修正值" —— 两三个点造不出一条曲线."""
    got = table.lookup("bw_l1_gm", implementation=A8W8WaveV1().identity(),
                       compile_fingerprint=CORPUS_COMPILE.fingerprint,
                       shape={**CORPUS_SHAPE, "h": 6144},
                       topology={"world_size": 4, "active_cores": 18})
    assert got.verdict == "out_of_domain"
    assert got.record.value == 51900.0, "越域不得改写原值"
    assert not hasattr(got, "corrected_value") and not hasattr(got, "extrapolated")


def test_the_default_example_scenario_is_out_of_domain():
    """项目自己的缺省场景跑在标定域之外 —— 这件事必须能被查出来.

    scenario_basic.toml 是 h=6144 / hidden_dim=4096 / topk=8, 而全部实测都在
    h=5120 / hidden=4608 / topk=6 上做的。不是说结果没用, 而是那几个带宽常数的来源条件与
    它不同, 读结论时要知道。
    """
    from moe_cost_model.implementations.calibration import audit_run
    scenario = m.load_scenario(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples" / "scenario_basic.toml")
    result = m.simulate(scenario)["rank_results"][0]
    rows = audit_run(result["implementation"],
                     {"token_num": scenario.workload.tokens, "h": scenario.h,
                      "hidden_dim": scenario.hidden_dim,
                      "topk": scenario.workload.topk})
    out = [n for n, got in rows.items() if got.verdict == "out_of_domain"]
    assert out, "缺省场景的形状在标定域外, 审计却一条都没报"
    assert "bw_l1_gm" in out


# --------------------------------------------------------- A8W4: 声明但未建模

def test_a8w4_is_declared_and_refuses_with_the_missing_facts():
    """第三份实现有身份、有源码依据, 但会拒绝 —— 并说清差哪一步.

    三种答案信息量不同: 没有这个名字 (像没想过)、有名字但凭空给数 (最坏)、有名字且说清
    差什么 (可以照着补)。这里是第三种。差的是一个**量** (4bit->8bit 展开的向量吞吐),
    不是一个参数 —— 仓内没有 A8W4 的打点。
    """
    adapter = A8W4WaveV1Declared()
    assert adapter.identity().key == "ascend950.megamoe.a8w4_wave.v1"
    assert adapter.identity().source_refs
    names = [name for name, _ in adapter.MISSING]
    assert "weight_antiquant_bytes_per_us" in names
    with pytest.raises(Unsupported) as exc:
        adapter.accepts(CompileConfig(), m.ModelOptions())
    text = str(exc.value)
    assert "weight_antiquant_bytes_per_us" in text and "实测" in text
    for name, why in adapter.MISSING:
        assert len(why) > 30, f"{name} 没说清为什么缺"


def test_a8w4_source_refs_point_at_real_files():
    """声明的源码依据必须真的在仓里 —— 包括那段权重反量化前段的头文件."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for ref in A8W4WaveV1Declared().identity().source_refs:
        assert (root / ref.split(":")[0]).exists(), ref


def test_a8w4_cannot_be_used_by_accident():
    """按名字选到 A8W4 时, plan/lower 也要拒绝, 不能半路给出一张残图."""
    from moe_cost_model.implementations import resolve
    adapter = resolve("a8w4")
    assert adapter.identity() == A8W4WaveV1Declared().identity()
    with pytest.raises(Unsupported):
        adapter.plan(object(), CompileConfig(), m.ModelOptions())
    with pytest.raises(Unsupported):
        adapter.lower(object(), None, None, m.ModelOptions())


#: 标定记录名 -> 它登记的那个模块常数。标定表的 value 必须**就是**那个常数:
#: 两边各写一遍数字时, 重标常数后这张表会留着旧值, 还继续声称自己是它的域记录 ——
#: 正是这个模块存在的目的要防的事。
RECORD_TO_CONSTANT = {
    "bw_l1_gm": "BW_L1_GM",
    "bw_remote_write": "BW_REMOTE_WRITE",
    "bw_unpermute_agg": "BW_UNPERMUTE_AGG",
    "t_rank_sync_rtt_us": "T_RANK_SYNC_RTT_US",
    "bw_local_gm": "BW_LOCAL_GM",
    "urma_get_lat_us": "URMA_GET_LAT_US",
    "urma_get_bw_single": "URMA_GET_BW_SINGLE",
}


def test_every_record_value_is_the_module_constant():
    from moe_cost_model.config import hardware as hw
    from moe_cost_model.implementations.calibration import default_table

    by_name = {r.name: r for r in default_table().records()}
    for record_name, const in RECORD_TO_CONSTANT.items():
        assert record_name in by_name, f"标定表里没有 {record_name}"
        assert by_name[record_name].value == float(getattr(hw, const)), (
            f"{record_name} 登记 {by_name[record_name].value}, "
            f"而 hardware.{const} 是 {float(getattr(hw, const))}")


def test_no_record_is_left_unmapped():
    """新加一条标定记录时, 要么它对应某个常数 (进上表), 要么说明它为什么不对应."""
    from moe_cost_model.implementations.calibration import default_table

    #: 不对应单个模块常数的记录 (留空 = 当前没有)
    STANDALONE: set = set()
    names = set(default_table().names())
    unmapped = names - set(RECORD_TO_CONSTANT) - STANDALONE
    assert not unmapped, (
        f"这些标定记录没说清对应哪个常数: {sorted(unmapped)}")
