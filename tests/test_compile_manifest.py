"""编译清单: 从 C++/CMake 抽编译期参数, 并与 Python 侧对账.

针对的问题是一个**已经发生过**的 bug 类: 编译期参数在 kernel 源码与 Python 里各写一份,
没有任何机制让它们对上。KernelConfig.swizzle_direction 曾经缺省 1、注释声称 kernel 用
<3, 1>, 而 common/mega_moe_gmm_common.h:33 写的是 BlockSchedulerSwizzle<3, 0> ——
m 组 > 1 时模型的 tile 遍历顺序相对 kernel 是 M/N 转置的, 实测墙钟差 +5.0%。注释不会报错。

这组测试守三件事:
  1. 抽取**认不出就报错**, 不静默跳过 (少一项清单就失去意义);
  2. 派生关系不丢 (L1_TILE_M_256 = MEGAMOE_TILE_M 这种转写要能解到整数), 否则 Python 把
     派生量抄成字面量的脱钩查不出来;
  3. 对账能抓到**两个方向**的漂移 —— Python 改了, 或者 kernel 改了。
"""
from pathlib import Path

import pytest

import moe_cost_model as m
from moe_cost_model.implementations.manifest import (INTENTIONAL, compare, extract)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest():
    return extract(ROOT)


def test_extraction_finds_every_whitelisted_constant_without_errors(manifest):
    """白名单里的常数一个都不能少, 且不能有抽取错误.

    白名单 (manifest.STRUCTURAL) 是个断言: 名字消失或写法变了必须报, 不能默默少一项 ——
    少一项就等于那一项回到"没人对账"的状态。
    """
    assert manifest.errors == [], manifest.errors
    from moe_cost_model.implementations.manifest import STRUCTURAL
    for name in STRUCTURAL:
        assert name in manifest.entries, f"白名单常数 {name} 没抽到"
        assert manifest.value(name) is not None, f"{name} 解不成整数"


def test_cmake_cache_knobs_are_found_with_their_documentation(manifest):
    """五个 MEGAMOE_* cache 变量连说明一起抽出来 (CMakeLists 是它们唯一的定义处)."""
    for name in ("MEGAMOE_TILE_M", "MEGAMOE_TILE_N", "MEGAMOE_L1_BUF_NUM",
                 "MEGAMOE_TOPK_PREFETCH", "MEGAMOE_TOPO_URMA"):
        entry = manifest.entries[name]
        assert entry.kind == "cmake_cache"
        assert entry.doc, f"{name} 没抽到说明"
        assert "CMakeLists.txt:" in entry.source


def test_derived_constants_resolve_through_their_reference(manifest):
    """L1_TILE_M_256 = MEGAMOE_TILE_M 这种转写要解到整数, 派生关系不能丢.

    为什么重要: Python 的 URMA_FLAG_WINDOW_TOKENS 把它抄成了字面量 256, 于是改 tile_m
    带不动它。清单保留派生, 对账才能发现这种脱钩。
    """
    assert manifest.value("L1_TILE_M_256") == manifest.value("MEGAMOE_TILE_M") == 256
    assert manifest.value("L1_TILE_N") == manifest.value("MEGAMOE_TILE_N") == 256
    assert manifest.value("LAYERED_USABLE_UB_BYTES") == 248 * 1024   # 248U * 1024U
    assert manifest.value("META_INFO_SIZE") == 8


def test_include_guards_are_not_mistaken_for_compile_knobs(manifest):
    """头文件保护宏不是编译参数."""
    assert not [k for k in manifest.entries if k.endswith("_H")]


def test_the_calibration_harness_instantiation_is_recorded(manifest):
    """标定语料那份实例化要记下来: 模板实参决定这些标定值适用于哪个编译点.

    include/kernel.cpp 里写死了 CombineQuantMode=COMBINE_NO_QUANT 与
    IsGmm1Interleaved=false, TopkWeightsPrefetch 走 MEGAMOE_TOPK_PREFETCH (缺省 0)。
    也就是说全部实测数据都来自**一个**编译点 —— 这正是标定值要按编译指纹分域的理由。
    """
    h = manifest.harness
    assert h["class"] == "MegaMoeA8W8Wave"
    assert h["CombineQuantMode"].endswith("COMBINE_NO_QUANT")
    assert h["IsGmm1Interleaved"] == "false"
    assert h["TopkWeightsPrefetch"] == "MEGAMOE_TOPK_PREFETCH"
    assert manifest.value("MEGAMOE_TOPK_PREFETCH") == 0


def test_python_agrees_with_the_in_tree_kernel(manifest):
    """缺省 Python 取值与仓内源码一致 —— 这是这套工具的日常用法."""
    report = compare(manifest)
    assert report["errors"] == []
    assert report["mismatch"] == [], report["mismatch"]
    assert report["checked"] >= 18


def test_the_reference_profile_matches_the_kernel_on_meta_bytes(manifest):
    """声称"复现那份实现"的 profile, 在元数据字节上要与 kernel 相同.

    缺省 16 是算法下界口径 (四个具名字段), kernel 搬满 META_INFO_SIZE=8 个 int32 = 32B。
    profile 显式声明 32, 所以它**不走**有意不同那条豁免 —— 它必须真的相等。
    """
    report = compare(manifest, kernel=m.MEGAMOE_A8W8.kernel)
    assert report["mismatch"] == [], report["mismatch"]
    assert not [s for s in report["skipped"] if "combine_meta_bytes_per_row" in s]


def test_a_python_side_divergence_is_caught_with_its_source_line(manifest):
    """Python 改了而源码没改 -> 失配, 并指出源码在哪一行."""
    report = compare(manifest, kernel=m.KernelConfig(tile_n=128))
    assert len(report["mismatch"]) == 1
    bad = report["mismatch"][0]
    assert bad["python"] == "KernelConfig.tile_n"
    assert (bad["python_value"], bad["manifest_value"]) == (128, 256)
    assert "CMakeLists.txt:30" in bad["source"]


def test_a_kernel_side_divergence_is_caught(tmp_path):
    """**kernel 改了**而 Python 没跟上 -> 失配. 这是真实会发生的方向.

    做法: 把源码树复制到 tmp, 只改 swizzle 的那一个模板实参, 再抽一次。复刻的正是
    2026-10-05 修掉的那个错 —— 只不过这次是从源码那一侧发生。
    """
    import shutil
    src = ROOT / "mega_moe" / "op_kernel" / "arch35"
    dst = tmp_path / "mega_moe" / "op_kernel" / "arch35"
    dst.parent.mkdir(parents=True)
    shutil.copytree(src, dst)
    shutil.copytree(ROOT / "mega_moe" / "include", tmp_path / "mega_moe" / "include")
    shutil.copytree(ROOT / "mega_moe" / "op_host", tmp_path / "mega_moe" / "op_host")
    gmm = dst / "common" / "mega_moe_gmm_common.h"
    gmm.write_text(gmm.read_text().replace("BlockSchedulerSwizzle<3, 0>",
                                           "BlockSchedulerSwizzle<3, 1>"))
    moved = extract(tmp_path)
    assert moved.value("SWIZZLE_DIRECTION") == 1
    report = compare(moved)
    assert [b["python"] for b in report["mismatch"]] == [
        "KernelConfig.swizzle_direction"]
    assert report["mismatch"][0]["manifest_value"] == 1


def test_missing_sources_are_reported_not_silently_skipped(tmp_path):
    """源码树不在 -> 抽取错误要报出来, 而不是给一张空清单说"一致"."""
    man = extract(tmp_path)
    assert man.errors, "空目录竟然没有抽取错误"
    report = compare(man)
    assert report["errors"], "抽取错误必须传到对账结果里"


def test_intentional_differences_carry_a_reason():
    """登记为"有意不同"的项必须写明理由 —— 否则它就是个没人管的失配."""
    assert INTENTIONAL, "至少有一项 (combine 元数据字节)"
    for field, (entry, reason) in INTENTIONAL.items():
        assert entry and len(reason) > 30, f"{field} 的理由太短"


def test_the_profile_reproduces_the_in_repo_compile_point_with_nothing_exempted(manifest):
    """`profiles.MEGAMOE_A8W8` 声称"复现那份实现" —— 那就必须 0 失配且 **0 豁免**.

    缺省 `KernelConfig()` 跑对账是 0 失配 + **1 项有意豁免**
    (`combine_meta_bytes_per_row` 16 vs 32: 缺省取"四个具名字段"的算法下界, kernel 搬满
    META_INFO_SIZE=8 个 int32)。那条豁免是模型"缺省不引用任何实现"的体现, 不是对不上。

    而 profile 的意思正是"按那份实现来", 所以在它身上**连豁免都不该有**。这条把
    profiles.py 的那句话变成机械判据: 哪天 kernel 改了某个宏而 profile 没跟上, 这里就红。
    """
    import moe_cost_model as m

    default = compare(manifest, kernel=m.KernelConfig())
    assert not default["mismatch"]
    assert len(default.get("skipped", ())) == 1, (
        "缺省编译点的有意差异应当只有 combine_meta_bytes_per_row 一项")

    profile = compare(manifest, kernel=m.MEGAMOE_A8W8.kernel)
    assert not profile["mismatch"], profile["mismatch"]
    assert not profile.get("skipped"), (
        "profile 身上不该有任何豁免项 —— 它声称的就是复现仓内编译点: "
        f"{profile.get('skipped')}")
