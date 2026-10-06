"""README 的项目结构树必须与 src/ 下的实际文件一一对应.

树是手写的 (每个文件配一句"是什么", 生成不出来), 所以用测试钉双向一致:
列了不存在的文件 -> 红; 有文件没列进去 -> 红。
2026-10-06 之前这棵树停在四层架构之前, implementations/ ir/ validation/ 三个包与
config/ 下四个文件都不在树里。
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "moe_cost_model"

#: 树里不逐个列的文件 (每个包都有, 列出来只是噪声)
SKIP = {"__init__.py"}


def _listed() -> set:
    """树里出现的所有 .py 文件名 (树里的 __init__.py 只是示意, 不计)."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tree = readme[readme.index("## 项目结构"):]
    first = tree.index("```")
    tree = tree[first + 3:tree.index("```", first + 3)]
    return set(re.findall(r"[a-z_0-9]+\.py", tree)) - SKIP


def _real() -> set:
    out = set()
    for dirpath, _, filenames in os.walk(PKG):
        if "__pycache__" in dirpath:
            continue
        out.update(f for f in filenames if f.endswith(".py"))
    return out - SKIP


def test_tree_lists_no_file_that_does_not_exist():
    # examples/ 下的两个脚本也出现在树里, 它们不在包内
    extra = {"run_scenario.py", "run_basic.py"}
    ghosts = _listed() - _real() - extra
    assert not ghosts, f"README 的结构树里有不存在的文件: {sorted(ghosts)}"


def test_tree_lists_every_module():
    missing = _real() - _listed()
    assert not missing, (
        f"这些模块不在 README 的结构树里: {sorted(missing)} —— "
        "新模块要进树, 否则读者看不到它存在")


def test_readme_documents_every_returned_field():
    """"输出解读"那张表必须覆盖 simulate() 真实返回的每个字段.

    表头说"返回以下可分析字段", 所以漏一个就是表在骗人。
    """
    import re

    import moe_cost_model as m

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## 输出解读"):readme.index("## 常数的出处")]
    named = set(re.findall(r"`([a-z_0-9]+)`", section))
    named |= set(re.findall(r'rank_results\[r\]\["([a-z_0-9]+)"\]', section))

    res = m.simulate(m.load_scenario(str(ROOT / "examples" / "scenario_basic.toml")))
    missing_top = set(res) - named - {"rank_results"}
    missing_rank = set(res["rank_results"][0]) - named
    assert not missing_top, f"顶层字段没在表里: {sorted(missing_top)}"
    assert not missing_rank, f"rank_results 字段没在表里: {sorted(missing_rank)}"


#: README「精度边界」里那两张表声称的常数值。单位: 带宽按 GB/s (代码是 B/µs),
#: 其余按代码的原单位。2026-10-06 核对时 ACT 每向量字节数在 README 里还写着 580,
#: 而代码早已是 722 (见 config/hardware.py 的两点 m 扫说明)。
README_CONSTANTS = {
    "BW_L1_GM": (51.9, 1000.0),
    "BW_UB": (93.0, 1000.0),
    "BW_SCATTER": (139.5, 1000.0),
    "BW_LOCAL_GM": (157.0, 1000.0),
    "BW_REMOTE_GM": (31.0, 1000.0),
    "BW_WINDOW": (33.0, 1000.0),
    "URMA_GET_BW_SINGLE": (2.25, 1000.0),
    "URMA_PUT_BW_SINGLE": (2.25, 1000.0),
    "BW_UNPERMUTE_AGG": (950.0, 1000.0),
    "VEC_REG_WIDTH": (256, 1.0),
    "T_STARTUP_VEC": (1.48, 1.0),
    "ACT_BYTES_PER_VEC": (722, 1.0),
}


def test_readme_quotes_the_real_constant_values():
    """README 引用的每个常数值必须与代码一致, 且那个数字真的出现在 README 里."""
    from moe_cost_model.config import hardware as hw

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, (claimed, scale) in README_CONSTANTS.items():
        value = float(getattr(hw, name)) / scale
        tol = max(0.01, abs(claimed) * 0.005)
        assert abs(value - claimed) <= tol, (
            f"{name}: 代码 {value:g}, README 这张表按 {claimed:g} 写的")
        shown = f"{claimed:g}"
        assert shown in readme, f"{name} 的值 {shown} 没出现在 README 里"
