"""Python 注释里对 kernel 源码的 `文件:行号` 引用必须指得到.

这些引用是模型与 kernel 之间的唯一纸面链接 (一个常数为什么是这个值、一条边为什么是
这个形状, 全靠它们), 而它们会随 kernel 改版无声腐烂: 文件改名、行号漂移, Python 侧
一个字都不会变。

本测试核两件机械可判的事:
  * 被引用的文件在仓内存在 (缩写名也要能解析, 所以引用要写得够全);
  * 行号在文件长度内。

"那一行是不是真讲这件事"判不了, 要人看 —— 但上面两条能挡住绝大多数腐烂。
kernel 不在仓内 (只装了 Python 包) 时整体跳过。
"""
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "mega_moe"

#: 引用形如 <名字>.h:<行号> / <名字>.cpp:<行号> / <路径>CMakeLists.txt:<行号>,
#: 带目录与行号区间都行
CITE = re.compile(
    r"([A-Za-z0-9_/]+\.(?:h|cpp)|[A-Za-z0-9_/]*CMakeLists\.txt):(\d+)"
    r"(?:[-\u2013](\d+))?")

pytestmark = pytest.mark.skipif(not KERNEL.is_dir(), reason="仓内没有 kernel 源码")


def _index() -> dict:
    out = {}
    for dirpath, _, filenames in os.walk(KERNEL):
        for name in filenames:
            if name.endswith((".h", ".cpp")) or name == "CMakeLists.txt":
                out.setdefault(name, []).append(Path(dirpath) / name)
    return out


def _resolve(index: dict, cited: str):
    """引用 -> 仓内真实路径.

    **带目录的引用按完整路径核**: 只按文件名匹配时 `CMakeLists.txt` 这种常见名字
    在任何目录下都能"找到", 于是写错目录查不出来 (2026-10-06 就有三处写着
    megamoe_profile/CMakeLists.txt, 真实位置是 mega_moe/include/)。
    裸文件名才回落到按名字找 (允许省略 mega_moe_ 前缀)。
    """
    if "/" in cited:
        for prefix in ("", "mega_moe/", "mega_moe/op_kernel/arch35/"):
            candidate = ROOT / (prefix + cited)
            if candidate.is_file():
                return [candidate]
        return []
    base = os.path.basename(cited)
    # 裸文件名可能有多个候选 (CMakeLists.txt 在仓内有十来个), 全给出去, 由调用方
    # 判"有没有任一候选满足"。只取第一个会按 os.walk 的顺序随机误报。
    return list(index.get(base, ())) + list(index.get("mega_moe_" + base, ()))


def _citations():
    for base in ("src", "tools", "tests"):
        for dirpath, _, filenames in os.walk(ROOT / base):
            if "__pycache__" in dirpath:
                continue
            for name in filenames:
                if not name.endswith(".py") or name == Path(__file__).name:
                    continue        # 本文件的说明文字里有示例引用, 不是真引用
                path = Path(dirpath) / name
                for lineno, line in enumerate(
                        path.read_text(encoding="utf-8").splitlines(), 1):
                    for m in CITE.finditer(line):
                        yield path, lineno, m.group(1), int(m.group(2)), m.group(3)


def test_every_cited_kernel_file_exists():
    index = _index()
    missing = [(str(p.relative_to(ROOT)), n, cited)
               for p, n, cited, _, _ in _citations() if not _resolve(index, cited)]
    assert not missing, (
        "这些引用在仓内指不到文件 (写全文件名, 或 kernel 真的改名了): "
        + "; ".join(f"{p}:{n} -> {c}" for p, n, c in missing))


def test_every_cited_line_is_within_the_file():
    index = _index()
    bad = []
    for path, lineno, cited, begin, end in _citations():
        paths = _resolve(index, cited)
        if not paths:
            continue
        last = int(end) if end else begin
        lengths = [len(q.read_text(encoding="utf-8", errors="ignore").splitlines())
                   for q in paths]
        if not any(begin <= n and last <= n for n in lengths):
            bad.append(f"{path.relative_to(ROOT)}:{lineno} -> {cited}:{begin}"
                       f"{'-' + end if end else ''} "
                       f"(候选文件行数 {sorted(set(lengths))})")
    assert not bad, "引用的行号超出文件长度: " + "; ".join(bad)


def test_there_are_citations_to_check():
    """防止上面两条因为正则失配而空转通过."""
    assert len(list(_citations())) > 30
