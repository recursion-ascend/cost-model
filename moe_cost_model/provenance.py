"""参数出处系统: 机器可读的常数来源标签.

分类 (source 前缀):
  measured:<实验>   单点/差分实测 (注明实验与适用域)
  kernel:<文件:行>  kernel 源码结构常数 (移植值)
  derived:<公式>    从其他常数/规格推导
  assumed:<原因>    假设值 — 必须显式标注, 仿真报告中高亮

SourcedValue 是 float 子类: 算术透明, 原常数保留 .source 标签.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Tuple


class SourcedValue(float):
    """带出处标签的 float. 算术结果退化为普通 float (标签只在原常数上)."""

    __slots__ = ("source",)

    def __new__(cls, value: float, source: str):
        obj = float.__new__(cls, value)
        obj.source = source
        return obj

    def __reduce__(self):
        return (SourcedValue, (float(self), self.source))

    def __repr__(self):
        return f"SourcedValue({float(self)!r}, {self.source!r})"


class SourcedInt(int):
    """带出处标签的 int — 保持索引/range/整除语义.

    注: int 子类不支持非空 __slots__ (CPython 限制), source 存 __dict__.
    """

    def __new__(cls, value: int, source: str):
        obj = int.__new__(cls, value)
        obj.source = source
        return obj

    def __reduce__(self):
        return (SourcedInt, (int(self), self.source))

    def __repr__(self):
        return f"SourcedInt({int(self)!r}, {self.source!r})"


def collect_provenance(obj: Any, prefix: str = "") -> Dict[str, Tuple[float, str]]:
    """递归收集 dataclass/dict/tuple 中全部 SourcedValue.

    返回 {路径: (值, 出处)}; 非 SourcedValue 的叶子不收录.
    """
    out: Dict[str, Tuple[float, str]] = {}
    if isinstance(obj, (SourcedValue, SourcedInt)):
        out[prefix or "<root>"] = (float(obj), obj.source)
    elif is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            out.update(collect_provenance(getattr(obj, f.name),
                                           f"{prefix}.{f.name}" if prefix else f.name))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(collect_provenance(v, f"{prefix}[{k!r}]" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(collect_provenance(v, f"{prefix}[{i}]" if prefix else str(i)))
    return out


def provenance_report(entries: Dict[str, Tuple[float, str]]) -> Dict[str, Any]:
    """汇总: 分类计数 + assumed 高亮 (报告中必须显式暴露假设值)."""
    cats: Dict[str, List[str]] = {}
    for path, (val, src) in entries.items():
        cat = src.split(":", 1)[0]
        cats.setdefault(cat, []).append(path)
    return {
        "entries": entries,
        "summary": {c: len(v) for c, v in sorted(cats.items())},
        "assumed": {p: entries[p] for p in cats.get("assumed", [])},
        "measured": {p: entries[p] for p in cats.get("measured", [])},
        "kernel": {p: entries[p] for p in cats.get("kernel", [])},
    }
