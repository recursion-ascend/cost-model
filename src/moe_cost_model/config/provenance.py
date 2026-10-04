"""参数出处系统: 机器可读的常数来源标签.

分类 (source 前缀) —— 按**这个数是谁定的**分, 不按它写在哪里:

  spec:<文档>       硬件规格/格式标准: 容量、位宽、聚合带宽、MX 量化格式。换一份实现
                     不会变。规格算力是**上界**, 用它算出的是时间下界, 该配效率系数。
  algo:<算法>       算法本身的定义: SwiGLU 有 gate+up 两个投影、GMM2 的 K 就是 GMM1
                     的 N。换算法才变, 换实现不变。
  impl:<实现>       **某一份实现的选择**: tile 几何、缓冲槽数、档位阈值、每行搬几个
                     元数据字段。换一份实现就会变 —— 所以它应该能被参数覆盖
                     (KernelConfig / InstancePolicy / ModelOptions), 模块常数只是
                     那份实现的缺省来源。缺省值不该引用它 (见 profiles.py)。
  measured:<实验>   单点/差分实测 (注明实验与适用域)。已含争用与开销, 不该再乘效率。
  derived:<公式>    从其他常数/规格推导
  assumed:<原因>    假设值 — 必须显式标注, 仿真报告中高亮

为什么把原来的 kernel: 拆成 spec/algo/impl: 那一个标签同时盖着"硬件容量"、"算法定义"
与"某实现的取值"三类东西, 于是读到一个 kernel: 常数时分不出"物理上只能这样"还是
"那份实现这么选的"。这正是把实现取值当成"应该的值"的来源。

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
        "spec": {p: entries[p] for p in cats.get("spec", [])},
        "algo": {p: entries[p] for p in cats.get("algo", [])},
        "impl": {p: entries[p] for p in cats.get("impl", [])},
        "measured": {p: entries[p] for p in cats.get("measured", [])},
    }


#: 允许的出处类别。新增类别要同时想清楚"这个数是谁定的"。
CATEGORIES = ("spec", "algo", "impl", "measured", "derived", "assumed")


def unknown_categories(entries: Dict[str, Tuple[float, str]]) -> Dict[str, str]:
    """返回类别不在 CATEGORIES 里的条目 {路径: 标签} —— 空字典表示全部合规."""
    bad = {}
    for path, (_val, src) in entries.items():
        cat = str(src).split(":", 1)[0]
        if cat not in CATEGORIES:
            bad[path] = src
    return bad
