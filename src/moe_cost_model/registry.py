"""命名策略注册表: 场景文件用名字引用策略对象.

策略写法 (场景文件与 Scenario 字段通用):
  "balanced_waves"                                  名字, 无参构造
  {name = "priority_by_stage", stage_order = [...]}  名字 + 构造参数
  BalancedWaves()                                   Python 里直接给对象

orchestration (建图器) 不实例化, 取的是类, 另外支持 "包.模块:类" 直接引用:
  orchestration = "mte"
  orchestration = "my_pkg.my_builder:MyBuilder"

新增策略: register("wave_packing", "my_packing", MyPacking).
"""
from __future__ import annotations

import difflib
import importlib
from typing import Callable, Dict, List

from .analysis.stealing import idle_core_stealing
from .planning.core_assignment import ContiguousBlock, GreedyLeastBusy, StaticRoundRobin
from .planning.tile_grid import (RowMajorTileGrid, SplitRowsTileGrid,
                                 SwizzledTileGrid)
from .planning.wave_packing import BalancedWaves, LongestExpertFirst, SequentialGreedy
from .scheduler.policies import CriticalPathFirst, EarliestStart, PriorityByStage

_REGISTRY: Dict[str, Dict[str, Callable[..., object]]] = {
    "wave_packing": {
        "sequential_greedy": SequentialGreedy,
        "longest_expert_first": LongestExpertFirst,
        "balanced_waves": BalancedWaves,
    },
    "core_assignment": {
        "static_round_robin": StaticRoundRobin,
        "greedy_least_busy": GreedyLeastBusy,
        "contiguous_block": ContiguousBlock,
    },
    "scheduling_policy": {
        "earliest_start": EarliestStart,
        "critical_path_first": CriticalPathFirst,
        "priority_by_stage": PriorityByStage,
    },
    "restructure": {
        "idle_core_stealing": idle_core_stealing,
    },
    "tile_grid": {
        "row_major": RowMajorTileGrid,
        "swizzled": SwizzledTileGrid,
        "split_rows": SplitRowsTileGrid,
    },
}

# 建图器: 值是类的引用路径, 用时才导入 (避免与 builders 的导入环)
_ORCHESTRATION: Dict[str, str] = {
    "mte": "moe_cost_model.builders.mte:MteEventBuilder",
    "layered": "moe_cost_model.builders.layered:LayeredEventBuilder",
}

KINDS = tuple(_REGISTRY) + ("orchestration",)


def suggest(word: str, choices) -> str:
    """拼写提示: 返回 ", 是否想写 'x'?" 或空串."""
    close = difflib.get_close_matches(str(word), [str(c) for c in choices], n=1, cutoff=0.6)
    return f", 是否想写 '{close[0]}'?" if close else ""


def register(kind: str, name: str, factory: Callable[..., object]) -> None:
    """注册一个策略. kind="orchestration" 时 factory 是建图器类 (不实例化)."""
    if kind == "orchestration":
        _ORCHESTRATION[name] = factory
        return
    if kind not in _REGISTRY:
        raise ValueError(f"未知策略类别 '{kind}'; 可选: {', '.join(KINDS)}")
    _REGISTRY[kind][name] = factory


def names(kind: str) -> List[str]:
    return sorted(_ORCHESTRATION if kind == "orchestration" else _REGISTRY[kind])


def builder_class(spec, where: str = "orchestration"):
    """建图器: 注册名 / "包.模块:类" / 直接给类. None 表示按 kernel 自动选."""
    if spec is None or isinstance(spec, type):
        return spec
    if not isinstance(spec, str):
        raise ValueError(f"{where}: 应为注册名、\"包.模块:类\" 或建图器类, 得到 {spec!r}")
    if ":" in spec:
        module, _, cls = spec.partition(":")
        try:
            return getattr(importlib.import_module(module), cls)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"{where}: 无法从 '{spec}' 取到建图器类 — {exc}") from None
    target = _ORCHESTRATION.get(spec)
    if target is None:
        raise ValueError(f"{where}: 未知建图器 '{spec}'{suggest(spec, _ORCHESTRATION)}; "
                         f"可选: {', '.join(names('orchestration'))}, "
                         "或写 \"包.模块:类\"")
    return target if isinstance(target, type) else builder_class(target, where)


def resolve(kind: str, spec, where: str = ""):
    """策略写法 → 策略对象. None 原样返回 (模型缺省); 非 str/dict 视为现成对象."""
    where = where or kind
    if kind == "orchestration":
        return builder_class(spec, where)      # 取类, 不实例化
    if spec is None or not isinstance(spec, (str, dict)):
        return spec
    if isinstance(spec, str):
        name, kwargs = spec, {}
    else:
        kwargs = dict(spec)
        name = kwargs.pop("name", None)
        if not isinstance(name, str):
            raise ValueError(f"{where}: 表写法必须带 name 字段, 可选: {', '.join(names(kind))}")
    table = _REGISTRY[kind]
    if name not in table:
        raise ValueError(f"{where}: 未知策略 '{name}'{suggest(name, table)}; "
                         f"可选: {', '.join(names(kind))}")
    try:
        return table[name](**kwargs)
    except TypeError as exc:
        raise ValueError(f"{where}: 策略 '{name}' 的参数不对: {exc}") from None
