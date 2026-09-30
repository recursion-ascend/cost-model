"""统一入口: 一个 Scenario 承载全部旋钮, simulate(scenario) 算出执行时间.

三种用法:
  Python   Scenario(workload=Workload(...), policy=InstancePolicy(gmm2_lag_waves=2))
  文件     load_scenario("base.toml")
  改旋钮   base.with_overrides({"policy.gmm2_lag_waves": 2, "kernel.tile_n": 128})

字段路径与对象属性同名: 文件里的 [policy] 表、with_overrides 的 "policy.xxx"、
Python 里的 scenario.policy.xxx 是同一个东西. 未知字段与类型错误立即报错.
"""
from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from . import registry
from .api import simulate_routing_counts
from .config.hardware import BW_L1_GM, BW_SCATTER, KernelConfig
from .config.pipeline import (BufferSlots, PhaseRates, PipelineConstraints,
                              QueueDepths, SyncLatency)
from .config.policy import InstancePolicy, StageWaveOffsets
from .costs import (DispatchMechanisticLatency, PrimitiveCosts,
                    UrmaMechanisticLatency, build_analytical_costs)
from .scheduler.events import Channel, default_channels
from .shape import EngineQueueDepths, ModelOptions

ROUTING_MODES = ("uniform", "cyclic", "random", "explicit", "file")


# ---------------------------------------------------------------------------
# 工作量
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Workload:
    """路由工作量: 每 rank 的 token 数 + 路由计数 C[dst][expert][src] 的来源.

    routing:
      uniform   每个源 rank 的 tokens×topk 行均分到全部专家; 除不尽的余数行
                等间隔撒开, 各目的 rank 收到的行数相差不超过 1
      cyclic    token t 的第 k 个 slot 发给专家 (t + k + ((src+1) % world)×local) % 专家总数
      random    每 token 随机选 topk 个不同专家 (Python 随机流, 种子 seed + src;
                与 prof 仓 torch 生成器的随机流不同, 同种子的路由不相同)
      explicit  counts 直接给出
      file      从 JSON 文件读 (嵌套列表, 或带 routing_counts 键的对象)
    生成类路由 (前三种) 必须给 world 与 local_experts; explicit/file 从数据推出.
    """

    tokens: int = 64
    topk: int = 8
    world: Optional[int] = None
    local_experts: Optional[int] = None
    routing: str = "uniform"
    seed: int = 0
    counts: Optional[tuple] = None
    file: Optional[str] = None
    shared_expert_num: int = 0

    def __post_init__(self) -> None:
        if self.routing not in ROUTING_MODES:
            raise ValueError(f"routing: 未知模式 '{self.routing}'"
                             f"{registry.suggest(self.routing, ROUTING_MODES)}; "
                             f"可选: {', '.join(ROUTING_MODES)}")
        if self.tokens <= 0 or self.topk <= 0:
            raise ValueError("tokens 与 topk 必须为正")
        if self.counts is not None:
            object.__setattr__(self, "counts", _freeze_counts(self.counts))
        if self.routing == "explicit" and self.counts is None:
            raise ValueError("routing = 'explicit' 需要 counts")
        if self.routing == "file" and not self.file:
            raise ValueError("routing = 'file' 需要 file")
        if self.counts is not None and self.routing != "explicit":
            raise ValueError(f"给了 counts 时 routing 只能是 'explicit', 现为 '{self.routing}'")
        if self.routing == "explicit":
            world = len(self.counts)
            local = len(self.counts[0]) if world else 0
            if self.world not in (None, world) or self.local_experts not in (None, local):
                raise ValueError(f"counts 是 {world} rank × {local} 专家, 与 world="
                                 f"{self.world} / local_experts={self.local_experts} 不符")
        if self.routing in ("uniform", "cyclic", "random"):
            if not self.world or not self.local_experts:
                raise ValueError(f"routing = '{self.routing}' 需要 world 与 local_experts")
            if self.topk > self.world * self.local_experts:
                raise ValueError("topk 不能超过专家总数 world × local_experts")

    def routing_counts(self) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        """C[dst][expert][src] 行数."""
        if self.routing == "explicit":
            return self.counts
        if self.routing == "file":
            data = json.loads(Path(self.file).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data["routing_counts"]
            return _freeze_counts(data)
        world, local = self.world, self.local_experts
        experts = world * local
        counts = [[[0] * world for _ in range(local)] for _ in range(world)]

        def send(src: int, gid: int, rows: int = 1) -> None:
            counts[gid // local][gid % local][src] += rows

        for src in range(world):
            if self.routing == "uniform":
                base, rem = divmod(self.tokens * self.topk, experts)
                for gid in range(experts):
                    send(src, gid, base)
                # 余数行等间隔撒到专家上, 并按源 rank 错位: 各目的 rank 与各专家
                # 收到的行数都尽量平 (不堆在编号靠前的专家上)
                for i in range(rem):
                    send(src, (i * experts // rem + src) % experts)
            elif self.routing == "cyclic":
                shift = ((src + 1) % world) * local
                for t in range(self.tokens):
                    for k in range(self.topk):
                        send(src, (t + k + shift) % experts)
            else:
                rng = random.Random(self.seed + src)
                for _ in range(self.tokens):
                    for gid in rng.sample(range(experts), self.topk):
                        send(src, gid)
        return _freeze_counts(counts)


def _freeze_counts(counts) -> tuple:
    try:
        return tuple(tuple(tuple(int(x) for x in row) for row in dst) for dst in counts)
    except TypeError:
        raise ValueError("routing counts 必须是三层嵌套列表 C[dst][expert][src]") from None


# ---------------------------------------------------------------------------
# 标定覆盖
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """解析公式的标定参数. None = 取 config/hardware.py 的实测常数.

    字段与 build_analytical_costs 的参数一一对应; dispatch / urma 是两条传输
    路径的机制延迟参数。cube_mac_per_us 缺省 0 = 不计计算项 —— 实测域内
    GMM1/GMM2 都是权重载入绑定, 计算远小于载入。
    """

    cube_mac_per_us: float = 0.0
    bw_l1_gm: Optional[float] = None
    bw_l1_gm_b_nz: float = 0.0
    gmm1_fill_us: float = 0.0
    gmm1_tile_restart_us: float = 0.0     # 单缓冲下 L1 换块停顿; 只作用于 GMM1
    bw_ub: Optional[float] = None
    t_startup_us: Optional[float] = None
    bw_scatter: Optional[float] = None
    count_table_prepare_us: Optional[float] = None
    dispatch: DispatchMechanisticLatency = field(default_factory=DispatchMechanisticLatency)
    urma: Optional[UrmaMechanisticLatency] = None


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """一次仿真的完整输入.

    策略字段 (wave_packing / core_assignment / scheduling_policy / restructure /
    tile_grid / orchestration) 接受注册名、{name=..., 参数} 表或现成对象, 见
    registry.py; None = 模型缺省。tile_grid 决定 GMM1/GMM2 的 tile 怎么切,
    orchestration 决定用哪个建图器 (还能写 "包.模块:类" 引用自己的实现)。
    default_channels: 为相位流水启用默认 L2 信道 (每核应得速率 × 核数), 随
    aic_num 与标定带宽自动取值; 与 options.pipeline.channels 显式列表互斥.
    costs: 显式给出的公式容器; 给定时 calibration 不生效.
    """

    workload: Workload
    h: int = 6144
    hidden_dim: int = 4096
    aic_num: int = 28
    name: str = ""
    p1_override: int = 0
    p2_override: int = 0
    kernel: KernelConfig = field(default_factory=KernelConfig)
    policy: InstancePolicy = field(default_factory=InstancePolicy)
    options: ModelOptions = field(default_factory=ModelOptions)
    calibration: Calibration = field(default_factory=Calibration)
    default_channels: bool = False
    wave_packing: object = None
    core_assignment: object = None
    scheduling_policy: object = None
    restructure: object = None
    tile_grid: object = None
    orchestration: object = None
    costs: Optional[PrimitiveCosts] = None

    def __post_init__(self) -> None:
        for kind in registry.KINDS:
            registry.resolve(kind, getattr(self, kind), where=kind)   # 名字尽早校验
        pipe = self.options.pipeline
        if self.default_channels and pipe is not None and pipe.channels:
            raise ValueError("default_channels 与 options.pipeline.channels 只能给一个")

    # ---- 构造 ----

    @classmethod
    def from_dict(cls, data: Mapping[str, object], base_dir=None) -> "Scenario":
        """从嵌套字典 (场景文件的解析结果) 构造. 未知字段与类型错误立即报错."""
        data = dict(data)
        if "costs" in data:
            raise ValueError("costs 只能在 Python 里给对象; 文件里用 [calibration] 覆盖标定参数")
        if "workload" not in data:
            raise ValueError("缺 [workload] 表")
        wl = dict(data["workload"]) if isinstance(data["workload"], dict) else data["workload"]
        if isinstance(wl, dict) and wl.get("file") and base_dir is not None:
            wl["file"] = str((Path(base_dir) / str(wl["file"])).resolve())
        if isinstance(wl, dict) and "counts" in wl and "routing" not in wl:
            wl["routing"] = "explicit"
        data["workload"] = wl
        return _build(cls, data, "")

    def with_overrides(self, overrides: Mapping[str, object]) -> "Scenario":
        """按点分路径改旋钮, 返回新场景. 例: {"policy.gmm2_lag_waves": 2}."""
        scenario = self
        for path, value in overrides.items():
            scenario = _set_path(scenario, path.split("."), value, "")
        return scenario

    # ---- 导出 ----

    def to_dict(self, defaults: bool = True) -> Dict[str, object]:
        """嵌套字典. defaults=False 时只保留与缺省值不同的字段 (便于看改了什么)."""
        out = _dump(self) if defaults else _dump_changed(self)
        out.pop("costs", None)
        if self.costs is not None:
            out["costs"] = "<显式 PrimitiveCosts 对象>"
        return out

    # ---- 仿真输入 ----

    def build_costs(self) -> PrimitiveCosts:
        if self.costs is not None:
            return self.costs
        cal = self.calibration
        extra = {}
        if cal.count_table_prepare_us is not None:
            extra["count_table_prepare_us"] = cal.count_table_prepare_us
        return build_analytical_costs(
            h=self.h, kernel=self.kernel,
            dispatch_mechanistic=cal.dispatch, urma_mechanistic=cal.urma,
            bw_l1_gm=cal.bw_l1_gm, bw_l1_gm_b_nz=cal.bw_l1_gm_b_nz,
            cube_mac_per_us=cal.cube_mac_per_us,
            gmm1_fill_us=cal.gmm1_fill_us,
            gmm1_tile_restart_us=cal.gmm1_tile_restart_us, bw_ub=cal.bw_ub,
            t_startup_us=cal.t_startup_us, bw_scatter=cal.bw_scatter, **extra)

    def resolved_options(self) -> ModelOptions:
        """default_channels 展开后的 ModelOptions."""
        if not self.default_channels:
            return self.options
        cal = self.calibration
        channels = default_channels(
            self.aic_num,
            bw_l1_gm=cal.bw_l1_gm if cal.bw_l1_gm is not None else BW_L1_GM,
            bw_scatter=cal.bw_scatter if cal.bw_scatter is not None else BW_SCATTER)
        pipe = self.options.pipeline or PipelineConstraints()
        return dataclasses.replace(
            self.options, pipeline=dataclasses.replace(pipe, channels=channels))


def simulate(scenario: Scenario) -> Dict[str, object]:
    """场景 → 执行时间. 返回 simulate_routing_counts 的结果, 另带 scenario."""
    wl = scenario.workload
    result = simulate_routing_counts(
        routing_counts=wl.routing_counts(), token_num_per_rank=wl.tokens,
        h=scenario.h, hidden_dim=scenario.hidden_dim, aic_num=scenario.aic_num,
        costs=scenario.build_costs(), topk=wl.topk,
        shared_expert_num=wl.shared_expert_num,
        kernel=scenario.kernel, options=scenario.resolved_options(),
        policy=scenario.policy,
        restructure=registry.resolve("restructure", scenario.restructure),
        p1_override=scenario.p1_override, p2_override=scenario.p2_override,
        wave_packing=registry.resolve("wave_packing", scenario.wave_packing),
        core_assignment=registry.resolve("core_assignment", scenario.core_assignment),
        scheduling_policy=registry.resolve("scheduling_policy", scenario.scheduling_policy),
        tile_grid=registry.resolve("tile_grid", scenario.tile_grid),
        orchestration=registry.resolve("orchestration", scenario.orchestration),
    )
    result["scenario"] = scenario
    return result


def load_scenario(path) -> Scenario:
    """读场景文件 (.toml 或 .json). workload.file 的相对路径以场景文件所在目录为基准."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = _toml().loads(text)
    try:
        return Scenario.from_dict(data, base_dir=path.parent)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def _toml():
    try:
        import tomllib
        return tomllib
    except ImportError:          # Python 3.10
        try:
            import tomli
            return tomli
        except ImportError:
            raise ImportError("Python 3.10 读 TOML 需要 tomli: pip install tomli") from None


# ---------------------------------------------------------------------------
# 字典 ↔ 对象
# ---------------------------------------------------------------------------

# 值为对象的字段: (所属类, 字段名) → 字段类
_NESTED = {
    (Scenario, "workload"): Workload,
    (Scenario, "kernel"): KernelConfig,
    (Scenario, "policy"): InstancePolicy,
    (Scenario, "options"): ModelOptions,
    (Scenario, "calibration"): Calibration,
    (InstancePolicy, "wave_offsets"): StageWaveOffsets,
    (ModelOptions, "pipeline"): PipelineConstraints,
    (ModelOptions, "engine_queue_depths"): EngineQueueDepths,
    (PipelineConstraints, "sync"): SyncLatency,
    (PipelineConstraints, "buffers"): BufferSlots,
    (PipelineConstraints, "queues"): QueueDepths,
    (PipelineConstraints, "phases"): PhaseRates,
    (Calibration, "dispatch"): DispatchMechanisticLatency,
    (Calibration, "urma"): UrmaMechanisticLatency,
}
# 策略字段: 名字 / 表 / 对象, 由 registry 校验
_STRATEGY_FIELDS = {(Scenario, kind) for kind in registry.KINDS}


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _field(cls, key: str, path: str) -> dataclasses.Field:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    if key not in fields:
        raise ValueError(f"{_join(path, key)}: 未知字段{registry.suggest(key, fields)}; "
                         f"{path or '场景'} 可用字段: {', '.join(fields)}")
    return fields[key]


def _default(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


def _convert(cls, key: str, value, path: str):
    """校验并转换一个字段值."""
    f = _field(cls, key, path)
    where = _join(path, key)
    if (cls, key) in _STRATEGY_FIELDS:
        registry.resolve(key, value, where=where)
        return value
    nested = _NESTED.get((cls, key))
    if nested is not None:
        if isinstance(value, dict):
            return _build(nested, value, where)
        if value is None or isinstance(value, nested):
            return value
        raise ValueError(f"{where}: 应为表 (对应 {nested.__name__}), 得到 {value!r}")
    if (cls, key) == (PipelineConstraints, "channels"):
        return _channels(value, where)
    if (cls, key) == (Workload, "counts"):
        return None if value is None else _freeze_counts(value)
    if value is None:
        return None
    want = _leaf_kind(f)
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    ok = {"布尔值": isinstance(value, bool),
          "整数": number and isinstance(value, int),
          "数值": number,
          "字符串": isinstance(value, str)}[want]
    if not ok:
        raise ValueError(f"{where}: 应为{want}, 得到 {value!r}")
    return value


def _leaf_kind(f: dataclasses.Field) -> str:
    """叶子字段收什么值: 先看缺省值的类型, 缺省为 None 时看类型标注."""
    default = _default(f)
    if isinstance(default, bool):
        return "布尔值"
    if isinstance(default, int):
        return "整数"
    if isinstance(default, float):
        return "数值"
    if isinstance(default, str):
        return "字符串"
    annotation = str(f.type)
    for token, kind in (("bool", "布尔值"), ("int", "整数"), ("str", "字符串")):
        if token in annotation:
            return kind
    return "数值"


def _channels(value, where: str) -> Tuple[Channel, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{where}: 应为信道表的列表; 默认信道用顶层 default_channels = true")
    out = []
    for i, item in enumerate(value):
        out.append(item if isinstance(item, Channel)
                   else _build(Channel, item, f"{where}[{i}]"))
    return tuple(out)


def _build(cls, data, path: str):
    if not isinstance(data, Mapping):
        raise ValueError(f"{path or '场景'}: 应为表, 得到 {data!r}")
    kwargs = {key: _convert(cls, key, value, path) for key, value in data.items()}
    return _construct(cls, kwargs, path)


def _construct(cls, kwargs, path: str):
    try:
        return cls(**kwargs)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(f"{path or '场景'}: {exc}") from None


def _set_path(obj, keys, value, path: str):
    key = keys[0]
    cls = type(obj)
    _field(cls, key, path)
    if len(keys) == 1:
        new = _convert(cls, key, value, path)
    else:
        nested = _NESTED.get((cls, key))
        if nested is None:
            raise ValueError(f"{_join(path, key)}: 不是表, 不能再往下取 "
                             f"'{'.'.join(keys[1:])}'")
        child = getattr(obj, key)
        if child is None:
            child = _construct(nested, {}, _join(path, key))   # 以缺省值起步
        new = _set_path(child, keys[1:], value, _join(path, key))
    try:
        return dataclasses.replace(obj, **{key: new})
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(f"{_join(path, key)}: {exc}") from None


def _dump(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _dump(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    return f"<{type(value).__name__}>"


def _dump_changed(obj) -> Dict[str, object]:
    """只导出与缺省值不同的字段. 缺省为 None 的表字段一旦给出就保留 (可为空表)."""
    cls = type(obj)
    out: Dict[str, object] = {}
    for f in dataclasses.fields(cls):
        value = getattr(obj, f.name)
        default = _default(f)
        if (cls, f.name) in _NESTED and value is not None:
            sub = _dump_changed(value)
            if sub or default is None:
                out[f.name] = sub
        elif _dump(value) != _dump(default):
            out[f.name] = _dump(value)
    return out
