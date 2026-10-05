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
from typing import Dict, List, Mapping, Optional, Tuple

from . import registry
from .api import simulate_routing_counts
from . import guardrails
from .config.hardware import EpilogueOverheads, KernelConfig
from .config.pipeline import parse_tiling
from .config.pipeline import (BufferSlots, PhaseRates, PipelineConstraints,
                              QueueDepths, SyncLatency)
from .config.granularity import resolve_granularity
from .config.platform import resolve_platform
from .config.links import StageLink
from .config.roles import RoleAssignment
from .config.policy import InstancePolicy, StageWaveOffsets
from .costs import (DispatchDataLayout, DispatchMechanisticLatency, PrimitiveCosts,
                    UrmaMechanisticLatency, build_analytical_costs)
from .shape import ModelOptions

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
    # COMBINE: 读回 + 本卡行写走 bw_combine_local, 跨卡行写走 bw_combine_remote
    bw_combine_local: Optional[float] = None
    bw_combine_remote: Optional[float] = None
    # 晚绑定下每取一次活的开销 (原子加/核间同步标志的读改写)。缺省 0 不表示没有代价,
    # 表示本模型没有声称它是多少 —— 见 PrimitiveCosts.late_bind_fetch_us 与 R7。
    late_bind_fetch_us: float = 0.0
    dispatch: DispatchMechanisticLatency = field(default_factory=DispatchMechanisticLatency)
    urma: Optional[UrmaMechanisticLatency] = None


@dataclass(frozen=True)
class TilingSource:
    """kernel tiling 真值的来源 (raw/tiling_rank*.bin).

    给了它就不用在场景文件里手抄 kernel 参数: 形状逐字段核对 (含用 p1/p2 重算
    mGroupsPerWave), 不一致 strict=True 直接报错; adopt=True 时把 kernel 真值
    (行级软流水槽数、路由批大小) 直接采用, 不再靠缺省常数碰巧相等。
    """

    path: str = ""
    strict: bool = True     # 与场景不一致时 raise; False 只把说明放进结果
    adopt: bool = True      # 采用 tiling 里的 kernel 真值


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
    costs: 显式给出的公式容器; 给定时 calibration 不生效.
    """

    workload: Workload
    h: int = 6144
    hidden_dim: int = 4096
    aic_num: int = 28
    #: 平台 (硬件规格档位) 的名字: "950pr" / "950dt"。**缺省空 = 不声称平台**。
    #
    # 给了它会做两件事:
    #   1. 把单核带宽按聚合上界收敛 —— PlatformSpec.gm_bw_per_core 取
    #      min(单核常数, 聚合 HBM / 活跃核数)。一个单核常数在核数足够多时会突破整卡
    #      聚合带宽, 那在物理上不可能。
    #   2. 让带宽下界把聚合 HBM 这条规格算进去 (analysis/bounds.py)。
    #
    # 2026-10-05 之前**这个字段不存在**, 于是第 1 条在场景文件这条日常路径上完全失效
    # (build_costs() 不传 platform)。缺省标定下看不出来 —— BW_L1_GM 51900 x 28 核 =
    # 1.45 TB/s 在 950PR 的 1.60 规格内 —— 但 NZ 布局或更高的带宽标定就会漏过去:
    # 实测给 bw_l1_gm_b_nz=80000 时 28 核合计 2.24 TB/s, 超规格 40%。
    #
    # 缺省不填某个平台, 是因为"用哪张卡"是使用者的事实, 不是模型该替人假定的;
    # 不填时带宽下界只用"单核带宽 x 核数", 不含聚合这条。
    platform: str = ""
    name: str = ""
    # 实例层 profile: 以某一份实现的取值为底 (profiles.PROFILES 的名字), 文件里
    # 显式写出的字段再覆盖它。不给 = 模型缺省 = 最少假设, 不复现任何实现。
    profile: str = ""
    p1_override: int = 0
    p2_override: int = 0
    kernel: KernelConfig = field(default_factory=KernelConfig)
    policy: InstancePolicy = field(default_factory=InstancePolicy)
    options: ModelOptions = field(default_factory=ModelOptions)
    calibration: Calibration = field(default_factory=Calibration)
    tiling: Optional[TilingSource] = None
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
        # [tiling] path 与 workload.file 同规则: 相对场景文件所在目录
        til = data.get("tiling")
        if isinstance(til, dict) and til.get("path") and base_dir is not None:
            til = dict(til)
            til["path"] = str((Path(base_dir) / str(til["path"])).resolve())
            data["tiling"] = til
        if isinstance(wl, dict) and "counts" in wl and "routing" not in wl:
            wl["routing"] = "explicit"
        data["workload"] = wl
        return _build(cls, data, "", base=_profile_base(data.get("profile")))

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

    # ---- tiling 真值 ----

    def tiling_truth(self) -> Dict[str, int]:
        """解析 [tiling] path; 未给则空表."""
        if self.tiling is None or not self.tiling.path:
            return {}
        return parse_tiling(self.tiling.path)

    def check(self) -> Tuple[List[str], List[str]]:
        """全部护栏, 返回 (硬错, 警告).

        硬错两类, 都无歧义:
          - 与显式给出的 tiling 真值矛盾 (场景声称的形状与跑出数据的 kernel 配置不符)。
          - 路由不守恒: 每源 rank 发出行数 != tokens x topk。这是算法事实 (每个 token
            恰好选 topk 个路由专家)。2026-10-05 之前它只是警告, 理由是"tokens 与 counts
            在本模型里是两个独立输入, 测试夹具就故意让它们不一致" —— 那是夹具的方便,
            不是事实: 不守恒时主 stage 按 counts 计、lag 阈值/共享专家/UNPERMUTE 按
            tokens 计, 产出一张看似有效的 DAG。
        """
        errors: List[str] = []
        warnings: List[str] = []
        wl = self.workload
        errors += guardrails.check_routing_conservation(
            wl.routing_counts(), wl.tokens, wl.topk)
        til = self.tiling_truth()
        if til:
            errors += guardrails.check_against_tiling(self, til)
        return errors, warnings

    def build_dispatch_layout(self) -> DispatchDataLayout:
        """dispatch 行布局; adopt 时路由批大小取 tiling 真值."""
        layout = DispatchDataLayout.from_hidden(self.h)
        til = self.tiling_truth()
        if til and self.tiling is not None and self.tiling.adopt:
            items = til.get("dispatchRouteItemsPerBatch") or 0
            if items > 0 and items != layout.route_items_per_batch:
                layout = dataclasses.replace(layout, route_items_per_batch=items)
        return layout

    # ---- 仿真输入 ----

    def platform_spec(self):
        """platform 名字 -> PlatformSpec; 空字符串 -> None (不声称平台)."""
        if not self.platform:
            return None
        return resolve_platform(self.platform)

    def build_costs(self) -> PrimitiveCosts:
        if self.costs is not None:
            return self.costs
        cal = self.calibration
        extra = {}
        if cal.late_bind_fetch_us:
            extra["late_bind_fetch_us"] = cal.late_bind_fetch_us
        # 行级软流水槽数的真值来源, 依次: [tiling] path > pipeline.buffers > 缺省常数
        dispatch = cal.dispatch
        til = self.tiling_truth()
        window = 0
        if til and self.tiling is not None and self.tiling.adopt:
            window = til.get("dispatchBufferCount") or 0
        if window <= 0:
            pipe = self.options.pipeline
            window = pipe.buffers.dispatch_window if pipe is not None else 0
        if window > 0 and int(window) != int(dispatch.buffer_count):
            dispatch = dataclasses.replace(dispatch, buffer_count=int(window))
        return build_analytical_costs(
            platform=self.platform_spec(), active_cores=self.aic_num,
            h=self.h, kernel=self.kernel,
            dispatch_mechanistic=dispatch, urma_mechanistic=cal.urma,
            bw_l1_gm=cal.bw_l1_gm, bw_l1_gm_b_nz=cal.bw_l1_gm_b_nz,
            cube_mac_per_us=cal.cube_mac_per_us,
            gmm1_fill_us=cal.gmm1_fill_us,
            gmm1_tile_restart_us=cal.gmm1_tile_restart_us, bw_ub=cal.bw_ub,
            t_startup_us=cal.t_startup_us,
            bw_combine_local=cal.bw_combine_local,
            bw_combine_remote=cal.bw_combine_remote, **extra)

    def resolved_options(self) -> ModelOptions:
        """信道模型已停用, 不再有需要展开的字段; 保留以稳定调用方接口."""
        return self.options


def simulate(scenario: Scenario, *, platform=None,
             check_bounds: bool = True) -> Dict[str, object]:
    """场景 → 执行时间. 返回 simulate_routing_counts 的结果, 另带 scenario.

    platform / check_bounds 直通 simulate_routing_counts: 前者给聚合 HBM 规格 (让带宽
    下界取 min(每核x核数, 聚合)), 后者决定穿透物理下界时抛异常还是只记录
    (见 analysis/bounds.py 与 api.simulate_routing_counts 的说明)。

    先跑护栏 (guardrails):
      - 路由不守恒: 硬错, 不可降级 (算法事实, 见 Scenario.check)。
      - 给了 [tiling] 时逐字段核对 kernel 真值 (含用 p1/p2 重算 mGroupsPerWave),
        矛盾即报错, tiling.strict=false 可降级为警告。
    全部说明都放进结果的 "warnings"。
    """
    wl = scenario.workload
    conservation = guardrails.check_routing_conservation(
        wl.routing_counts(), wl.tokens, wl.topk)
    if conservation:
        raise ValueError("routing 不守恒 (每个源 rank 应发出 tokens x topk 行):\n  - "
                         + "\n  - ".join(conservation))
    errors, warnings = scenario.check()
    strict = scenario.tiling is None or scenario.tiling.strict
    if errors and strict:
        raise ValueError("与 tiling 真值矛盾 (tiling.strict=false 可降级为警告):\n  - "
                         + "\n  - ".join(errors))
    # platform 以场景文件里声明的为缺省; 显式传参可覆盖 (扫平台时用)
    result = simulate_routing_counts(
        platform=platform if platform is not None else scenario.platform_spec(),
        check_bounds=check_bounds,
        routing_counts=wl.routing_counts(), token_num_per_rank=wl.tokens,
        h=scenario.h, hidden_dim=scenario.hidden_dim, aic_num=scenario.aic_num,
        costs=scenario.build_costs(), topk=wl.topk,
        dispatch_layout=scenario.build_dispatch_layout(),
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
    if errors or warnings:
        result["warnings"] = tuple(errors + warnings)
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
    (Scenario, "tiling"): TilingSource,
    (InstancePolicy, "wave_offsets"): StageWaveOffsets,
    (ModelOptions, "pipeline"): PipelineConstraints,
    (PipelineConstraints, "sync"): SyncLatency,
    (PipelineConstraints, "buffers"): BufferSlots,
    (PipelineConstraints, "queues"): QueueDepths,
    (PipelineConstraints, "phases"): PhaseRates,
    (ModelOptions, "epilogue_overheads"): EpilogueOverheads,
    (Calibration, "dispatch"): DispatchMechanisticLatency,
    (Calibration, "urma"): UrmaMechanisticLatency,
}
# 值为"对象数组"的字段: (所属类, 字段名) → 元素类。场景文件里写 [[options.links]]
_LIST_NESTED = {
    (ModelOptions, "links"): StageLink,
}
# 值为"stage -> 整数"映射的字段: 场景文件里写 [options.granularity] 下 gmm2 = 2。
# 每 stage 一个取值, 所以是表而不是表数组 (links 那种每条边一个对象才用表数组)。
_MAP_NESTED = {
    (ModelOptions, "granularity"): resolve_granularity,
}
# 值为"stage -> 字符串"映射的字段: 场景文件里写 [options.roles] 下 combine = "AIV0"。
# 与 granularity 同形 (每 stage 一个取值), 只是取值是角色名而不是整数。
# 2026-10-05 之前 options.roles 在场景文件/with_overrides 这条日常路径上**根本写不出来**
# (会报"应为数值"), 于是"哪个 stage 跑在哪个核上"这一类编排只能在 Python 里构造对象 ——
# 一个在日常路径上写不出的旋钮等于没有。epilogue_overheads 同病, 它走 _NESTED。
_MAP_STR_NESTED = {
    (ModelOptions, "roles"): lambda v: RoleAssignment(overrides=dict(v or {})),
}
# 值为"字符串元组"的字段: 场景文件里写 late_bind_pools = ["AIC"] 或 barriers = ["wave"]。
# 这两个都是编排旋钮 (晚绑定池 / 分段栅栏), 日常路径是场景文件, 在那儿写不出等于没有。
_SEQ_STR = {
    (ModelOptions, "late_bind_pools"),
    (ModelOptions, "barriers"),
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


def _convert(cls, key: str, value, path: str, base=None):
    """校验并转换一个字段值. base = 该字段的起步取值 (profile), 嵌套表以它为底."""
    f = _field(cls, key, path)
    where = _join(path, key)
    if (cls, key) in _STRATEGY_FIELDS:
        registry.resolve(key, value, where=where)
        return value
    if (cls, key) in _SEQ_STR:
        if value is None:
            return ()
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError(
                f"{where}: 应为字符串数组 (空数组 [] 表示关掉), 得到 {value!r}")
        bad = [v for v in value if not isinstance(v, str)]
        if bad:
            raise ValueError(f"{where}: 每一项应为字符串, 得到 {bad!r}")
        return tuple(value)
    mapper_str = _MAP_STR_NESTED.get((cls, key))
    if mapper_str is not None:
        if value is None:
            return mapper_str({})
        if not isinstance(value, Mapping):
            raise ValueError(f"{where}: 应为表 (stage = \"角色\"), 得到 {value!r}")
        bad = {k: v for k, v in value.items() if not isinstance(v, str)}
        if bad:
            raise ValueError(f"{where}: 每个 stage 的角色应为字符串, 得到 {bad!r}")
        try:
            return mapper_str(value)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
    mapper = _MAP_NESTED.get((cls, key))
    if mapper is not None:
        if value is None:
            return mapper(None)
        if isinstance(value, Mapping):
            bad = {k: v for k, v in value.items()
                   if not (isinstance(v, int) and not isinstance(v, bool))}
            if bad:
                raise ValueError(f"{where}: 每个 stage 的粒度应为整数, 得到 {bad!r}")
        try:
            return mapper(value)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
    item = _LIST_NESTED.get((cls, key))
    if item is not None:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"{where}: 应为表数组 (对应 {item.__name__} 的列表), 得到 {value!r}")
        return tuple(
            v if isinstance(v, item) else _build(item, v, f"{where}[{i}]")
            for i, v in enumerate(value))
    nested = _NESTED.get((cls, key))
    if nested is not None:
        if isinstance(value, dict):
            sub = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)} \
                if isinstance(base, nested) else None
            return _build(nested, value, where, sub)
        if value is None or isinstance(value, nested):
            return value
        raise ValueError(f"{where}: 应为表 (对应 {nested.__name__}), 得到 {value!r}")
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


def _build(cls, data, path: str, base: Optional[Mapping[str, object]] = None):
    """字典 → 对象. base 给出"起步取值" (profile): 文件写了的字段覆盖它.

    嵌套表 (如 [options]) 以 base 里的同名对象为底做 replace, 所以场景文件只需写
    它想改的那几项, 其余沿用 profile 而不是类缺省。
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"{path or '场景'}: 应为表, 得到 {data!r}")
    base = dict(base or {})
    kwargs = {key: _convert(cls, key, value, path, base.get(key))
              for key, value in data.items()}
    for key, val in base.items():
        kwargs.setdefault(key, val)
    return _construct(cls, kwargs, path)


def _profile_base(name) -> Dict[str, object]:
    """profile= 的取值 → 起步字段表; 没给就空表 (纯缺省)."""
    if not name:
        return {}
    from .profiles import resolve_profile
    return dict(resolve_profile(str(name)).scenario_fields())


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
        new = _convert(cls, key, value, path, getattr(obj, key, None))
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


def _dump_changed(obj, base=None) -> Dict[str, object]:
    """只导出与缺省值不同的字段. 缺省为 None 的表字段一旦给出就保留 (可为空表).

    base = 对照的缺省对象。嵌套表要拿**父字段的缺省实例**对照, 不能拿该类自己的缺省:
    ModelOptions.epilogue_overheads 的缺省是 EpilogueOverheads(literal=True), 而
    EpilogueOverheads() 自己的缺省是 literal=False —— 拿类缺省对照, 一个没人动过的
    字段也会被报成"改过"。
    """
    cls = type(obj)
    out: Dict[str, object] = {}
    for f in dataclasses.fields(cls):
        value = getattr(obj, f.name)
        default = getattr(base, f.name) if base is not None else _default(f)
        if (cls, f.name) in _NESTED and value is not None:
            sub_base = default if isinstance(default, type(value)) else None
            sub = _dump_changed(value, sub_base)
            if sub or default is None:
                out[f.name] = sub
        elif _dump(value) != _dump(default):
            out[f.name] = _dump(value)
    return out
