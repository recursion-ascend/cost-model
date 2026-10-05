"""仓内两份 MegaMoE 实现的适配器.

  ascend950.megamoe.a8w8_wave.v1   mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h
                                   (MTE 通信路径; 建图器 builders/mte.py)
  ascend950.megamoe.layered.v1     mega_moe/op_kernel/arch35/mega_moe_layered.h
                                   (URMA Layered 路径; 建图器 builders/layered.py)

**行为冻结**: 两个适配器的 plan/lower 逐行复刻原先 model.py 里按 topo_urma 分支的那三处
(m_groups_per_wave / waves / build_events), 没有改任何计算。golden 的判据见 adapter.py。

尚未搬进适配器、仍留在 model.py 与 builders/base.py 的实现专属知识 (下一步的清单):
  DRAIN_STAGES (stage -> 引擎)        builders/base.py:296-301
  尾段链与其五个常数                  builders/base.py:250-291
  ACT 与 GMM1 共位 / dispatch_call    model.py 的 _rewrite_for_late_binding
  onchip act->gmm2 的整组共位         model.py 的 _apply_onchip_act_to_gmm2
  gmm2 的 K 段命名约定 (.h / .k{j})   builders/gmm2.py
它们都会改事件名或 Event.order, 所以必须各自单独一步 + 单独一次 golden 重生成。
"""
from __future__ import annotations

from typing import Any, List, Tuple

from ..config.hardware import KernelConfig
from ..planning.waves import calc_m_groups_per_wave, plan_layered_waves, plan_waves
from .adapter import Unsupported, WavePlan
from .compile import CompileConfig
from .identity import ImplementationId


class _MegaMoeAdapterBase:
    """两份实现共用的部分: 身份、接受性判据、执行时间的终点 stage."""

    #: 子类给
    ID: ImplementationId = None
    #: 执行时间记到哪个 stage 的最后一个事件结束 (model.completion_event 用)
    END_STAGE = "combine"

    def identity(self) -> ImplementationId:
        return self.ID

    def accepts(self, compile_cfg: CompileConfig, options: Any) -> None:
        """不支持的编译点/编排组合在这里拒绝.

        目前只有一条: TopkWeightsPrefetch=true 没建模。它在 kernel 里有确定的后果
        (EPILOGUE_TILE_M 256 -> 128, mega_moe_arch35.h:161; 外加 maxTilesPerExpert
        的 tiling, op_host/op_tiling/arch35/mega_moe_tiling.cpp:1023-1030), 所以这是
        "能建模但还没建", 不是"物理上不可能" —— 拒绝的措辞要说清这一点。
        """
        if getattr(options, "topk_weights_prefetch", False) or \
                compile_cfg.topk_weights_prefetch:
            raise Unsupported(
                f"{self.ID.key}: TopkWeightsPrefetch=true 未建模。"
                "它在 kernel 里的后果是确定的 (EPILOGUE_TILE_M 256->128, "
                "mega_moe_arch35.h:161), 但本模型还没把 epilogue tile 几何参数化; "
                "要评估它需要先把那条路补上, 不是改一个旋钮")

    def measured_end_stage(self) -> str:
        return self.END_STAGE

    # ---- 子类实现 ----
    def plan(self, shape, compile_cfg, options) -> WavePlan:
        raise NotImplementedError

    def lower(self, shape, plan, costs, options):
        raise NotImplementedError


class A8W8WaveV1(_MegaMoeAdapterBase):
    """A8W8 Wave + MTE 通信 (仓内缺省编译点对应的那份实现)."""

    ID = ImplementationId(
        hardware_id="ascend950",
        implementation_id="megamoe.a8w8_wave",
        variant="v1",
        source_refs=(
            "mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h",
            "mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h",
            "mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h",
            "mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h",
        ),
    )

    def plan(self, shape, compile_cfg, options) -> WavePlan:
        """波宽与波序列 —— 原 model.m_groups_per_wave / model.waves 的非 URMA 分支."""
        if options.m_groups_per_wave > 0:
            mgw = options.m_groups_per_wave            # C4: 直接给波宽
        else:
            p1 = shape.p1_override if shape.p1_override > 0 else 1
            p2 = shape.p2_override if shape.p2_override > 0 else 1
            mgw = calc_m_groups_per_wave(
                hidden_dim=shape.hidden_dim, h=shape.h, aic_num=shape.aic_num,
                p1=p1, p2=p2, tile_n=compile_cfg.tile_n)
        packing = getattr(shape, "wave_packing", None)
        planner = packing.plan if packing is not None else plan_waves
        waves = planner(shape.expert_tokens, mgw, tile_m=compile_cfg.tile_m)
        return WavePlan(m_groups_per_wave=mgw, waves=tuple(waves))

    def lower(self, shape, plan, costs, options):
        from ..builders.mte import MteEventBuilder
        return MteEventBuilder(costs, options).build(shape, list(plan.waves))


class LayeredV1(_MegaMoeAdapterBase):
    """URMA Layered 宏波路径 (mega_moe_layered.h)."""

    ID = ImplementationId(
        hardware_id="ascend950",
        implementation_id="megamoe.layered",
        variant="v1",
        source_refs=(
            "mega_moe/op_kernel/arch35/mega_moe_layered.h",
            "mega_moe/op_kernel/arch35/stage/mega_moe_layered_dispatch.h",
            "mega_moe/op_kernel/arch35/stage/mega_moe_layered_combine.h",
        ),
    )

    def plan(self, shape, compile_cfg, options) -> WavePlan:
        """宏波 = 专家范围, 没有 m-group 波宽概念 —— 原分支返回 0, 这里照旧.

        0 不是"未知"而是"这条路径上这个量不存在"; plan_waves 会拒绝 0 (waves.py 的
        m_groups_per_wave 必须为正), 所以 Layered 必须走 plan_layered_waves, 不能回落。
        """
        waves = plan_layered_waves(shape.expert_tokens, shape.token_num, shape.topk,
                                   tile_m=compile_cfg.tile_m)
        return WavePlan(m_groups_per_wave=0, waves=tuple(waves))

    def lower(self, shape, plan, costs, options):
        from ..builders.layered import LayeredEventBuilder
        return LayeredEventBuilder(costs, options).build(shape, list(plan.waves))


#: 适配器表, 键 = ImplementationId.key。选哪一个由编译点的 comm_mode 决定
#: (对应 kernel 的 TILINGKEY_COMM_MODE, mega_moe_tiling_key.h:28-29)。
ADAPTERS = {
    A8W8WaveV1.ID.key: A8W8WaveV1,
    LayeredV1.ID.key: LayeredV1,
}

#: 旧名字 -> 实现 id。registry._ORCHESTRATION 的 "mte"/"layered" 继续可用。
ALIASES = {
    "mte": A8W8WaveV1.ID.key,
    "layered": LayeredV1.ID.key,
    "a8w8_wave": A8W8WaveV1.ID.key,
}


def adapter_for(compile_cfg: CompileConfig):
    """按编译点选适配器 (comm_mode: mte -> A8W8 Wave, urma -> Layered)."""
    key = LayeredV1.ID.key if compile_cfg.comm_mode == "urma" else A8W8WaveV1.ID.key
    return ADAPTERS[key]()


def resolve(name) -> Any:
    """名字 -> 适配器实例. 接受完整 id、别名、适配器类或实例."""
    if name is None:
        return None
    if isinstance(name, str):
        key = ALIASES.get(name, name)
        if key not in ADAPTERS:
            raise ValueError(
                f"未知实现 {name!r}; 可用: {sorted(ADAPTERS)} "
                f"(别名 {sorted(ALIASES)})")
        return ADAPTERS[key]()
    return name() if isinstance(name, type) else name


class _BuilderShim:
    """把一个**建图器类** (registry 里注册的那种) 当适配器用.

    为什么保留: registry.register("orchestration", name, cls) 的契约是"值是建图器类",
    examples/ 与 tests/ 都按这个契约注册过自定义建图器。适配器接口是新的, 不能让旧注册
    失效 —— 所以这里给一个薄壳: 身份标成 custom, 接受性不额外设限 (自定义建图器自己负责),
    波计划沿用 A8W8 Wave 的算法 (原先这些建图器拿到的就是 model.waves 的结果)。
    """

    def __init__(self, builder_cls):
        self._cls = builder_cls
        self._base = A8W8WaveV1()

    def identity(self) -> ImplementationId:
        return ImplementationId(
            hardware_id="ascend950", implementation_id="custom",
            variant=getattr(self._cls, "__name__", "builder").lower().replace("_", ""),
            source_refs=(f"{self._cls.__module__}:{self._cls.__qualname__}",))

    def accepts(self, compile_cfg, options) -> None:
        return None

    def plan(self, shape, compile_cfg, options) -> WavePlan:
        return self._base.plan(shape, compile_cfg, options)

    def lower(self, shape, plan, costs, options):
        return self._cls(costs, options).build(shape, list(plan.waves))

    def measured_end_stage(self) -> str:
        return self._base.measured_end_stage()


class A8W4WaveV1Declared(_MegaMoeAdapterBase):
    """A8W4 动态 Wave (mega_moe_wave_a8w4.h): **已声明, 未建模**.

    为什么要有一个"会拒绝"的适配器 —— 使用者问"这个框架支持 A8W4 吗"时, 三种答案的信息量
    完全不同: 没有这个名字 (看起来像没想过)、有名字但凭空给个数 (最坏)、有名字且说清差哪一步
    (可以照着补)。这里取第三种。

    从源码能确定的 (所以 DAG 的结构部分是可以写的):
      * 独立 kernel 类 MegaMoeA8W4Wave, 7 个模板参数 (没有 IsGmm1Interleaved;
        基类以 false 绑定) —— mega_moe_wave_a8w4.h:24-34
      * 多一个**权重反量化前段**, A8W8 完全没有这一段:
        common/mega_moe_gmm_common.h:106-107 里 BlockPrologue 只在 IsA8W4 时非 void
        (blaze/prologue/block_prologue_mx_fp8fp4.h), 它在 **AIV** 上跑, 三步:
          CopyGmToUb                 4bit 权重 GM->UB   (MTE2)
          WeightAntiQuantComputeNzNk 4bit -> 8bit 展开   (向量)
          CopyWeightToL1             UB->L1             (MTE2/MTE3)
        L1 双缓冲, WEIGHT_L1_DB_OFFSET = 384 KiB (同文件)
      * B 矩阵的分形与布局都不同: C0_SIZE_B = 32 (A8W8 走 AuxGetC0Size<ElementB>),
        LayoutB = Te::ZNLayoutPtn —— common/mega_moe_gmm_common.h:76, 87
      * 角色分工不同: AIV0 跑 prologue、AIV1 跑 combine (mega_moe_wave_a8w4.h:149-150),
        而模型的角色表把 activation 钉在 AIV0、通信钉在 AIV1 (config/roles.py)

    **差的是一个量, 不是一个旋钮**: WeightAntiQuantComputeNzNk 的向量吞吐 (每 µs 能展开多少
    字节权重)。它的地位与 ACT 的 ACT_BYTES_PER_VEC / BW_UB 相同 —— 那两个也是实测+源码计数
    定下来的, 不是算出来的。仓内没有这条路径的任何打点 (data/ 下六个 run 全是 A8W8,
    config.json5 的 dtype 都是 fp8_e5m2), 所以现在给不出。

    按本项目的边界: 改同一个 variant 的参数可以自动出结果; 改了 C++ 控制流 / 同步协议 /
    缓冲复用 / 流水阶段结构, 就必须重新生成实现描述并重新标定 —— A8W4 属于后者。
    """

    ID = ImplementationId(
        hardware_id="ascend950",
        implementation_id="megamoe.a8w4_wave",
        variant="v1",
        source_refs=(
            "mega_moe/op_kernel/arch35/mega_moe_wave_a8w4.h",
            "mega_moe/op_kernel/arch35/blaze/prologue/block_prologue_mx_fp8fp4.h",
            "mega_moe/op_kernel/arch35/common/mega_moe_gmm_common.h",
        ),
    )

    #: 还缺什么才能建图 (机器可读, 供报告与测试引用)
    MISSING = (
        ("weight_antiquant_bytes_per_us",
         "WeightAntiQuantComputeNzNk 的向量吞吐 (4bit->8bit 展开, 每 µs 字节)。"
         "地位同 ACT 的 BW_UB/ACT_BYTES_PER_VEC: 要实测, 算不出来。"
         "仓内没有 A8W4 的打点 (data/ 下六个 run 的 dtype 都是 fp8_e5m2)"),
        ("prologue_stage_in_role_table",
         "角色表要按实现给: A8W4 的 AIV0 跑 prologue、AIV1 跑 combine, 而 config/roles.py "
         "的 DEFAULT_STAGE_ROLES 把 activation 钉在 AIV0、通信钉在 AIV1 —— 这张表现在是全局的"),
    )

    def accepts(self, compile_cfg: CompileConfig, options: Any) -> None:
        missing = "; ".join(f"{name} ({why})" for name, why in self.MISSING)
        raise Unsupported(
            f"{self.ID.key}: 结构已从源码确定 (多一段 AIV 上的权重反量化前段, B 矩阵分形与"
            f"布局不同, 角色分工不同), 但还不能建图, 差: {missing}")

    def plan(self, shape, compile_cfg, options) -> WavePlan:
        self.accepts(compile_cfg, options)

    def lower(self, shape, plan, costs, options):
        self.accepts(CompileConfig(), options)


ADAPTERS[A8W4WaveV1Declared.ID.key] = A8W4WaveV1Declared
ALIASES["a8w4"] = A8W4WaveV1Declared.ID.key
ALIASES["a8w4_wave"] = A8W4WaveV1Declared.ID.key
