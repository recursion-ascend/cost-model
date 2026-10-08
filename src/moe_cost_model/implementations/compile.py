"""编译期配置与**编译指纹**.

分层里的位置: Workload -> Runtime -> **Compile** -> Lowering -> DAG -> Scheduler。
这一层回答的是"这个二进制是怎么编出来的", 它决定 tile 几何、L1 组织、量化格式、
调度器模板, 但**不**决定 token 怎么路由 (那是 workload) 也不决定每波怎么推进 (runtime)。

为什么要指纹而不是把参数编进名字: kernel 自己的 tiling key 只编码 5 个轴
(mega_moe/op_kernel/arch35/mega_moe_tiling_key.h:33-45: dispatch 量化模式 / dispatch
量化输出类型 / combine 量化输出类型 / 通信模式 / topk 权重类型), 而
TILE_M、TILE_N、L1_BUF_NUM、IsGmm1Interleaved、TOPK_PREFETCH 都在 key 之外, 是纯编译轴
(mega_moe/include/CMakeLists.txt:28-32 的 MEGAMOE_* cache 变量 +
mega_moe_apt.cpp:51-53 的 MEGA_MOE_WEIGHT1_INTERLEAVED)。所以"同一个 tiling key"可以对应
多个不同的二进制 —— 单靠 key 认不出来, 必须另给一个覆盖全部编译轴的指纹。

**指纹只盖编译轴**。形状 (h / hidden_dim / token 数) 与拓扑 (核数 / world) 不进指纹:
它们每次运行都在变, 混进来指纹就失去"同一个二进制"的含义。它们属于
identity.ShapeDomain 与 identity.RuntimeTopology。

当前这一层只是**如实记录**: 取值仍由调用方给 (场景文件的 [kernel] 表或 KernelConfig)。
从 C++/CMake/tiling 自动生成 manifest 是下一步 (步骤 4), 届时 from_kernel_config 会多一个
"与 manifest 核对"的入口; 现在先把指纹这件事做对, 否则第 8 步的标定键没有东西可挂。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

#: 指纹里按这个顺序取字段。显式列出而不是遍历 dataclass, 因为"哪些轴算编译点"是一个
#: 断言, 不该随字段增删悄悄改变: 新增字段必须有人决定它进不进指纹。
FINGERPRINT_AXES: Tuple[str, ...] = (
    "tile_m", "tile_n", "l1_tile_k", "l1_buf_num", "weight_nz",
    "gmm1_interleaved", "topk_weights_prefetch", "activation_n_half",
    "swizzle_offset", "swizzle_direction",
    "combine_quant_mode", "combine_meta_bytes_per_row",
    "dispatch_quant_mode", "dispatch_quant_out_dtype",
    "x_dtype", "weight_dtype", "comm_mode",
)


@dataclass(frozen=True)
class CompileConfig:
    """一个编译点的全部轴. 缺省值 = 仓内 kernel 当前的编译点.

    出处 (每一项都能在仓内指到):
      tile_m / tile_n          MEGAMOE_TILE_M / _TILE_N, mega_moe/include/CMakeLists.txt:29-30,
                               缺省见 common/mega_moe_constants.h:82-87
      l1_tile_k                L1_TILE_K, common/mega_moe_gmm_common.h:30 (无宏可覆盖)
      l1_buf_num               MEGAMOE_L1_BUF_NUM, mega_moe/include/CMakeLists.txt:31,
                               common/mega_moe_gmm_common.h:114-116
      weight_nz                **不是编译轴**: kernel 按 groupedMatmulMode 在运行期选
                               (stage/mega_moe_gmm1_activation.h:1074-1088 两个特化都实例化)。
                               这里保留它是因为模型的公式要知道走哪条带宽, 见下面 runtime_selected。
      gmm1_interleaved         IsGmm1Interleaved <- MEGA_MOE_WEIGHT1_INTERLEAVED,
                               mega_moe_apt.cpp:51-58, 缺省 0
      topk_weights_prefetch    TopkWeightsPrefetch <- MEGAMOE_TOPK_PREFETCH,
                               include/kernel.cpp:24-28, 缺省 0。它有可建模的后果:
                               EPILOGUE_TILE_M = prefetch ? 128 : 256 (mega_moe_arch35.h:161)
      activation_n_half        ACTIVATION_N_HALF = 2, common/mega_moe_constants.h:92 (算法常数)
      swizzle_offset/direction BlockSchedulerSwizzle<3, 0>, common/mega_moe_gmm_common.h:33
      combine_quant_mode       combineQuantMode (运行期 attr, tiling @40; key 值 0/3/4,
                               mega_moe_tiling_key.h:25-27)
      combine_meta_bytes_per_row  META_INFO_SIZE(8) x int32 = 32B
                               (common/mega_moe_constants.h:73);
                               模型缺省 16 是"四个具名字段"的下界口径, 见 KernelConfig
      dispatch_quant_mode      DISPATCH_QUANT_MODE_MXFP = 4 (mega_moe_tiling_key.h:21)
      dispatch_quant_out_dtype 3=E5M2 / 4=E4M3FN / 5=E2M1 (mega_moe_tiling_key.h:22-24)
      x_dtype / weight_dtype   mega_moe_apt.cpp:102-111 限定 X=Y=BF16, weight1 ∈ {E5M2,E4M3FN,E2M1}
      comm_mode                TILINGKEY_TPL_MTE=0 / URMA=1 (mega_moe_tiling_key.h:28-29)
    """

    tile_m: int = 256
    tile_n: int = 256
    l1_tile_k: int = 256
    l1_buf_num: int = 2
    weight_nz: bool = False
    gmm1_interleaved: bool = False
    topk_weights_prefetch: bool = False
    activation_n_half: int = 2
    swizzle_offset: int = 3
    swizzle_direction: int = 0
    combine_quant_mode: int = 0
    combine_meta_bytes_per_row: int = 16
    dispatch_quant_mode: int = 4
    dispatch_quant_out_dtype: int = 4
    x_dtype: str = "bf16"
    weight_dtype: str = "fp8_e4m3fn"
    comm_mode: str = "mte"
    #: 哪些轴实际上是**运行期**选的, 只是模型按编译轴记。记下来免得有人以为换它要重编。
    #: weight_nz: kernel 两个特化都在二进制里, 按 groupedMatmulMode 选。
    #: combine_quant_mode: 运行期 attr。
    runtime_selected: Tuple[str, ...] = ("weight_nz", "combine_quant_mode")
    #: 指纹外的记录项: 这个编译点是从哪儿读出来的 (manifest 路径 / tiling 文件 / "手写")。
    provenance: str = "hand-written"

    def axes(self) -> Dict[str, Any]:
        """进指纹的轴, 按 FINGERPRINT_AXES 的顺序."""
        return {name: getattr(self, name) for name in FINGERPRINT_AXES}

    @property
    def fingerprint(self) -> str:
        """编译指纹: 全部编译轴的规范 JSON 的 sha256 前 16 位.

        规范化用排序后的键与紧凑分隔符, 所以同一组取值在任何 Python 版本上同一个值;
        bool 与 int 在 JSON 里区分 (true vs 1), 不会把 l1_buf_num=1 与 weight_nz=True 混同。
        """
        blob = json.dumps(self.axes(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        """一行人读: 指纹 + 偏离缺省的轴 (全缺省时说"仓内缺省编译点")."""
        base = CompileConfig()
        diff = {k: v for k, v in self.axes().items() if v != getattr(base, k)}
        tail = ", ".join(f"{k}={v}" for k, v in sorted(diff.items())) or "仓内缺省编译点"
        return f"{self.fingerprint} ({tail})"

    @classmethod
    def from_kernel_config(cls, kernel, *, comm_mode: Optional[str] = None,
                           provenance: str = "KernelConfig") -> "CompileConfig":
        """从现有的 KernelConfig 取编译轴.

        KernelConfig 混着编译轴与建模参数 (gmm1_b_reuse_frac 是后者, l1_size 是硬件),
        所以这里**只取**编译轴, 不是逐字段搬。topo_urma 在 KernelConfig 里是布尔, 在
        kernel 里是 tiling key 的一个轴, 映射成 comm_mode 的两个取值。
        """
        if kernel is None:
            return cls(provenance=provenance)
        urma = bool(getattr(kernel, "topo_urma", False))
        return cls(
            tile_m=int(getattr(kernel, "tile_m", 256)),
            tile_n=int(getattr(kernel, "tile_n", 256)),
            l1_tile_k=int(getattr(kernel, "l1_tile_k", 256)),
            l1_buf_num=int(getattr(kernel, "l1_buf_num", 2)),
            weight_nz=bool(getattr(kernel, "weight_nz", False)),
            gmm1_interleaved=bool(getattr(kernel, "gmm1_interleaved", False)),
            topk_weights_prefetch=bool(
                getattr(kernel, "topk_weights_prefetch", False)),
            activation_n_half=int(getattr(kernel, "activation_n_half", 2)),
            swizzle_offset=int(getattr(kernel, "swizzle_offset", 3)),
            swizzle_direction=int(getattr(kernel, "swizzle_direction", 0)),
            combine_quant_mode=int(getattr(kernel, "combine_quant_mode", 0)),
            combine_meta_bytes_per_row=int(
                getattr(kernel, "combine_meta_bytes_per_row", 16)),
            comm_mode=comm_mode or ("urma" if urma else "mte"),
            provenance=provenance,
        )
