# mega_moe_cost_model

MegaMoE A8W8 (Ascend 950 / DAV_3510) 算子物理公式 cost model。
对应 kernel: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h`。

## 理论框架

**资源约束与数据依赖共同驱动的发射机制**：任意操作仅在 (a) 全部前置依赖完成、
(b) 输入数据就绪、(c) 所需计算单元/访存通道/缓冲区/引擎队列存在可用容量时才具备
发射条件；条件满足即立即发射，**不引入软件调度/控制逻辑/指令派发的等待开销**
（理论假设边界）。执行过程 = 依赖关系 + 资源容量 + 硬件时序共同约束的动态调度，
无预设固定执行顺序；计算竞争、访存冲突、流水停顿、依赖等待均由仿真自然呈现。
算子总时长 = 关键依赖路径 × 资源并发能力 × 操作时延 × 动态资源竞争的涌现结果。

| 理论条款 | 实现机制 |
|---|---|
| 前置依赖完成/数据就绪 | DAG 依赖边，仅调度 indegree=0 事件 (`dag.py`) |
| 计算单元容量 | 互斥资源 `AIC:/AIV0:/AIV1:{n}` (`Event.resources`) |
| 访存通道容量 | L2 速率服务器信道：聚合带宽+单流上限+端口 (`Channel`) |
| 缓冲区/引擎队列容量 | 容量令牌 acquire/release (`BUF:*`, `QUEUE:*`) |
| 条件满足即发射 | `dep_latency_us` 默认 0，无发射开销项 |
| 无预设执行顺序 | 就绪集取最早可行启动；程序序仅作为源码导出的依赖边 |
| 吞吐率/流水深度/时延 | 物理公式时长 (字节/带宽, MAC/速率) + L1 相位流水 |
| 完成后释放资源/传播就绪 | end/ledger 归还 + children 就绪更新 |
| 等待与竞争可观测 | 逐事件 dependency/resource/capacity/channel wait 归因 |

**边界声明**：软件调度与指令派发开销（假设为零，属理论边界外）；跨 rank fabric
竞争（各 rank 独立仿真）；核内指令级流水（操作粒度为 tile/stage 级抽象）。

## 项目章程

1. **零拟合**: 一切时长由物理公式推导（字节数÷实测带宽、MAC 数÷实测速率、
   流水启动常数），不做回归拟合。拟合时代的代码（LinearLatency/GmmTileLatency/
   fit_linear_latency）已删除。
2. **结构常数溯源**: TILE_M/TILE_N/ACTIVATION_N_HALF/schedulerN=hiddenDim/2、
   wave 规划（plan_waves）等与 kernel 源码逐行对应，出处写在注释里。
3. **硬件参数溯源**: 每个带宽/延迟常数注明测量方法（如 "H 扫描差分"）。
4. **版本走 git**: 模块名不带版本号；模型迭代用 git tag + CHANGELOG 追踪。
5. **默认行为冻结**: 任何新机制默认关闭或中性，`tests/test_regression.py`
   锚定的数值（402.335µs 确定性用例）必须逐字节复现。

## 模块地图

| 模块 | 职责 |
|---|---|
| `constants.py` | 硬件实测参数 + 源码结构常数（全部带出处） |
| `waves.py` | host wave 规划（PlanNextExpertTokenRangeInWave 移植） |
| `primitives.py` | `PrimitiveCosts`（五个延迟 callable 必填）+ GMM/ACT/COMBINE 物理公式 |
| `dispatch.py` | Dispatch 机制模型（FCFS 窗口队列 + 逐核 begin offset） |
| `dag.py` | 事件 DAG + 多资源调度器（L0 逐边延迟 / L1 容量令牌 / L2 速率服务器信道） |
| `model.py` | shape/options/cursor + `A8W8WaveCostModel` 事件构建 |
| `phases.py` | L0/L1/L2 约束施加器（同步延迟、队列/缓冲令牌、相位拆分、信道需求） |
| `pipeline.py` | `PipelineConstraints`（L0 同步/L1 队列与槽位/L2 信道/相位速率）+ tiling 解析 |
| `api.py` | `simulate_routing_counts` 外层入口 |

```
tools/
  eval_all_ranks.py   全量对比: 参数全部从 tiling_rank0.bin + config.json5 自动推导
  artifacts.py        prof_rank*.bin 打点流解析
  routing.py          路由生成器 (vendor 自 prof 仓 run_case.py, 双向同步)
tests/
  test_regression.py  默认行为锚点 + 中性约束/无争用信道不变量
  test_pipeline.py    L0/L1/L2 机制测试 (16 项)
```

## 用法

```python
from moe_cost_model import (
    PrimitiveCosts, ModelOptions, simulate_routing_counts,
    AnalyticalGmmCosts, AnalyticalActCosts, AnalyticalCombineCosts,
    DispatchMechanisticLatency, PipelineConstraints, QueueDepths,
    SyncLatency, BufferSlots, default_channels, parse_tiling,
)

costs = PrimitiveCosts(
    dispatch_mechanistic=DispatchMechanisticLatency(begin_offset_us=bo),
    gmm1_tile=AnalyticalGmmCosts().gmm1_tile,
    gmm2_tile=AnalyticalGmmCosts().gmm2_tile,
    activation_tile=AnalyticalActCosts().tile,
    combine_tile=AnalyticalCombineCosts(h=6144).tile,
)
res = simulate_routing_counts(
    routing_counts=C,            # C[dst][expert][src] 行数
    token_num_per_rank=64, h=6144, hidden_dim=4096, aic_num=28,
    costs=costs,
)
```

### L0/L1/L2 流水线约束（可选，默认关闭）

```python
pipeline = PipelineConstraints(
    sync=SyncLatency(gmm1_act_handshake_us=2.0),       # L0: Event/Flag 握手延迟
    buffers=BufferSlots(gmm1_act_ub=2),                 # L1: UB ping-pong 槽位
    queues=QueueDepths(mte_aic=2, cube=2, fix=2, vec=2, mte_aiv=2),  # L1: 引擎队列深度
    phases=PhaseRates(cube_mac_per_us=2.7e7),           # L1: 相位速率(触发 load/cube/fix 拆分)
    channels=default_channels(28, bw_l1_gm=BW_L1_GM, bw_scatter=BW_SCATTER),  # L2
)
res = simulate_routing_counts(..., options=ModelOptions(pipeline=pipeline))
```

| 层 | 建模目标 | 机制 |
|---|---|---|
| L0 | 同步原语、生产者/消费者距离 | 逐边 `dep_latency_overrides`；`gmm1_activation_depth`/`gmm2_combine_credit` 距离参数 |
| L1 | 缓冲槽位、MTE/Cube/Vector/Fix 队列、跨 tile 流水 | 容量令牌（acquire/release 分离）；相位拆分（load∥cube 稳态周期=max）；`BUF:*`/`QUEUE:*` 资源 |
| L2 | HBM/L1/片间带宽与并发 | 速率服务器信道：聚合带宽+单事件速率上限+端口数，先到先得无重定价 |

**关键不变量**（`tests/` 守护）：
- 中性约束 = 默认行为，逐字节一致
- 28 核 × 应得速率 = 聚合带宽 → 无争用时与闭式公式一致
- 计算子临界时相位流水稳态周期 = max(load, cube) = 闭式时长
- 队列更深/槽位更多/带宽更宽 → 总时长单调不增

### 从 tiling 真值构建约束

```python
t = parse_tiling("prof_runs/<run>/raw/tiling_rank0.bin")  # bs/h/hidden/topk/mgw/槽位数
```

评估工具（参数零硬编码，CSV 与 kernel bs 不符时自动重建路由）：

```bash
python tools/eval_all_ranks.py    # 在 prof 仓 prof_runs/ 下自动发现数据集
```

## 输入参数参考

```python
simulate_routing_counts(
    # 工况 (输入规模)
    routing_counts=C,              # C[dst][expert][src] 行数
    token_num_per_rank=64, h=6144, hidden_dim=4096, aic_num=28,
    topk=8, shared_expert_num=1,
    # 实现超参数
    kernel=KernelConfig(           # kernel 编译期参数 (CMake cost-sweep knobs)
        weight_nz=False,          #   权重 GM 布局: Z(线性,标定域) / NZ(分形,需标定 bw)
        tile_m=256, tile_n=256,    #   MEGAMOE_TILE_M / TILE_N
        l1_buf_num=2,              #   MEGAMOE_L1_BUF_NUM (L1 ping-pong)
        topk_weights_prefetch=False,  # MEGAMOE_TOPK_PREFETCH
        topo_urma=False,           #   MEGAMOE_TOPO_URMA (URMA 路径未建模)
        swizzle_offset=3, swizzle_direction=0,   # Blaze 调度器模板参数
        activation_n_half=2, l1_tile_k=256, l1_size=512*1024, aiv_num=0),
    p1_override=4, p2_override=1,  # GMM1/GMM2 每核最少逻辑 tile 数 (mgw 策略)
    options=ModelOptions(          # 运行时行为参数
        gmm1_activation_depth=1,   #   生产者/消费者距离
        gmm2_combine_credit=None, gmm2_lag_threshold=4096,
        gmm2_kl1=None,             #   None=kernel 自适应规则, int=显式 K-窗
        combine_no_quant=True, topk_weights_prefetch=False,
        pipeline=PipelineConstraints(...)),   # L0/L1/L2 (可选)
    costs=PrimitiveCosts(...),     # 物理公式 + 硬件常数
    dispatch_layout=...,           # Dispatch 数据布局
)
```

**覆盖说明**: kernel 侧 CMake cost-sweep knobs (TILE_M/TILE_N/L1_BUF_NUM/TOPO_URMA/
TOPK_PREFETCH)、Blaze 模板参数 (swizzle)、p1/p2、kL1、同步距离、队列/缓冲/信道、
权重布局 Z/NZ 均已 Python 可设 (NZ 需提供分形路径带宽标定值, 缺标定拒绝执行);
**不可设** (算法分支, 非参数): IsGmm1Interleaved 交织路径、A4W4/W4 dtype 系、
layered 内核、actMode 激活函数变体、URMA 通信拓扑 — 独立代码路径, 需分别移植.

## 数据约定

prof 工件（tiling/prof bin/routing CSV）默认在 `<本仓上一级>/prof_runs/`，
即本项目位于 prof 仓库内时开箱即用；独立部署时改 `tools/eval_all_ranks.py`
的 `REPO`。
