# 设计空间覆盖缺口

本模型的定位: **让算子工程师自行选择不同编排方式, 由模型评估收益**。kernel 实现是
模型的**验证点**, 不是约束 —— 所以"与现有 kernel 不一致"不算缺陷, "某种编排方式
表达不出来"才是缺陷。

本文件记录当前表达不了的编排维度, 按补齐价值排序。已覆盖的维度见 README。

**常数的出处分成 spec / algo / impl** (2026-10-04): 原来的 `kernel:` 一个标签盖着硬件容量、
算法定义与某实现的取值三类, 读到它分不出"物理上只能这样"还是"那份实现这么选的"。`impl:`
类的数必须能被参数覆盖 —— 本轮接出两个原先焊死的: Layered 每行元数据字节、URMA flag
轮询窗口 (后者原先是**复制出来的派生值** 2048, 注释声明等于两个常数之积, 但改常数不会
跟着变)。

**stage 边收成一个概念** (2026-10): `gmm2_k_segments` / `act_to_gmm2` /
`InstancePolicy.gmm1_activation_depth` 三个旋钮合并为 `ModelOptions.links`
(一条边一个 `StageLink`, 见 `config/links.py`)。旧名字已移除 —— 留着等于保留两套说法。

**取值的含义不引用实现** (2026-10 分层): 缺省值一律是"最少假设", 某一版实现的取值集中在
`profiles.MEGAMOE_A8W8`。所以下表的"备注"里写 `MEGAMOE_A8W8` 的地方, 意思是"那份实现选了
这个", 不是"缺省是这个"。

## 速查: 哪些编排已经能表达

| 维度 | 旋钮 | 备注 |
| --- | --- | --- |
| wave 打包 | `wave_packing` | SequentialGreedy / LongestExpertFirst / BalancedWaves, 可自定义 |
| wave 容量 | `p1_override` / `p2_override` | 经 `calc_m_groups_per_wave` |
| tile 网格 | `tile_grid` | RowMajorTileGrid (缺省) / SwizzledTileGrid (`MEGAMOE_A8W8`) / SplitRowsTileGrid, 可自定义 |
| tile 几何 | `KernelConfig.tile_m` / `tile_n` | 实测 tile_n 128 -> dag_end -6.6%, tile_m 128 -> +39% |
| GMM1 交织 | `KernelConfig.gmm1_interleaved` | kernel 的 `IsGmm1Interleaved`: n-tile 18->36, 每 tile B 流减半, UB 深度 1->2。**A 流翻倍是模型口径的推论, 未经交织路径实测** |
| swizzle | `KernelConfig.swizzle_offset` / `swizzle_direction` | **仅在每专家多于 1 个 m-group 时有效** (1 组时退化为无效) |
| tile->核 分配 | `core_assignment` | StaticRoundRobin / GreedyLeastBusy / ContiguousBlock |
| tile->核 绑定时机 | `ModelOptions.late_bind_pools` | 派发时 (缺省) / 建图时静态 (`MEGAMOE_A8W8`) —— 缺口 3 已补齐 |
| stage->执行角色 | `ModelOptions.roles` | 哪个 stage 跑 AIC / AIV0 / AIV1 (缺口 2 已补齐) |
| **事件粒度 (五个 stage)** | `ModelOptions.granularity` | 每 stage 一个: 1 (缺省, 最细) / N (攒 N 个单元) / 0 (整片; dispatch 的 0 = routeItemsPerBatch)。缺口 12 已补齐 |
| combine 粒度 (兼容视图) | `ModelOptions.combine_granularity` | per_tile / per_expert —— 是上面那一行的视图, 两边矛盾会报错 |
| dispatch 粒度 (兼容视图) | `ModelOptions.dispatch_rows_per_item` | 同上, 对应 granularity 的 dispatch |
| combine 落点布局 | `ModelOptions.combine_layout` | token_scatter (缺省) / expert_contiguous (缺口 10 结构已补齐; 系数留 0, 见下) |
| ready 集选序 | `scheduling_policy` | EarliestStart / WorkConservingCriticalPath / PriorityByStage |
| stage 波偏移 | `InstancePolicy.wave_offsets` | dispatch 超前波数、GMM2 滞后波数 |
| dispatch 配速 | `ModelOptions.dispatch_pacing` | none (缺省) / per_core (`MEGAMOE_A8W8`) / wave |
| dispatch 分工 | `ModelOptions.dispatch_partition` | pooled (缺省) / precut (`MEGAMOE_A8W8`) |
| stage 边: 就绪粒度 | `StageLink.readiness` | 1 (缺省, 等齐) / 2 (`MEGAMOE_A8W8`) / 0 (逐块) / N |
| stage 边: 落点 | `StageLink.location` | gm (缺省) / onchip (不物化, 代价是共位) |
| stage 边: 片上槽数 | `StageLink.depth` | gmm1→act 缺省 1 (UB 单槽); 0 = 不设限 |
| GMM2 kL1 | `ModelOptions.gmm2_kl1` | 自适应或显式 |
| B 复用 | `KernelConfig.gmm1_b_reuse` | 实测 -16.3% (多 m-group 时); "付几次"的规律未定 |
| 通信路径 | `KernelConfig.topo_urma` | MTE / URMA Layered 两套建图器 |
| 建图器本身 | `MegaMoeShape.orchestration` | 扩展点: 可传自己的建图器类 |
| 相位流水 | `ModelOptions.pipeline` | load/cube/fix 相位拆分 + 每核队列深度 |
| 跨卡搬运串行化 | `ModelOptions.serialize_dispatch_comm` | `DISPATCH_COMM` 独占资源 |

**只有旋钮、但缺省标定下是空操作的**: `KernelConfig.l1_buf_num` (只在
`gmm1_tile_restart_us > 0` 时生效, 缺省 0)、`KernelConfig.weight_nz` (开启需显式给 NZ
带宽, 否则直接报错)、
`KernelConfig.topk_weights_prefetch` (无读者, 硬门查的是 `ModelOptions` 的同名字段)。

---

## 缺口 1: 跨核 K-split (split-K GEMM) — **不值得建, 2026-10-04 用闭式结清**

原本的理由: `location="onchip"` (ACT 不物化) 会把并行度压到 m-group 数 —— 一个 GMM2 tile
要吃整条 K (该 m-group 的全部 18 个 ACT), 不落 GM 就只能和产它的 ACT 同核。跨核 K-split
是唯一的出路: 把 K 拆给多个核, 各算部分和再归约。

**结论: 归约流量恒为物化的 ≈8 倍, 与 K、tile 几何无关。** 所以这个方向在本算子的数据
类型下不用建。

推导 (一个 m-group):
```
物化      = 写 A 一次 + 被 n_g2 个 GMM2 n-tile 各读一次
          = (1 + n_g2) · TILE_M · K_gmm2            字节 (A 是 1B/元素的量化激活)
K-split   = 每个 n-tile 要 n_g1 = K_gmm2/TILE_N 份 fp32 部分和, 写一次读一次
          = n_g2 · n_g1 · TILE_M · TILE_N · 4 · 2   字节
比值      = 8·n_g2 / (1 + n_g2)                     <- TILE_M / TILE_N / K 全部约掉
```

| n_g2 (= h/TILE_N) | 4 | 10 | 20 | 40 | 80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| K-split / 物化 | 6.4x | 7.3x | **7.6x** | 7.8x | 7.9x |

本算子 h=5120, TILE_N=256 -> n_g2=20 -> **7.6 倍**。而且这是物化**最差**的情形 (读 A 完全
不命中 L2)。全命中时物化只付"写 1 + 读 1", 比值变成 8·n_g2/2 = **80 倍**。

实测字节 (9216 hidden, 一个 m-group):

| | A 侧 GM 流量 | 并行度上限 |
| --- | ---: | --- |
| 物化 (`location="gm"`) | 写 1.18 + 读 20x1.18 = 24.8 MB | 核数 (28) |
| 片上 (`location="onchip"`) | 0 | m-group 数 G |
| 跨核 K-split | 20x18x0.26x2 = 188.7 MB | min(G x 18, 28) |

为什么 8 这个数躲不开: 部分和是 fp32 (4B) 且要写一次读一次 = 每元素 8B, 而被它替换掉的
A 是 1B/元素。**改变它只有三条路**: 部分和降到 fp16 (比值减半到 3.8 倍, 仍然亏);
在片上归约而不过 GM (那就不是跨核了); 或者 GMM2 的 N 轴小到 n_g2 接近 1 (h 极小)。

顺带回答"片上不物化到底什么时候划得来": 它的代价是并行度 G 而不是流量, 所以
**G >= 核数时才不亏**。实测 9216/3 专家 (G=3, 28 核) 片上版 1551.78 us vs 物化 196.92 us
= 7.9 倍, 与并行度比 28/3 = 9.3 同量级 —— 慢在没活干的 25 个核上, 不在带宽上。

## 缺口 2: 角色分配不可配 — **已补齐 (2026-10-04)**, 但奖品不在这里

原缺口: 资源名是建图代码里写死的 f-string (`f"AIV1:{core}"`), 所以"换个角色干这件事"
问不出来。现在是 `ModelOptions.roles` (`config/roles.py` 的 `RoleAssignment`), 建图器一律
走 `options.role_resource(stage, core)`。

能表达的编排:

```python
RoleAssignment({"combine": "AIV0"})                 # combine 挪到跑 ACT 的那个向量核
RoleAssignment({"activation": "AIV1", "combine": "AIV0",
                "dispatch": "AIV0", "dispatch_call": "AIV0"})   # A8W4 式角色互换
```

物理边界由构造时校验, 不许被映射表改掉:

| | 为什么 |
| --- | --- |
| 矩阵乘只能在 `AIC` | Cube 独有, 没有别处可去 |
| 向量 stage 不能放 `AIC` | 没有物理依据 |
| ACT 必须与它的 GMM1 **同核** | L0C->UB 的 Fixpipe 只在绑定对内 (这条在 `StageLink.colocated_by_hardware`, 本模块不碰) |

晚绑定跟着走: "GMM1 入池隐含跑 ACT 的那个角色一起入池"原先写死 AIV0, 现在查映射表。

### 量出来的结论: 重分角色回收不了空闲的向量核

实测 9216/3 专家/28 核 (规格 Cube 速率):

| 角色 | busy 核·us | 利用率 |
| --- | ---: | ---: |
| AIC | 6818.8 | 77.2% |
| AIV0 | 509.2 | **5.8%** |
| AIV1 | 463.8 | **5.2%** |

两个向量核合起来利用率不到 6%, 看着有 8328 核·us 可捡。但**把 combine 挪到 AIV0 墙钟
一点不变** (315.63 -> 315.63): 关键路径在 AIC 上, 在两个向量角色之间挪工作不碰它。

所以"AIV0 闲着"这件事**不能靠重分角色回收** —— 能挪的工作本来就不在关键路径上, 而 AIC
上的矩阵乘没有别处可去。要用上空闲的向量核, 得给它们**新的**工作 (例如跨核 K-split 的
归约、或把 GMM 尾段的一部分搬过去), 那是另一个问题。

旋钮确实是活的 (不是装饰): 把全部向量工作挤到一个角色上会变慢 —— 2048/8 专家/2 核
3861.56 -> 3924.70 (+1.6%), 且挤到 AIV0 与挤到 AIV1 同值 (两个向量角色对称, 合理性校验)。

补齐它顺带解锁: 缺口 11 (combine 换角色/粒度) 的角色那一半、缺口 5 的 A8W4 角色互换。

---

## 缺口 3: tile->核 的绑定时机 — **已补齐 (2026-10)**

原缺口: `core_assignment` 的三种策略 (StaticRoundRobin / GreedyLeastBusy / ContiguousBlock)
都在**建图时**定核, 表达不了"事件只声明要一个 AIC, 调度器在派发那一刻选最早空闲的成员"。

### 现在怎么用

```python
options = m.ModelOptions(late_bind_pools=("AIC", "AIV1"))   # 角色入池, 派发时绑定
res = m.simulate_routing_counts(..., options=options,
                                scheduling_policy=m.WorkConservingCriticalPath())
```

- `"AIC"` 入池隐含 `"AIV0"` 入池: `GMM1 -> ACT` 同核是**物理约束** (L0C->UB Fixpipe 直给
  配对 AIV0), 所以 ACT 用 `Event.colocate_with` 跟着它的 GMM1 落核, 整对一起漂移。
- `"AIV1"` 入池让 dispatch/combine 落任意空闲 AIV1。`GMM2 -> combine` **不是**物理共位
  (GMM2 写 GM, combine 从 GM 读), 故不加约束。
- 每波每核的 dispatch 调用开销改成 `Event.once_per_core`: 由该核**本波第一段 dispatch**
  承担, 不钉核、不加边 (`DispatchMechanisticLatency.t_call_oh_us`, 缺省 0)。
- `WorkConservingCriticalPath` 的排序键是 `(start, -remaining_path_us, order, name)`:
  start 仍排第一位, 所以核不会为等一个更关键但未就绪的事件而空闲 —— 关键路径只在**同样
  能立刻开始**的候选之间定先后。`remaining_path_us` 由调度器反向拓扑算出 (此前
  `CriticalPathFirst` 读的 `downstream_slack` 无人写入, 一直在静默退化成 `EarliestStart`)。

### 效果

6 个形状 (hidden 9216/14336/18432 x 专家 3/6) x 3 种 `dispatch_pacing` 下,
AIC/AIV0/AIV1 的 `avoidable_idle_us` 全为 0; `busy` 与静态绑定逐位相同 (只换"哪个核做")。
墙钟见 README 的空闲分解一节。缺省 `late_bind_pools=()` 保持静态绑定, 44 个 golden
用例逐位一致。

### 残留的保守之处

1. **按核建的边在晚绑定后指向"原核号"**: `gmm1_activation_depth` 产生的 L1 反压边
   (`activation -> gmm1`) 与 `dispatch_pacing="per_core"` 的配速边, 都按建图时的核号连,
   而那个事件可能已落到别的核。这**不违反不变量** (那段等待计入 `forced`), 但墙钟略微
   高估。该类边占总边数 1.4%~3.0%。要彻底解决需要在调度过程中才知道的信息, 即用重构钩子
   表达。
2. **与相位流水不可同用**: 相位拆分后 `.lg/.ld/fix` 自己不持核资源, 只靠带核号的队列 token
   绑核, 晚绑定下无从回填 —— 代码直接抛 `NotImplementedError` 而不是算出一个错数。
3. **一次性开销不参与准入探测**: `once_per_core` 的开销加在事件结束时刻上, 调度器选核时
   看不到它。

---

## 缺口 4: 尾段事件不占任何核

```python
# builders/base.py:264-281
self._event("epilogue.counts_export", (), T_COUNTS_EXPORT_US, ...)   # 资源元组是空的
self._event("epilogue.unpermute",     (), unpermute_bytes / BW_UNPERMUTE_AGG, ...)
```

后果:

- 表达不了"unpermute 分一部分给 AIC" —— 尾段 AIC 空转占空闲的 16%~32%
- `unpermute` 用一个聚合带宽算完, **没有分核结构**; 而 kernel 里是 56 个 AIV 按
  `aivJob_ = {aivCoreIdx_, blockAivNum_}` 分 token

---

## 缺口 5: 量化模式只覆盖 A8W8 主路径

`KernelConfig.combine_quant_mode` 只管 combine 侧 (NO_QUANT / QUANT)。A8W4 / A4W4 的
**主路径没有建模**:

- A8W4 的权重反量化 prologue: AIV0 做 W4->W8 (`ShiftW4ToW8`), 再 `CopyUB2L1Weight8Bit`
  直通配对 AIC 的 L1 —— 这条 UB->L1 通路在模型里没有对应的访存申报
- A8W4 下激活核换到 AIV1 (`runsActivation = GetSubBlockIdx() == 1`), 依赖缺口 2
- A4W4 的激活也是 4bit

---

## 缺口 6 (已补齐, 2026-10-03 C3): 全局栅栏 / 分段式执行

`ModelOptions.barriers`:

| 取值 | 含义 |
| --- | --- |
| `()` (缺省) | 不加栅栏 = 逐核推进 = **融合算子** |
| `("wave",)` | 波间全核对齐 —— 下一波的任何事件都等上一波全做完 |
| `("stage",)` | 波内每个 stage 之后对齐 (dispatch→gmm1→act→gmm2→combine) = 最彻底的**分段式执行** |

实测"融合值多少" (同为 `gmm1_activation_depth=0`, 公平对比):

| 形状 | 融合 | 波间栅栏 | 分段栅栏 |
| --- | ---: | ---: | ---: |
| 9216 / 3 | 226.04 | 254.82 (+12.7%) | 226.38 (+0.2%) |
| 9216 / 6 | 362.57 | 458.86 (+26.6%) | 402.59 (+11.0%) |
| 18432 / 6 | 632.79 | 678.57 (+7.2%) | 716.01 (+13.2%) |

### stage 栅栏与 UB 槽的冲突: 范围是**一个波内**, 而且**有条件**

`("stage",)` 的栅栏是按波建的 (`barrier.w{w}.{stage}`), 所以冲突只在一个波内成立:

    该波第 depth+1 个 GMM1 等 ACT 还 UB 槽
      -> 那个 ACT 等 barrier.w{w}.activation
      -> 那道栅栏等该波**全部** GMM1, 包括第 depth+1 个        => 成环

判据是"**某个核在同一波里的 GMM1 tile 数 > UB 深度**", 不是"只要用了 UB 就不行":

| 配置 (9216/3, 28 核, 每专家每 m-group 18 个 n-tile) | 单核单波最多 tile | 结果 |
| --- | ---: | --- |
| 波宽 1, 深度 1 | 1 | 可行, dag_end 230.58 |
| 波宽 2, 深度 1 | 2 | **死锁** |
| 波宽 2, 深度 2 | 2 | 可行, dag_end 235.81 |
| 波宽 2, 深度 0 | 2 | 可行, dag_end 226.38 |

晚绑定下调度器能在池内摊平, 所以下界取 `ceil(该波 tile 数 / 核数)`。

模型按这个条件判并给出三条出路 (深度 0 / 调小波宽 / 加大深度), 而不是报
"capacity deadlock"。深度 0 的物理依据: 分段式执行里 GMM1 本来就该走 L0C->GM
(kernel 有 `Gmm1AicMmadTileToGmGeneric` vs `...ToUbGeneric` 两条路), UB 不是交接缓冲。

同时把**逐核排空节点**从 84 个 (28 核 x 3 引擎, 复刻 `WAIT_GMM_DRAIN`) 换成**一个**
全核排空栅栏。对尾段完全等价 (尾段本来依赖全部 84 个, 每个又依赖本核该引擎的全部事件,
传递闭包就是"依赖全部 MoE 事件"), 40 个基准用例的 `kernel_total_us` 逐位不变, 事件数
659 -> 576。

### 原文

现在图里只有**每核每引擎**一个排空节点 (`moe_expert_stage_done.{role}.c{core}`,
`builders/base.py:287` 的 `DRAIN_STAGES`), 对应 kernel 的 `WAIT_GMM_DRAIN`。**全核栅栏**
没有对应物 —— kernel 里那两个 `SyncAll` 都在波循环**外面**, 模型按"前导/尾段不入图"的
口径退场了。

试不了的编排:

- **分段式执行**: 像 DeepEP-Ascend 那样把 dispatch / GMM / combine 拆成几个独立 kernel,
  段间全核对齐。这是"融合 vs 不融合"的收益评估, 而本模型的定位正是评估这类选择。
- **波间全核对齐**: 下一波的任何事件都等上一波全部做完 (而不是现在的逐核排空)。

要补: 一个"栅栏事件"原语 —— 零时长、依赖某组事件的全部、且后续某组事件全部依赖它。
`DRAIN_STAGES` 那张表已经是现成的归集机制, 缺的是"把它升格成全核"的旋钮与建图器支持。
注意这会与晚绑定叠加: 栅栏之后所有核重新开始, 晚绑定的收益可能被栅栏吃掉 —— 正是值得
量化的那件事。

---

## 缺口 7: 单 Server 假设 (跨 Server 中继未建模)

URMA Layered 路径显式声明了 `serverNum=1` 的建模边界 (`builders/layered.py:11`): 全部 src
rank 走直连通道, **跨 Server 的一级中继 PUT** (`BuildDispatchRelayQueues` /
`SendDispatchRelayQueues`) 没有对应事件。

后果: 评估不了"超节点内 vs 跨超节点"的编排差异 —— 而 ep 规模一上去, 中继跳数与带宽分层
是主要变量。片间搬运现在只有一个带宽档 (`bw_remote_bytes_per_us`), 没有"同 Server /
跨 Server 两档"的概念。

MTE 路径同理: `DispatchMechanisticLatency` 只分 local / remote 两档, 没有第三档。

同一类的小边界 (都在 `layered.py` 顶部有声明): flag 轮询重试、mask 扫描的标量开销、
`maxOutputSize` 截断、world >= 5 的并发外推。

---

## 缺口 8 (已消解, 2026-10-03): A 流翻倍是相加口径的推论

原缺口: 交织把 n-tile 从 18 变成 36, 而模型按**每 tile 各付一次 A 流**计费, 于是 A 流总量
翻倍, AIC busy +25%, 结论变成"交织更慢"。

搬运口径改成 `max(A流, B流)` 之后这个问题不存在了:

```
非交织: max(A=25.26, B=50.51) = 50.51 /tile x 54 tile  = 2727.5 us
交织:   max(A=25.26, B=25.26) = 25.26 /tile x 108 tile = 2727.5 us
```

tile 数翻倍与每 tile B 流减半正好抵消, **搬运总量守恒**。交织于是成为净收益 (负载更均衡 +
UB 握手深度 1->2), 不再背 A 流的账。

残留的是另一件事: `gmm1_b_reuse` 表达的"B 流在 m-group 之间付几次"仍然待定。不过换成 max
口径后它不再是解释 bs8192 实测偏差的必要条件 (max 口径下 bs8192 只差 -6.1%, 相加口径
+40.8%)。

---

## 口径记录: 搬运 = max(A流, B流), 数据释放不计 (2026-10-03)

两条口径由算子工程师定, 不是从实测反解出来的:

1. **搬运事件 = max(A流, B流)**。A/B 两股并发, 事件时长取较慢的一股。
2. **数据释放事件 (结果 L0C -> GM/UB) 忽略不计**。闭式公式本来就只计搬入;
   相位流水的 fix 相位时长归 0 (节点保留, 仍承载 QUEUE:fix 与归还 L1 缓冲槽的语义),
   于是 `PipelineConstraints.phases.fix_bw_bytes_per_us` 不再影响任何时长。

**与实测的已知冲突** (采用这个口径就等于接受它):

| 实测点 | 实测 | max 口径 | 误差 | 相加口径 | 误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bs36  m= 72, 1 个 m-group | 55.645 | 50.509 | **-9.2%** | 57.61 | +3.5% |
| bs128 m=256, 1 个 m-group | 74.810 | 50.509 | **-32.5%** | 75.76 | +1.3% |
| bs8192 m=256,12 个 m-group | 53.810 | 50.509 | -6.1% | 75.76 | +40.8% |

冲突的根源是: B 流恒大于 A 流时, max 口径的 tile 时长**完全不随 m 变化**。而
bs36 -> bs128 是干净的单变量对比 (只有 m 从 72 变到 256), 实测从 55.645 升到 74.810,
斜率 0.10416 us/行。max 口径预测斜率 0。

反过来 max 口径在 12 个 m-group 的 bs8192 上吻合得好得多。两个口径各自命中一半实测点,
没有哪一个能同时解释三点 —— 要分开只能补一个扫 m 的 run (固定 m-group 数)。

**连带影响**: 访存量申报 (`rank_results["traffic_bytes"]` 的 `gm_to_l1` 一项) 从 A+B 降为
max(A,B) —— 它是按载入相位时长折算的。统计访存量时要记得这一点。

---

## 信道模型已停用 (2026-10-03)

速率服务器那一层整体移除, 只保留 `Event.channel_bytes` 的字节申报 (汇总在
`rank_results["traffic_bytes"]`, 不参与准入、不影响时长)。理由与影响见 README 的
"带宽争用"一节。

**对"核不得在有就绪活时空闲"这条规定的影响**: 准入条件从七条减到六条, 其中 ④ (信道带宽)
这一类虚报随之消失。剩下仍会让 `avoidable_idle_us` 虚报的只有 ③ (计数信号量) 与
⑦ (非核独占资源, 即 `DISPATCH_COMM`)。

---

## C1 (2026-10-03): 事件名不再带核号

原先 503 个事件里 494 个名字带 `.c{核}`, 而依赖边是按名字连的 —— 换一种分核方式, 图的
结构就跟着变。那是在复现 kernel 的记账, 不是建模: 事件的身份应该是"哪一份工作 (哪些
数据)", 核是**调度的产出**。

改完之后 GMM1 / ACT / GMM2 / combine / dispatch 的名字都只含 (波, 专家, 切片, tile),
核号只出现在 `resources` 和调度结果里。(URMA 路径一直就是这样命名的, 这次是让 MTE 路径
对齐。)

**这是纯改名**: 40 个基准用例的 `kernel_total_us` 与各 stage 忙碌时长**全部逐位不变**,
只有含事件名的 `schedule_sha256` 变了。

dispatch 的名字去掉核号后仍然唯一, 因为行区间按核互不重叠 —— (专家, 段, 行区间) 已经
唯一标识一份搬运工作。

剩下 140 条 (7.5%) 边仍指向带核号的事件名, 全在两处:

| 边 | 条数 | 归谁 |
| --- | ---: | --- |
| `dispatch_call -> moe_stage_done` | 56 | C5 (把每波每核的调用开销彻底变成 once_per_core) |
| `moe_stage_done -> epilogue` | 84 | C3 (逐核排空栅栏换成栅栏原语) |

---

## C6 (2026-10-03): dispatch 的行->核分配变成调度决策

原先 `_rotated_balanced_range` 把一个波的行按 kernel 的"均衡 + 轮转"先分给 28 核, 每核
再按 `routeItemsPerBatch` 切批。前半步是 kernel 的记账: 物理事实只是"这些行要被取回来",
谁取哪一行是调度决策。

`ModelOptions.dispatch_partition`:

| 取值 | 切法 |
| --- | --- |
| `"kernel"` (缺省) | 先按核预切, 再切批 —— 复现 kernel, `compare_measured` 用它对齐实测 trace |
| `"rows"` | 不预切, 只按 `dispatch_rows_per_item` 切整个切片; 核号只是轮转占位, AIV1 入池后由调度器决定 |

`ModelOptions.dispatch_rows_per_item` 把工作项的行粒度变成旋钮 (0 = 用 tiling 的
`routeItemsPerBatch`)。实测 9216/3:

| 配置 | 工作项 | 用到的核 | dag_end |
| --- | ---: | ---: | ---: |
| `kernel` | 66 | 28 | 251.81 |
| `rows`, 缺省粒度 (256 行) | 12 | 8 | 257.89 |
| `rows`, 16 行 | 48 | 28 | **249.72** |

缺省粒度下每个 (专家, 源卡) 段就是一项, 只有 12 项喂 28 核 —— 比 kernel 差。切到 16 行
就比 kernel 还快。**粒度本身是个要扫的维度**, 这是预切掩盖掉的东西。

`"kernel"` 模式逐位复现: 40 个基准用例的事件名、起止时刻、落核完全一致 (指纹变化只来自
建图序字段 `order`)。行守恒由建图器原有的 `contributed_rows == required_rows` 校验看着,
`tests/test_dispatch_partition.py` 把四种切法都钉住。

---

## C5 (2026-10-03): 固定开销按"实现残留"对待

分清两类:

| | 性质 | 怎么处理 |
| --- | --- | --- |
| 尾段五项 (counts_export / core_sync / rank_sync / output_init / finalize) | **实现残留** —— 某一版 kernel 的实测耗时, 换一版就变 | `ModelOptions.epilogue_overheads` (`EpilogueOverheads`); 缺省沿用实测常数, `literal=True` 时按字面取 (含 0) 跑"纯物理"基线 |
| 每波每核的 dispatch 调用开销 | **实现残留** | `DispatchMechanisticLatency.t_call_oh_us` (缺省已是 0); `"rows"` 切法下由 `Event.once_per_core` 挂在该核本波第一段 dispatch 上 |
| unpermute | **物理** —— token 数 x topk x h 的字节量 / 带宽 | 不动 |

实测 9216/3: 五项归零后 dag_end 251.807 -> 244.607, 正好少 7.2 = 1 + 2 + 2.2 + 1 + 1。

`once_per_core` 现在在**静态绑定下同样生效** (核号直接取绑定到的资源, 不依赖资源池)。

### 顺带: `"rows"` 切法下一个带核号的事件名都不剩

`dispatch_call` 是唯一剩下的按核事件。它在 `"kernel"` 切法里保留 —— 实测 trace 有
`DISPATCH_SCHEDULE` 包络, `tools/compare_measured.py` 按它对齐。而 `"rows"` 切法里没有
按核的调用结构, 所以不发这个事件, 开销走 `once_per_core`。于是:

```
dispatch_partition="rows", t_call_oh_us=1.006:
  dispatch_call 事件 0 个
  带核号的事件名 0 个          <- C1 的目标在这条路径上完全达成
  once_per_core 计次 44       = 实际搬过数据的 (波, 核) 组合数
```

---

## 缺口 9 (已补齐, 2026-10-04): 相位流水与晚绑定同用

原缺口: 相位拆分 (`ModelOptions.pipeline` 的 `queues.mte_aic > 1`) 把一个 GMM tile 拆成
`.lg/.ld/.cb/fix` 几个相位事件。这些事件**不持核资源** —— 它们代表同一个核里不同引擎
(MTE / Cube / Fixpipe) 的工作, 在时间上重叠, 各自独占核资源就等于没拆。它们"属于哪个核"
靠名字里写死核号的按核计数信号量 (`QUEUE:mte_aic:c7`) 记着, 而晚绑定下核号到派发时刻
才定, 于是回填不了 —— 工作在 3 号核跑、L1 槽从 7 号核扣, 约束悄悄失效 (墙钟偏快)。
原先 `model.py` 直接拒绝两者同用, 于是"精细流水"与"有活不空闲"二选一。

补齐方式: **核组** (`Event.core_group = (组名, 角色)`)。同一个 tile 的几个相位编成一组,
核号由该组**最先派发**的那个事件选定, 同组其余事件跟随。

为什么不能用 `colocate_with`: 它要求锚点**先**绑定, 而相位里先跑的恰恰是不持核资源的
那一相 (`lg`/`ld` 先于 `cb`/`main`)。核组把"谁先到谁决定"写进语义, 不要求锚点先行。

引擎侧三处同口径 (与 `UB:gmm1act:c*` 现在的做法一致):
`_group_members` 给候选核表 / `_candidates` 与 `_tok_cores` 按组收窄 /
`_pick_core` 对不持核资源的事件只看"该核的槽有没有余量" / 提交时组先到先定核。

顺带修一处会误导下游的旧行为: `ScheduledEvent.meta["core"]` 现在一律改写成**真正落到
的核号**。原先晚绑定下它留着建图时的占位核号, 空闲分解、利用率与测试按它分组就会错核
(本次之前已经两次在测试里踩到)。不持核资源的相位事件也因此有了核号来源。

实测 9216/3 专家/28 核:

| 配置 | 执行时间 | 可避免空闲 |
| --- | ---: | --- |
| 静态钉核 + 拆相位 | 220.83 us | AIV1 264.2 核·us |
| 晚绑定 + 不拆相位 | 196.92 us | 全 0 |
| 晚绑定 + 拆相位 | **183.62 us** | 全 0 |

两个能力叠起来比各自单用都快, 且不变量仍然成立。L1 槽容量按真正落到的核计数
(`tests/test_phase_late_binding.py` 逐核核对在飞载入数 <= 深度) —— 这正是回填不了核号
时会悄悄失效的那条约束。

---

## 缺口 10: 输出落点布局 — **结构已补齐 (2026-10-04); 系数应当留 0, 散射机制已被现有数据否掉**

写落点是 `(tokenIdx·topK + topkIdx)·n`, 这是一种**输出布局选择**: 让 UNPERMUTE 顺序读,
代价是写侧按 token 散射。现在两种布局都能表达 (`ModelOptions.combine_layout`):

| 取值 | 跨度 | 换来什么 |
| --- | --- | --- |
| `"token_scatter"` (缺省) | token 数 x topk | UNPERMUTE 顺序读 |
| `"expert_contiguous"` | 本窗行数 (不散开) | 写侧局部性; 读侧改 gather |

布局 -> 跨度 -> 代价这条链通了: 建图时算出 `spread_slots` 并进 meta, 成本侧

    额外 = m · scatter_us_per_row · (spread_slots / m) ** scatter_exponent

### 系数缺省 0 —— 2026-10-04 修订: 不是“等实测”, 是数据不支持非 0

**先前这里写的理由是错的。** 原话是: 实测 bs36 m=72 **5.553us** / bs128 m=256
**36.751us**, m 比 3.56 倍而时长比 6.62 倍, 超线性, 而且“与 exponent ≈ 0.5 相容”。
把 trace 榨到底之后 (payload 解回 (专家, 波次, m 组, N 组), 去掉 pid=0/1 的重复记录),
这个论证站不住:

  * bs128 与 bs8192 **每事件工作量完全相同** (256 行 x 256 列), 而 bs8192 的最快 tile
    **11.3us** 比 bs128 的最快 tile **22.7us 快一倍** —— 偏偏 bs8192 的落点空间宽
    **64 倍** (8192x6 = 49152 槽 vs 128x6 = 768 槽)、密度稀 16 倍。跨度越宽反而越快,
    所以 `spread_slots` 这个量**解释不了**时长。
  * 同一个 bs8192 run 内部, m 组 0/1/2 中位 **11.46us** (各 206 个样本, p10 11.30,
    分布紧到只能是硬速率), m 组 9/10/11 中位 **55-58us** —— 行数、字节、跨度全一样。
    差的是**别的阶段有没有同时在挤带宽**。

所以 COMBINE 时长的主要变化来自**带宽争用**, 不是落点跨度, 也不是 tile 的 m。本模型
按资源独占排程、不建模带宽争用, 因此:

  * tile 公式对标**最快**那条 tile (没被挤住的那条) —— `BW_REMOTE_WRITE` 就是这么从
    31000 (假设) 改到 8600 (实测反扣) 的, 见 `config/hardware.py`;
  * `scatter_exponent` 留 0 **不是保守、不是待办**, 是现有证据不支持非 0。要立起这个
    机制, 得有一个“固定 m 与 n、只扫 token 数、并且把并发压住”的 run (R3), 而且它得
    先推翻上面那两条观察;
  * 结构保留是有意义的: 布局 -> 跨度这条链算得出来并进了 meta, 将来真有证据时只填系数,
    不动建图。

### 两处偏置, 别拿这个旋钮当免费收益

1. 写侧系数不填, 两种布局的时长就完全一样 (只有申报的跨度不同)。按上面的修订这
   **大概是对的**, 不再当作已知欠账。
2. 读侧**完全没建模**: UNPERMUTE 现在是"字节量 / BW_UNPERMUTE_AGG"一个除法
   (`builders/base.py`), 与落点布局无关。所以填了系数之后 `expert_contiguous` 会显得
   单方面变好 —— 那是模型的偏置, 不是结论。要让这笔交换两侧都算得出, UNPERMUTE 也得
   有它自己的跨度项。

---

## 缺口 11: combine 的角色与粒度 — **已补齐 (2026-10-04)**

原缺口: combine 只能是一种编排 —— AIV1 上与 GMM2 tile 1:1 配对同核。两个维度现在都可配,
而且**彼此正交** (参考实现把它们绑在同一个量化模板参数上, 那是它的耦合, 不是物理):

| 维度 | 旋钮 | 取值 |
| --- | --- | --- |
| 跑在哪个角色 | `ModelOptions.roles` | 见缺口 2 |
| 一个事件覆盖多少工作 | `ModelOptions.combine_granularity` | `"per_tile"` (缺省) / `"per_expert"` |

`"per_expert"`: 一个专家切片一个 combine 事件, 等**自己那个切片**全部 GMM2 段做完
(不是等整波 —— 专家 0 的 combine 不等专家 2 的 GMM2)。

### 量出来: 攒批省的字节远不抵丢掉的交错

实测 9216/3 专家/28 核 (规格 Cube 速率, 缺省晚绑定):

| 粒度 | combine 事件数 | 忙碌合计 | 墙钟 |
| --- | ---: | ---: | ---: |
| `per_tile` | 60 | 305.34 us | **315.63 us** |
| `per_expert` | 3 | 303.86 us (−0.5%) | 411.83 us (**+30.5%**) |

省的那 0.5% 是**路由元数据**: `per_tile` 下每个 n-tile 都要把本窗 m 行的元数据读一遍
(读 20 次), `per_expert` 下每行只读一次。丢的是流水交错 —— 3 个 101us 的大事件堆在各自
切片末尾, 而 60 个小事件能与 GMM2 交错。

### 但模型算不出 per_expert 的主要好处

`per_expert` 真正的卖点是**写侧落点跨度更可控** (一次写整片 h 列, 而不是按 n-tile 分 20 次
散射)。跨度不在模型里 (缺口 10), 所以现在这个对比只看得见它的代价。
**不要据上表下"攒批没用"的结论** —— 等缺口 10 补上才有意义。

---

## 缺口 12: 事件粒度只有 combine 有 — **已补齐 (2026-10-04)**

### 毛病出在哪

粒度 (一个事件覆盖多少份该 stage 的自然工作单元) 是**五个 stage 共有**的编排维度。
补齐前它被拆成了五个各自为政、名字都不一样的东西, 而且其中三个根本没有:

| stage | 补齐前由什么定 | 可配? |
| --- | --- | --- |
| dispatch | `dispatch_rows_per_item` + `dispatch_partition` + `dispatch_pacing` | 可配, 但不叫粒度, 三件事混在一起 |
| gmm1 | 恒 = 一个 `(m-group, n-tile)` | **不可配** |
| activation (SwiGLU) | 恒与 GMM1 tile 1:1 | **完全硬编码** |
| gmm2 | 同 gmm1 | **不可配** |
| combine | `combine_granularity` | 可配 |

`combine_granularity` 是缺口 11 的产物 —— 为回答**一个具体问题** (combine 能不能挪到
另一个向量角色、逐专家做一遍) 就地加的专用旋钮。dispatch 那三个更早, 为对齐 trace
加的。GMM1 / ACT / GMM2 的粒度从来没人问过, 所以一直写死。

这是"参数定义"那个毛病的另一种形态: 上一次是**用"等于某实现"定义取值**, 这一次是
**只有被问到的那一个维度才被抽象出来**。后果一样 —— 设计空间的洞在哪取决于提问历史,
不取决于物理。丢掉的比如"一个 ACT 事件处理一个波内多个 m-group 的输出" (用更大的 UB
驻留换更少的同步点), 它在物理上完全合法。

### 还有一处概念混淆: tile 几何 != 事件粒度

  * `KernelConfig.tile_m` / `tile_n` 受 L1/L0C 容量约束 —— **物理**;
  * "一个事件覆盖几个 tile" 是同步点密度 <-> 并行度的交换 —— **纯编排**。

补齐前只有前者, 所以改 `tile_n` 会同时动这两件事, 算子工程师没法分开扫。

### 补法

`config/granularity.py` 的 `StageGranularity` / `GranularityAssignment`, 与
`StageLink` (每 stage 一条边)、`RoleAssignment` (每 stage 一个角色) 平行。
`ModelOptions.granularity` 是唯一真相; `combine_granularity` 与
`dispatch_rows_per_item` 降级为**兼容视图**, `__post_init__` 把两边对齐,
**两边都离开缺省且矛盾就报错** —— 不允许两个真相。

合并规则由物理定, 不是口味 (`builders/tiling.coalesce_tiles`): 只合并**行范围相同、
列范围相邻**的连续项。行不同就不是一个 matmul 输出块, 合并后的矩形会盖住没算的格子;
列不相邻则下游 (`ctx.activation_ready`、GMM2 的 K 段、combine 的字节) 都按**连续区间**
挑依赖, 不连续的并集表达不出来。碰到边界就截断 —— 粒度是**上界**, 不是凑数配额。

粗粒度事件的时长按**成员逐个算再求和**, 不是拿合并后的大 tile 去套公式: 公式里的固定项
与 B 流复用都是按 tile 发生的, 合并只省掉事件之间的同步, 不省每个 tile 的搬运与计算。

### 三条必须写下来的耦合

1. **ACT 的粒度不是自由旋钮。** ACT 必须与产它的 GMM1 同核 (L0C→UB 的 Fixpipe),
   所以 g>1 只在"喂它的 GMM1 tile 既同核又 n 相邻"时才生效。轮转/贪心分核把相邻
   n-tile 散到不同核, 这时 g>1 是**空操作**。实测 (4 核 16 tile 夹具):
   `StaticRoundRobin` 下 act=2 仍是 16 个 ACT 事件, `ContiguousBlock` 下合成 8 个。
   这是物理与分核策略的耦合, 不是 bug —— 但它意味着这个旋钮会**静默无效**,
   所以每个 ACT 事件的 meta 里记了 `gmm1_events_in_event`, 用来看它到底有没有生效。
2. **ACT 粒度 g 要求 UB 槽数 >= g** (或 0 = 不设限): g 个 GMM1 各占一个槽, 要等齐才
   发 ACT, 槽不够直接死锁。`ActBatcher` 构造时就拒绝, 报错里给出三种改法。
3. **combine 的攒批不按核分组。** combine 从 GM 读 GMM2 的输出
   (`StageLink("activation","gmm2").location == "gm"` 之后那一段同理), 与 GMM2 同核
   **不是**物理约束, 所以一个 combine 事件可以吃不同核产的 tile。按核分组会让这个
   旋钮在轮转/晚绑定下静默失效 —— 第一版就踩了这个坑。

### 粗粒度不是免费的, 模型要能算出它变差

一个事件只能落一个核, 项数少于核数就有核闲着。确定性夹具 (28 核) 实测:

| 粒度 | 事件数 (gmm1/act/gmm2/combine) | 墙钟 |
| --- | --- | --- |
| 全 1 (缺省) | 40 / 40 / 240 / 120 | 245.967 |
| gmm1=2 | 21 / 21 / 240 / 120 | 408.812 |
| gmm2=2 | 40 / 40 / 122 / 61 | 355.385 |
| combine=2 | 40 / 40 / 240 / 60 | 269.265 |
| combine=0 (整片) | 40 / 40 / 240 / 4 | 596.465 |

全部变差 —— 这个夹具 tile 数本来就不够填满 28 核 (40 个 GMM1 tile / 28 核),
粗粒度只是把并行度进一步砍掉。

**另一个方向也真实存在**: `examples/scenario_basic.toml` (4 卡 x 64 本地专家, 28 核,
tile 数远多于核数) 上 `gmm2=2` 从 1751.48 快到 **1741.46** —— 省下的同步点这次赚回来了。

所以这不是收益开关, 是一笔交换, 而且**符号随形状翻转**: 必须扫, 不能照搬取值。
两条测试分别钉住两个方向 (`test_coarse_granularity_costs_parallelism_not_just_saves_sync`
与 `test_scenario_file_can_set_granularity_per_stage`)。

---

## 下界与漏账 (2026-10-04)

### 为什么要下界

只会推演的模型**无法证伪自己**: 它吐出的数不管对错都长得一样。`analysis/bounds.py`
把三类事实变成可检查的断言:

| 事实 | 内容 | 与编排的关系 |
| --- | --- | --- |
| 算法事实 | GMM1 乘加 = Σ m_e·h·hidden_dim (hidden_dim = 2I, SwiGLU 的 gate 与 up 都算); GMM2 = Σ m_e·I·h; 必搬字节 = 激活 + 每专家权重至少一次 | **无关**。只由形状决定 |
| 物理事实 | 一次乘加占 Cube 一拍; 一个字节占带宽一次; 依赖链上的事不能并行 | 无关 |
| 硬件事实 | 每核 Cube 速率、每核载入带宽、聚合 HBM、可用核数 (规格值) | 无关 |

    墙钟 >= max(算力下界, 带宽下界, 依赖下界)

带宽下界的速率取 `min(每核带宽 x 核数, 聚合 HBM)` —— 两个都是硬件规格, 谁小谁管。
下界**不是预测**: 换编排它不变, 所以它是用来检查编排结果的尺子。

### 它立刻抓出两处漏账

**漏账 1: GMM2 的权重流进了时长公式却没进字节申报。**
`builders/gmm2.py` 只申报 `a_gm` (激活), 不申报 B 流 (`k2 x cols` 的权重);
GMM1 两条都申报 (`a_bytes + b_bytes`)。于是 (examples/scenario_basic.toml):

    模型申报 gm_to_l1 = 1660.9 MB
    算法必搬          = 2420.1 MB     差 759.2 MB (≈ GMM2 权重 805.3 MB)

**申报量低于算法下界在物理上不可能**, 所以这是漏账, 不是口径差异。

**漏账 2: 相位流水下载入相位不占任何资源。**
信道模型 2026-10-03 停用后 `channel_bytes` 只做申报、不参与准入, 所以拆相位把载入从
Cube 的账上挪走, 却没挪到任何别的账上 —— 512 个载入可以无限并行:

    rank0 墙钟 1221.797us 低于物理下界 1665.368us (26.6%), 被穿透的是 bandwidth 界
    (算力 25.6 / 带宽 1665.4 / 依赖 7.4)

所以"相位流水省 30.24%"**不是收益, 是把搬运算成了免费**。golden 里
`pipeline_large_split` 也穿透 1.4%。

### 顺带纠正两处旧说法

1. `scenario_basic.toml` 这个形状**不是 Cube 绑定, 是带宽绑定**。先前拿
   `AIC busy / 核数 = 1697.12us` 当"算力界", 但 `busy` 含载入时间, 那不是算力。
   按算法事实: 算力下界 25.57us (夹具 Cube 速率 2.7e7), 带宽下界 1665.37us。
2. 可挖空间不是 3.2% 而是 **5.2%** (1751.48 对 1665.37)。那些 ±0.3% 的旋钮抢的是这
   5.2% 里的几个百分点; 1665us 那部分只能靠**少搬字节**降 (换 dtype、提高 L2 复用、
   改物化编排), 编排碰不到。

### 两处都已修掉 (2026-10-05)

**漏账 1 的修法**: `builders/gmm2.py` 补上 B 流字节申报 (`b_gm = Σ k2·cols`), 与 GMM1
对称。修后 scenario_basic 申报 2466.3MB >= 算法必搬 2420.1MB (比值 1.019 —— 高于下界是
对的: 模型按 tile 读权重, 多个 m-group 各读一次)。

**漏账 2 的修法不是口径选择, 是补一条硬件事实**: **一个 AI Core 只有一条 MTE2 管道**
(GM→L1 的搬运单元), 所以同一个核上同时只能有一笔 GM→L1 在飞。原先把两件事当成了一件:

| | 是什么 | 原先 | 现在 |
| --- | --- | --- | --- |
| `queues.mte_aic` / `l1_buf_num` | L1 **缓冲槽数** → 能提前多少发起下一笔 | 被当成载入并发上限 | 仍是计数信号量, 只管发起 |
| MTE2 管道 | 搬运**单元**, 每核一条 → 同时能搬几笔 | **没有建模** | `.ld` 独占 `MTE2:c{core}` |

两处都要占: GMM1 的 `.ld` (`pipeline_expand._expand_gmm1`) 与 GMM2 的载入份额
(`_annotate` 现在拆出前置 `.ld`)。只修 GMM1 不够 —— GMM2 的载入原先整段裹在 AIC 事件里,
等于给每个核**第二条载入管道**, 修完 GMM1 之后墙钟仍穿透 26.6%。
两处都占之后载入并发上限 = 核数, 聚合载入带宽自动不超过 `核数 x BW_L1_GM`,
带宽下界由构造满足, **不需要恢复速率服务器**。

**拆相位与不拆相位要分开**: GMM2 这个前置 `.ld` 只在 `queues.mte_aic > 1` (真的开了
相位流水) 时才拆。深度 1 时整段闭式时长记在 AIC 上, 载入含在其中、本来就被 AIC 独占
串起来, 不会多出并发 —— 而且 `PipelineConstraints()` 这种中性约束必须与不开相位流水
**逐字节一致** (`tests/test_api_smoke.py::test_neutral_pipeline_invariance`)。
第一版漏了这个门, 把中性约束也拆了, 40 个 golden 变了 10 个; 加上门之后只剩 2 个
(真正开相位流水的那两个), 中性约束回到与基线逐位相同。

### 后果: "相位流水省 30%" 是假的

| | 修前 | 修后 |
| --- | --- | --- |
| scenario_basic 相位流水 | 1221.80us (**-30.24%**, 穿透带宽下界 26.6%) | 1748.18us (**-0.19%**) |
| golden `pipeline_large_split` | 1650.076us (穿透 1.4%) | 2226.820us |

这个形状是**带宽绑定**的 (带宽下界 1665.37us vs 算力下界 25.57us), 把载入与计算重叠
不会让载入变快 —— 所以收益接近 0 才是物理上该有的答案。40 个 golden 里 10 个变了,
全是 `pipeline_*` 与 `stealing_pipeline`。

`check_bounds` 现已缺省 **True**: 穿透物理下界直接抛, 给 `False` 可降级为只记录
(`rank_results[i]["bounds"]["violation"]`), 那是排查用的, 不是出结论用的。

---

## 顺带修掉的一个值: combine 的 metaInfo 字节

原先写死 8B/行, 既不是算法下界也不是任何实现的取值 (同一个仓库里 dispatch 侧早就按 32B
算了)。现在是申报参数 `KernelConfig.combine_meta_bytes_per_row`:

| 取值 | 含义 |
| --- | --- |
| 12 | 算法下界 —— combine 只需 route 三项 (dstRankId / tokenIdx / topkIdx) |
| **16** | 缺省 —— 四个具名字段 |
| 32 | 某实现的取值 (`DataCopy` 搬满 `META_INFO_SIZE=8` 个 int32 槽), `MEGAMOE_A8W8` 用它 |

"搬几个字段"是编排选择: 多搬的字段不参与 combine 的计算, 只是跟着 cacheline 走。
