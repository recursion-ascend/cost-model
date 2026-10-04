# 设计空间覆盖缺口

本模型的定位: **让算子工程师自行选择不同编排方式, 由模型评估收益**。kernel 实现是
模型的**验证点**, 不是约束 —— 所以"与现有 kernel 不一致"不算缺陷, "某种编排方式
表达不出来"才是缺陷。

本文件记录当前表达不了的编排维度, 按补齐价值排序。已覆盖的维度见 README。

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

## 缺口 2: 角色分配不可配 (是缺口 3/4/5 的前提)

资源名是硬编码的 f-string, 没有旋钮:

```
gmm1.py:82                    (f"AIC:{core}",)     gmm1
gmm2.py:101,105               (f"AIC:{core}",)     gmm2 各 K 段
activation.py:22              (f"AIV0:{core}",)    激活
base.py:211,215,242           AIC / AIV0 / AIC     共享专家 gmm1 / act / gmm2
comm/mte.py:52,88,175         (f"AIV1:{core}",)    dispatch_call / dispatch / combine
comm/urma.py:78,96,112,226    (f"AIV1:{core}",)    recv / maskscan / localcopy / combine
```

试不了的编排:

- **combine 交给 AIV0**。A8W8 下 AIV0 做完激活就闲着, 而 `GMM2 -> combine` 的同核
  **不是物理约束**: GMM2 写 `gmm2OutGlobal` (GM), combine 用 `Copy(copyGM2UB, ...)`
  从 GM 读, 配对只靠 `gmmToEpilogueFlag[blockJob.jobIndex]` 这个索引约定。
  (对比 `GMM1 -> ACT` 的同核**是物理的**: `CopyCL0c2GmOrUb(..., copyUbToV1)` 走
  L0C->UB 的 Fixpipe 硬件通路, 只在绑定对内存在。)
- **dispatch 分给两个 AIV**
- **A8W4 的角色互换** (激活搬到 AIV1、AIV0 做权重 W4->W8 解压)

改动最小: 把 f-string 换成一张 `stage -> 角色` 的映射表, 建图时查表。

注意缺口 3 的晚绑定**不覆盖**这一条: 晚绑定只改"同一角色池里哪个核做", 换不了角色本身
(`combine` 仍然只能是 AIV1)。`_rewrite_for_late_binding` 里的 `POOLABLE_ROLES` 也是写死的
三个角色。

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

## 缺口 10: 输出落点布局不可选, 散射写没有"跨度"这个量

COMBINE 把每行写到 `(tokenIdx·topK + topkIdx)·n + nLoc` —— 落点由 token 全局编号决定。
这是一种**输出布局选择**: 它让 UNPERMUTE 可以顺序读, 代价是写侧按 token 散射。工程师
可以选别的布局 (例如按专家连续写、UNPERMUTE 侧改成 gather), 用写侧局部性换读侧顺序性。
**模型表达不出这个选择, 也没有"写落点跨度"这个量**, 所以这笔交换评估不了。

这不只是少一个旋钮 —— 跨度是**真的影响时长**的。2026-10-04 核对 20260930 的 run:

| 形状 | 实测单 tile | 模型 | 实测/模型 |
| --- | ---: | ---: | ---: |
| bs36  m= 72 | 5.553 us | 1.20 us | 4.6x |
| bs128 m=256 | 36.751 us | 4.27 us | 8.6x |

关键不是倍数, 是**标度**: m 比 3.56 倍, 实测时长比 **6.62 倍** —— 超线性。按字节计价与按
每行固定开销计价**都是线性的**, 两者都解释不了, 调带宽常数也对不上。说得通的机制就是
跨度: token 数越多, 同一个 tile 的 m 行落点铺得越宽 (bs36 是 36x6 = 216 个槽位,
bs128 是 768 个), 页局部性越差; 每行 512B (n=256, BF16) 本来就远小于高效突发长度。

补齐方向: COMBINE 写侧改成 `f(跨度) + 字节/带宽`, 并把"输出布局"变成可选项 (至少两种:
按 token 散射 / 按专家连续)。定 f 需要**扫 token 数**的 run (固定 m 与 n, 只变 batch) ——
现有三个 run 里 m 与 token 数一起变, 分不开。

在此之前: **COMBINE 在大 batch 上是乐观的**, 量级见上表。凡结论依赖 COMBINE 占比的
(combine 配速、EP 摆放/本地亲和度的收益), 都要记住这一点。

---

## 缺口 11: combine 的"在哪个角色、什么粒度"只有一种

模型只能把 combine 建成**一种**编排: AIV1 上与 GMM2 tile 1:1 配对、同核、紧跟其后。
另一种同样合理的编排表达不出来: **放在另一个向量角色上、逐专家独立跑一遍、挂在整个
GMM2 wave 之后**。两者的取舍很实在 —— 逐 tile 配对让 combine 紧跟计算、延迟低, 但
combine 与 GMM2 抢同一个核对; 逐专家独立一遍可以攒批、写侧跨度更可控, 代价是等整波。

参考实现里这两种恰好**绑在量化模板参数上** (`CombineQuantMode`): NO_QUANT 走前者
(`CombineTokenRange`, `GetSubBlockIdx()==1`), QUANT 走后者 (`ProcessCombineExperts`,
`GetSubBlockIdx()==0`, 见 `mega_moe_wave_a8w8.h:388,532-560`)。**那是那份实现的耦合,
不是物理** —— 数据格式和"combine 跑在哪"没有因果关系, 本模型不该跟着耦合。

所以 `KernelConfig.combine_quant_mode` 现在只管数据格式 (写侧每元素字节), 名实相符;
"combine 跑在哪个角色、什么粒度"缺一个独立的编排旋钮, 它正是缺口 2 (角色分配不可配)
的一个具体用例 —— 补齐缺口 2 时一并给出。

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
