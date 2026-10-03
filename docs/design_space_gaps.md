# 设计空间覆盖缺口

本模型的定位: **让算子工程师自行选择不同编排方式, 由模型评估收益**。kernel 实现是
模型的**验证点**, 不是约束 —— 所以"与现有 kernel 不一致"不算缺陷, "某种编排方式
表达不出来"才是缺陷。

本文件记录当前表达不了的编排维度, 按补齐价值排序。已覆盖的维度见 README。

## 速查: 哪些编排已经能表达

| 维度 | 旋钮 | 备注 |
| --- | --- | --- |
| wave 打包 | `wave_packing` | SequentialGreedy / LongestExpertFirst / BalancedWaves, 可自定义 |
| wave 容量 | `p1_override` / `p2_override` | 经 `calc_m_groups_per_wave` |
| tile 网格 | `tile_grid` | SwizzledTileGrid / SplitRowsTileGrid, 可自定义 |
| tile 几何 | `KernelConfig.tile_m` / `tile_n` | 实测 tile_n 128 -> dag_end -6.6%, tile_m 128 -> +39% |
| GMM1 交织 | `KernelConfig.gmm1_interleaved` | kernel 的 `IsGmm1Interleaved`: n-tile 18->36, 每 tile B 流减半, UB 深度 1->2。**A 流翻倍是模型口径的推论, 未经交织路径实测** |
| swizzle | `KernelConfig.swizzle_offset` / `swizzle_direction` | **仅在每专家多于 1 个 m-group 时有效** (1 组时退化为无效) |
| tile->核 分配 | `core_assignment` | StaticRoundRobin / GreedyLeastBusy / ContiguousBlock |
| tile->核 绑定时机 | `ModelOptions.late_bind_pools` | 建图时 / 派发时 (缺口 3 已补齐) |
| ready 集选序 | `scheduling_policy` | EarliestStart / WorkConservingCriticalPath / PriorityByStage |
| stage 波偏移 | `InstancePolicy.wave_offsets` | dispatch 超前波数、GMM2 滞后波数 |
| dispatch 配速 | `ModelOptions.dispatch_pacing` | per_core / wave / none |
| GMM2 K 分段 | `ModelOptions.gmm2_k_segments` | 2 (kernel) / 0 (逐块) / N |
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

## 缺口 1: 跨核 K-split (split-K GEMM)

### 是什么

GEMM `C[M,N] = A[M,K] × B[K,N]` 里 M/N 是并行轴 (输出元素互相独立), K 是归约轴
(`C[i,j] = Σ_l A[i,l]×B[l,j]`)。现在只切 N; split-K 是**再把 K 切开分给多个核**,
各核算部分和, 最后归约:

```
现在 (只切 N):   核 X 拿全部 K=5120, 算 C[:, n]        -> 直接是结果
split-K = 2:     核 X 算 A[:,0:2560]   × B[0:2560,n]   -> 部分和 P1
                 核 Y 算 A[:,2560:5120]× B[2560:5120,n]-> 部分和 P2
                 归约:  C[:, n] = P1 + P2
```

### 为什么值得做: 它对症"活不够分"

GMM1 的 n-tile 数 = `ceil(hidden_dim/activation_n_half / TILE_N)`。hidden_dim=9216 时
= 18, 而 aic_num = 28 -> **单个专家只够喂 18 个核**。wave0 装 E0+E1 共 36 个 tile 发给
28 核 -> 8 个核拿 2 个、20 个核拿 1 个, 拿 2 个的那批决定墙钟。

收益**不是"更多并行"**(总计算量不变), 而是**负载量化损失变小**。以整 tile 时长为 1,
按 36 个 tile / 28 核算:

| split-K | 任务数 | 每任务时长 | 每核最多 | 墙钟 |
| ---: | ---: | ---: | ---: | ---: |
| 1 (现) | 36 | 1.000 | ceil(36/28)=2 | **2.00** |
| 2 | 72 | 0.500 | ceil(72/28)=3 | 1.50 |
| 3 | 108 | 0.333 | ceil(108/28)=4 | **1.33** |
| 4 | 144 | 0.250 | ceil(144/28)=6 | 1.50 ← 非单调, 退回 |
| 7 | 252 | 0.143 | ceil(252/28)=9 | 1.286 |
| 理想下界 | — | — | 36/28 | 1.286 |

与 `gmm2_k_segments` 的结论同构: **收益主要在前两三步, 过细会因取整损失退化**。

### 代价

1. **归约访存** (模型现在完全没有的概念)。L0C 只有一块, 两个核的部分和不能在 L0C 里
   相加, 必须落 GM 再加。每个 n-tile 多出 `2 × 写(256×256)` + `读(256×256)` + 加法;
   bf16 下单 tile 128 KiB, 写两次读一次 = **384 KiB 额外访存**, 约为单 tile A+B 流
   (2.5 MiB) 的 15%。
2. **A/B 流总量不变**。`18 tile × (A 256×5120 + B 5120×256)` 与
   `36 份 × (A 256×2560 + B 2560×256)` 完全相等 —— split-K 不增加 A/B 访存, 只增加归约。
3. **精度变化**。FP8/MXFP8 下累加顺序改变, 数值结果与现在不同 (功能性差异, 不只是性能)。

### 模型缺什么

```python
# planning/tile_grid.py:33
class Tile:
    row_begin; row_end      # M
    col_begin; col_end      # N
    # 没有 k_begin / k_end  <- tile 无法表达"我只负责 K 的一段"
```

- `TileGrid.plan(stage, rows, cols, kernel)` 只接 rows/cols, 没有 K 参数
- 成本公式按全量 K 计费: `gmm1_tile(t.rows, shape.h, t.cols)`, `shape.h` 永远是整个 5120
- 全库无 `atomic / reduce / partial_sum` 的建模
- 依赖图拓扑不同: 现在同一 (m,n) 的各 K 段**串接** (同核, 累加同一块 L0C);
  split-K 下它们**并行** (不同核), 再汇聚到一个归约事件

### 待定: 归约走哪条路

这决定归约代价怎么建, 目前**无实测依据**, `mega_moe` 里也没有现成实现可参考 (kernel 只切 N):

| 路径 | 代价构成 |
| --- | --- |
| `AtomicAdd` 到 GM | 部分和直接原子加到目标地址; 无额外 workspace, 但有原子操作串行化 |
| workspace + 单独一轮规约 | 各写一份到 workspace, 再起一批 `Add` 事件; 多一轮读写, 无原子争用 |
| L0C 跨核累加 | 若硬件支持 (Ascend 950 是否有此通路未确认) |

建议实现时先按"workspace + 单独一轮规约"(最保守、最容易估), 常数留空待实测填。

---

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

## 缺口 6: 全局栅栏 / 分段式执行表达不了

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
