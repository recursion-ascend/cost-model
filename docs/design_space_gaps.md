# 设计空间覆盖缺口

本模型的定位: **让算子工程师自行选择不同编排方式, 由模型评估收益**。kernel 实现是
模型的**验证点**, 不是约束 —— 所以"与现有 kernel 不一致"不算缺陷, "某种编排方式
表达不出来"才是缺陷。

本文件记录当前表达不了的编排维度, 按补齐价值排序。已覆盖的维度见 README。

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
   看不到它; 这段时间也不占信道带宽。

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
  直通配对 AIC 的 L1 —— 这条 UB->L1 通路在模型里没有对应的信道
- A8W4 下激活核换到 AIV1 (`runsActivation = GetSubBlockIdx() == 1`), 依赖缺口 2
- A4W4 的激活也是 4bit
