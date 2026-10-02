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

---

## 缺口 3: tile->核 的绑定时机只有"建图时"

`core_assignment` 能换策略 (StaticRoundRobin / GreedyLeastBusy / ContiguousBlock), 但
三者都在**建图时**定核。**派发时刻晚绑定**表达不了 —— 即"事件只声明要一个 AIC, 调度器
在派发那一刻选最早空闲的成员"。

这是回收 `avoidable idle` (占 AIC 空闲的 22%~54%, 见 README 的空闲分解) 的手段。
依赖边字面不变 (边按名字解析), 但要改 `engine.py` 的 `res_ready` 三处计算 + 提交路径的
资源绑定 + 队列 token 重映射, 并给 ACT 加"跟随 GMM1 落核"的共位约束 (物理, 见缺口 2)。

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
