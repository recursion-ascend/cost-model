# 模型表达了哪些具体选择

逐个编排维度: 它表达什么、物理耦合在哪、取值怎么扫。README 的「使用方式」给命令, 本文给语义。


## stage 边: 一条边三个问题

stage 之间的编排收在 `StageLink` 里 (`config/links.py`), 对**任意**一条生产者→消费者的边
都是同一组问题:

| 字段 | 问的是 | 取值 |
| --- | --- | --- |
| `readiness` | 消费者沿共享轴分几段独立就绪 (段越多开工越早) | `"whole"` 等齐 (缺省) / `N>=2` 均分 N 段 / `"per_chunk"` 最细 (一个自然块一段) / `"first_chunk"` 首块+其余 |
| `segment_sync_us` | 每多一段, 消费者多付多少同步开销 | `0.0` **未标定** (缺省; 不是"量过是零", 所以细分段的收益是上界) / 实测值 |
| `location` | 中间结果放哪 | `"gm"` 物化 (缺省) / `"onchip"` 留片上 (不付 GM 字节, 代价是共位) |
| `depth` | 片上能同时存几块 (计数信号量) | `0` 不设限 / `N` 槽数 |
| `colocated_by_hardware` | 同核是硬件强制还是编排选择 | `gmm1→activation` 为真 (Fixpipe 只在绑定对内), 不让 `location` 去推它 |

```python
from moe_cost_model import ModelOptions, StageLink as L
ModelOptions(links=(
    L("gmm1", "activation", location="onchip", depth=1, colocated_by_hardware=True),
    L("activation", "gmm2", readiness="per_chunk")))       # GMM2 逐 K 块就绪
```

场景文件里写成表数组:

```toml
[[options.links]]
producer = "activation"
consumer = "gmm2"
readiness = "per_chunk"
```

### 哪条边能分段 (写得出就必须买账)

`readiness` 只在"两边共有一条轴、且消费者能沿它增量消费"时有意义。四条边的共享轴逐条写在
`config/links.EDGE_AXES` 里, 校验照着它**拒绝**写不出后果的取值 —— 不静默忽略, 否则在那儿
扫一圈得到的"0 收益"会被当成硬件上也没收益:

| 边 | 共享轴 (自然块) | 能分段? |
| --- | --- | --- |
| `dispatch→gmm1` | token 行 (m-group) | ✗ 一个 GMM1 事件的行落在一个 m-group 内, 一次只吃一个块 |
| `gmm1→activation` | N = GMM1 的输出列 (GMM1 tile) | ✗ 一个 ACT 对一个 GMM1 tile (1:1) |
| `activation→gmm2` | K = GMM2 的归约轴 (kL1 块) | ✓ 由 `builders/gmm2` 消费 |
| `gmm2→combine` | 切片内的 tile (GMM2 tile) | ✗ 这条轴就是 combine 的打包单元 —— 改 `granularity["combine"]` |

`readiness` 与 `granularity` 的分工: **readiness = 消费者多早能开始 (时序), granularity =
一个事件覆盖多少 (打包)**。两者正交只在"轴不同"时成立 —— GMM2 的 `granularity` 沿 M/N 打包
tile, `readiness` 沿 K 分段, 互不干涉; 而 `gmm2→combine` 的共享轴**就是** combine 的打包单元,
在那儿分段等于少打包, 所以那条边上非 `"whole"` 的 `readiness` 直接报错。

## 模型表达了哪些具体选择

### 事件粒度: 每个 stage 共有的一个维度

**粒度与 tile 几何是两件事**:

* `KernelConfig.tile_m` / `tile_n` 受 L1/L0C 容量约束 —— **物理**;
* "一个事件覆盖几个 tile" 是同步点密度 ↔ 并行度的交换 —— **纯编排**。

`ModelOptions.granularity` 每 stage 一个, 与 `links` (每 stage 一条边)、`roles`
(每 stage 一个角色) 平行。`1` = 最细 (缺省); `N` = 攒 N 个单元; `0` = 整片
(dispatch 的 `0` 特殊: 沿用 tiling 的 `routeItemsPerBatch`)。

```toml
[options.granularity]
gmm2 = 2        # 一个 GMM2 事件覆盖 2 个相邻 n-tile
combine = 0     # 一个 combine 事件覆盖整个专家切片
```

`combine_granularity` ("per_tile" / "per_expert") 与 `dispatch_rows_per_item` 是
`granularity` 的两个兼容视图: 只给一边就那一边生效, 两边都离开缺省且矛盾会报错 ——
只允许一个真相。

三条必须知道的物理耦合:

1. **ACT 的粒度 > 1 可能静默无效。** ACT 必须与产它的 GMM1 同核 (L0C→UB 的 Fixpipe),
   所以只在"喂它的 tile 既同核又 n 相邻"时才合并。轮转分核把相邻 n-tile 散到不同核,
   这时它是空操作 —— 查事件 meta 的 `gmm1_events_in_event` 确认它有没有生效。
2. **ACT 粒度 g 要求 UB 槽数 ≥ g** (`StageLink("gmm1","activation").depth`), 否则死锁,
   构造时直接报错。
3. **combine 的合并不按核分组**: combine 从 GM 读 GMM2 的输出, 同核不是物理约束。
   按核分组会让这个参数在轮转/晚绑定下静默失效。

**粗粒度不是收益开关, 符号随形状翻转**: 28 核确定性夹具 (40 个 GMM1 tile, 填不满核) 上
四种粗粒度全部变慢 (`combine=0` 从 245.97 慢到 596.47); 而 `scenario_basic.toml`
(4 卡 × 64 专家, tile 数远多于核数) 上 `gmm2=2` 从 1751.48 快到 1741.46。**必须扫, 不能
照搬取值。**

### 队列深度与执行单元数是两件事

`QueueDepths` 的五个字段是**在飞上限 / 缓冲槽数** —— 能提前多少发起, 由 L1/UB 槽数决定,
是编排选择, 所以它是参数。而**执行单元数**是硬件事实, 每核每种单元恒为 1:

| 队列深度 (参数) | 执行单元 (事实, 恒 1) | 怎么表达 |
| --- | --- | --- |
| `mte_aic` | 一个 AIC 一条 MTE2 (GM→L1) | `MTE2:c{core}` 容量 1 |
| `fix` | 一个 AIC 一条 FixPipe (L0C→UB/GM) | `FIXPIPE:c{core}` 容量 1 (见下) |
| `mte_aiv` | 一个 AIV 一条 MTE (GM↔UB) | `MTE_AIV:{eng}:c{core}` 容量 1 |
| `cube` | — | `.cb` 相位本身独占 AIC 核资源 |
| `vec` | — | ACT/COMBINE 主事件本身独占 AIV 核资源 |

深度恒为 1 时两者重合, 所以缺省下看不出区别。**深度 >1 时若只有队列深度、没有执行单元
约束, 等于给每个核多出几条管道**: 载入可无限并行, 总时长会低于带宽下界。所以执行单元单独
写成**容量 1 的计数信号量**, 而不是独占资源 —— 后者不行是因为晚绑定只改写
`acquires`/`releases` 的核后缀 (`:c7` → `:c*`), 独占资源会把相位固定在建图时的占位核号上。

`FIXPIPE` 现在**量不出来**: 结果写出 (数据释放事件) 按口径忽略不计, fix 相位时长恒为 0。
它是一条预置的一致性检查 —— 不花代价, 口径改了自动生效。`PhaseRates.fix_bw_bytes_per_us` 给了值会
**直接报错**而不是静默无效: 一个声明了却没有读者的参数会静默忽略用户的输入, 比没有这个
参数更糟。golden 里的 `pipeline_fix_phase` 覆盖 fix 相位的事件结构 (占 `QUEUE:fix` 与
`FIXPIPE`), 口径改成计时长时差异会在那里显形。

### 自定义切分方式

`tile_grid` 决定一个专家切片的输出怎么切成 tile，按**行范围 × 列范围**给。写一个 `TileGrid` 子类注册进去，事件 DAG 随之重建，不用动建图源码：

```python
import moe_cost_model as m

class SplitEveryGroup(m.TileGrid):
    """把每个 m-group 的行再切两半, 让更多核参与 GMM1"""
    def plan(self, *, stage, rows, cols, kernel):
        out = []
        for t in m.SwizzledTileGrid().plan(stage=stage, rows=rows, cols=cols, kernel=kernel):
            half = t.row_begin + t.rows // 2
            out.append(m.Tile(t.row_begin, half, t.col_begin, t.col_end))
            out.append(m.Tile(half, t.row_end, t.col_begin, t.col_end))
        return out

m.register("tile_grid", "split_every_group", SplitEveryGroup)
```

场景文件里写 `tile_grid = "split_every_group"` 即可。坐标是切片内相对值，左闭右开：

| 参数 | 含义 |
| --- | --- |
| `stage`  | `"gmm1"` 或 `"gmm2"` |
| `rows`   | 该专家切片的行数 |
| `cols`   | 输出列总数（GMM1 = `ceil(intermediate / 2)`，GMM2 = `h`） |
| 返回     | `Tile(row_begin, row_end, col_begin, col_end)` 列表，顺序即建图序 |

两条约束在建图时校验，违反了直接报错并指出缺口：

- tile 必须无重叠地铺满 `rows × cols`；
- 行范围不得跨越 m-group 边界（`tile_m` 行一组）。组内再切是允许的，这正是让更多核参与的办法。

`orchestration` 决定用哪个建图代码。继承 `MteEventBuilder` 改波主循环，注册后从场景引用；也可以直接写 `"包.模块:类"`。

内置的 `split_rows` 就是组内切行的现成实现：bs=72 / 8 卡的例子里 GMM1 从 27 个 tile（核 27 空闲）变成 54 个（28 核全用上），执行时间 72.543 → 68.641 µs。

### GMM tile 时长公式

| stage | 双缓冲 (`l1_buf_num=2`)      | 单缓冲 (`l1_buf_num=1`)          |
| ----- | ------------------------------ | ---------------------------------- |
| GMM1  | `max(载入, 计算/R_cube)`     | `载入 + 计算/R_cube + restart` |
| GMM2  | `max(载入, 计算/R_cube)`     | `载入 + 计算/R_cube + restart` |

两个 stage 的 `载入 = A流/BW + B流/BW_b` (**两股相加**)。

口径依据 (三个实测点, 把"m"与"每专家 m-group 数"两个混淆变量分开):

| 形状 | m | m-group 数 | 实测/tile | 相加 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| bs36 | 72 | 1 | 55.65 us | **+3.5%** | −9.2% |
| bs128 | 256 | 1 | 74.81 us | **+1.3%** | −32.5% |
| bs8192 | 256 | 12 | 53.81 us | **+0.4%** | −46.6% |

`max` 口径下 B 流 (2.62MB) 恒大于 A 流 (≤1.31MB), 所以 tile 时长**不随 m 变** —— 而
bs36→bs128 只有 m 变 (同样 28 核并发), 实测涨了 34%。bs8192 之所以曾像是支持 `max`,
真因是 B 流复用 (12 个 m-group, 每 tile 只付整份 B 的 56.5%), 由
`KernelConfig.gmm1_b_reuse_frac` 表达。`AnalyticalGmmCosts(load_overlap="max")` 仍可切回
去做对比。详见 `AnalyticalGmmCosts.gmm1_phases` 的口径沿革与 `tests/test_load_convention.py`。

- GMM1 的 A 流 = m·K 字节, B 流 = `wb`·K·cols (`wb` = 非交织 2 / 交织 1); GMM1 计算 = 2·m·cols·K MACs; GMM2 计算 = m·cols·K2 MACs。
- GMM2 载入 = `A流 + B流`, 与 GMM1 同口径。A 流 = m·K2 字节, **只在物化编排
  (`StageLink("activation","gmm2").location="gm"`, 缺省; 原 `ModelOptions.act_to_gmm2`,
  已并入 `links`) 下计**: ACT 把量化激活写回 GM, GMM2 的 A
  再从 GM 读回来 (参考 kernel 就是这样: epilogue 写 `activationQuantDataPtr`, GMM2
  从 `Location::GM` 取同一个指针)。`"onchip"` 编排下 A 留在片上 (硬件有 UB→L1 通路),
  A 流不付 GM 字节, 代价是一个 m-group 的 GMM1/ACT/GMM2 必须共位于一个核 ——
  并行度上限变成 m-group 数。单缓冲下 GMM2 同样加 `restart` (每 kL1 块一次)。
- `R_cube` (`cube_mac_per_us`) **仓里没有标定值**; 字段缺省 0.0, 其后果是计算项
  整个不生效 —— 不给**不报错**, 而载入绑定的 tile 给不给时长相同, 看输出分辨不出来。**规格峰值由 `cube_mac_per_us("fp8")` 给出
  = 1.35e7 MAC/µs** (A8W8 主路径), 出处见下。测试里的 `2.7e7` 是占位值, 且正好是把规格的
  FLOPS 当成 MAC/µs (差一倍) —— 它也等于 MXFP4 的峰值, 两种读法都不是 A8W8 该用的数。

相位流水 (`options.pipeline`, 可选) 与闭式同口径:

| stage | MTE 队列 (L1 缓冲槽) | Cube 队列 |
| ----- | -------------------- | --------- |
| GMM1  | 占                   | 占        |
| GMM2  | 占 (B 权重)          | 占        |

占用与计时是两回事: A 与 B 都经 L1 进 L0, 所以两个 stage 的 tile 都占 L1 缓冲槽; 权重仍在 L1 里占着位置。

- `queues.mte_aic > 1` 时 GMM1 拆相位: tile 内 load 与 cube 并行, 单 tile 时长 = `max(A流, 计算)`; 后一个 tile 的 A 流可在前一个 tile 计算时预取。
- load / cube 相位时长取自 GMM 公式的分解, 载入带宽与 Cube 速率只有公式这一个来源。
- `l1_buf_num = 1` 与 `queues.mte_aic > 1` 互相矛盾, 同时给会报错。
- `gmm1_tile` 换成自定义函数后没有 A 流/计算分解, 拆相位会报错。

### 共享专家

`[workload]` 里设 `shared_expert_num = 1` 打开。事件与依赖:

| 事件 | 占用 | 时长 | 依赖 |
| --- | --- | --- | --- |
| 共享 GMM1 tile | AIC 核 | GMM1 公式 | 无, 从 0 时刻开始 |
| 共享 ACT tile | AIV0 核 | ACT 公式 | 同 tile 的共享 GMM1 |
| 门控 | 无 | 0 | 全部共享 ACT |
| MoE dispatch_call | AIV1 核 | `t_call_oh_us` (缺省 **0**) | 门控 (MTE: 每个波; Layered: 仅首波接收) |
| 共享 GMM2 tile | AIC 核 | GMM2 公式 (纯计算) | 尾段 core_sync + 本 m-group 的全部共享 ACT |
| 汇合 | 无 | 0 | 全部共享 GMM2 tile |
| 尾段 rank_sync | 无 | — | 汇合 |

共享 GMM2 在最后一个 COMBINE 之后, 不计入 `kernel_total_us`, 只影响 `kernel_dag_end_us`。共享专家的 tile 行数取每卡 token 数 (每个 token 都过共享专家)。未覆盖: 共享专家个数只影响 UNPERMUTE 字节, GMM1/ACT/GMM2 只建一遍; 相位流水不处理共享 stage。

### 2. 写新编排循环

不改模型源码，写一个约 50 行的子类，复用以下现成件：

| 现成件     | 位置                                                 | 内容                       |
| ---------- | ---------------------------------------------------- | -------------------------- |
| stage 函数 | `builders/gmm1.py`、`builders/activation.py`、`builders/gmm2.py` | 事件生成，与传输协议无关 |
| 传输后端   | `builders/comm/`                                   | MTE 和 URMA 各一套，可混搭 |
| 共享状态   | `builders/context.py`                              | 跨 stage 传递的五个字典    |
| 尾段与完成 | `builders/base.py`                                 | 尾段链和完成事件           |

能做的实验举例：GMM2 滞后交替的新波循环；两个 stage 之间插入新 stage；MTE 编排配 URMA 接收的混搭传输。

### 工作量怎么切到核上 (MTE 路径)

两级切分: **先按专家, 每个专家内再按 m-group** (`tile_m` 行一组)。块 = (专家, m-group)。

| 阶段 | 一个块再怎么切 | 归属 |
| --- | --- | --- |
| dispatch | 不再切, 整块一次搬运 | **一块归一个 AIV1 核** |
| GMM1 | 按输出列切 tile (`ceil(intermediate / 2 / tile_n)` 个), 可换 `tile_grid` | 每个 tile 一个 AIC 核, 游标轮转 |
| ACT | 一对一跟随 GMM1 tile; `topk_weights_prefetch` 开着时按 128 行块切成两个 (见 `docs/architecture.md` 的 TopkWeightsPrefetch 一节) | 同核的 AIV0 |
| GMM2 | 按输出列切 tile (`ceil(h / tile_n)` 个), 可换 `tile_grid`; 每个 tile 要行范围相交且覆盖整个 K 的 ACT | 每个 tile 一个 AIC 核, 游标轮转 |
| COMBINE | 一对一跟随 GMM2 tile | 同核的 AIV1 |

块号按全局 m-group 序对核数取模决定归属, 与波无关 —— 同一个块落在哪个波, 归属的核都不变。

**块数少于核数时, 多出来的核在 dispatch 阶段没有活。** 块数 = Σ 各专家 `ceil(行数 / tile_m)`。每卡专家少、每专家行数不足 `tile_m` 时 (小 batch) 这一项很小, dispatch 的并行度会成为瓶颈。

### 3. 当前改不了的结构

波粒度只有两种：MTE 路径按 256 行组切波，Layered 路径按专家范围切波。ACT 总与 GMM1 同波。同核程序序没有建成依赖边，靠资源互斥保序。带宽争用不建模（`channel_bytes` 只统计字节，不影响时长）。

要突破这些需要改 `builders/` 或 `scheduler/` 的结构，改完重新对实测校准。

## GMM2 沿 K 维分段就绪 (编排选择, 非物理约束)

GMM2 的 K 就是 GMM1 切分的那个 N 轴 (`k_gmm2 = hidden_dim / activation_n_half`), 所以一个
ACT tile 只产出 GMM2 在 K 上 1/ceil(k/TILE_N) 的部分, GMM2 要累完整个 K 才有结果。**分几段
独立就绪是编排选择**: L0C 本来就沿 kL1 分块累加 (kernel 的 `ProcessTileL1`), 所以让第 j 段
只等覆盖自己 K 范围的 ACT 在物理上可行。

`StageLink("activation", "gmm2", readiness=...)` 给出这条边的分段 (取值与语义在
`config/readiness.py`, 一条边有没有共享轴、能不能分段在 `config/links.EDGE_AXES`):

| `readiness` | 含义 |
| --- | --- |
| `"whole"` (缺省) | 不分段: 等齐覆盖整个 K 的全部 ACT 再开工 —— 最少假设 |
| `N` (>= 2) | 按 kL1 块数**均分** N 段 (块数不足就是每块一段; 除不尽时多的块给前面的段) |
| `"per_chunk"` | 每个 kL1 块各一段, 第 j 段只等第 j 块的 ACT —— 最细 |
| `"first_chunk"` | 首块一段 (只等 1 个 ACT) + 其余一段 (`MEGAMOE_A8W8` 是这一档) |

取值**单调**: 段数 `whole`(1) ≤ `N`(min(N, 块数)) ≤ `per_chunk`(块数)。整数只表示"均分几段",
`0` 与 `1` 都报错: "一段"就是 `"whole"`, 不留两种写法; 而 `0` 若表示"最细", 又会与
`granularity` 的 `0`=整片、`dispatch` 的 `0`=沿用 tiling 三处相反。整数一律表示"均分几段"
—— 要"首块 + 其余"必须写 `"first_chunk"`, 写 `2` 得到的是均分两段。

实测 (ep=5, 每专家 256 行全远端, aic=28, `kl1=256` 即 18 个 kL1 块;
基线那一列是 `"first_chunk"`, 即那份实现的档):

| 形状 | `first_chunk` | 均分 3 段 | 均分 6 段 | `per_chunk` |
| --- | ---: | ---: | ---: | ---: |
| hidden=9216 专家=3 | 311.7 us | −4.1% | −4.6% | −5.2% |
| hidden=9216 专家=6 | 462.6 us | −4.6% | −5.0% | −5.0% |
| hidden=14336 专家=6 | 679.8 us | −1.0% | −1.9% | −2.4% |
| hidden=18432 专家=6 | 855.2 us | −3.0% | −2.8% | −2.5% |

两点结论: **收益主要在"首块+其余"→均分 3 段** (前者是 1/18 + 17/18 的极端不均分, 均分就把
等待链打散了); **过细会退化** (hidden=18432 专家=6 逐块反而比 3 段差, 事件数 240→4320 后
调度器的资源排队成为新瓶颈)。

⚠️ **这些数是上界**: 实现侧每段要多做一次标志等待 (GM 读 + 自旋), 逐块 = 18 次轮询
vs 两段 2 次。这笔开销由 `StageLink.segment_sync_us` 表达, **缺省 0 = 未标定** (不是"量过
是零"), 上表就是缺省下跑的。所以均分 3 段的 −4.1% 比逐块的 −5.2% 更可信 —— 前者只多
1 次轮询。要定它见 `docs/calibration_runs.md` 的 R8 (可从 trace 的 `WAIT_GMM2_INPUT`
打点反解); 它登记在 `analysis/sensitivity.UNCERTAIN_INPUTS` 里。

## 搬运口径: A 流 + B 流 相加, 数据释放不计

GMM tile 的搬运事件时长 = **A流 + B流**。
**数据释放事件 (结果 L0C -> GM/UB) 不计时长**: 闭式公式本就只计搬入, 相位流水的 fix 相位
时长归 0 (节点保留, 仍承载 QUEUE:fix 与归还 L1 缓冲槽的语义), 所以
`PipelineConstraints.phases.fix_bw_bytes_per_us` 不再影响任何时长。

第一条是**从实测定下来的**, 不是口径决定。三个实测点把两个混淆变量分开:

| 实测点 | 实测/tile | 相加 (现行) | max (旧) |
| --- | ---: | ---: | ---: |
| bs36  m= 72,  1 个 m-group, 28 核 | 55.645 | 57.61 (**+3.5%**) | 50.51 (−9.2%) |
| bs128 m=256,  1 个 m-group, 28 核 | 74.810 | 75.76 (**+1.3%**) | 50.51 (−32.5%) |
| bs8192 m=256, 12 个 m-group, 28 核 | 53.810 | 54.00 (**+0.4%**) | 28.75 (−46.6%) |

(bs8192 那一行按 `gmm1_b_reuse_frac=0.53` 算 B 流份额, 见下。)

三条证据:

1. **固定并发核数, 只变 m** —— `max` 预测斜率为 0, 因为 B 流 (2.62MB) 恒大于 A 流 (≤1.31MB):

   | 波 | 并发核 | bs36 (m=72) | bs128 (m=256) |
   | --- | ---: | ---: | ---: |
   | w0 | 28 | 57.12 us | 77.92 us |
   | w1 | 18 | 29.73 us | 55.20 us |

   两对独立比较都随 m 显著变长 (+36% / +86%)。`max` 被否。

2. **固定 m 与并发核数, 只变 m-group 数** (bs8192 曾像是支持 max 的真因):
   bs128 (1 组) 74.81 vs bs8192 (12 组) 53.80 —— 几何完全相同, 便宜 21.0 us。
   这是 **B 流复用** (L2 命中), 不是 max。反解每 tile 只付整份 B 的 56.5%, 按"首个
   m-group 付整份、其余各付 f"算 f ≈ 0.53 (`KernelConfig.gmm1_b_reuse_frac`, 缺省 1.0 =
   不声称有复用, 因为只有这一个点)。

3. **相加口径反解的带宽自洽**: 由 w0 斜率得 A 流 49.2 GB/s, 以 bs36 为截距得 B 流
   54.4 GB/s —— 同量级, 与标定值 `BW_L1_GM` = 51.9 GB/s 一致。一个共用速率同时解释两点。

`AnalyticalGmmCosts(load_overlap="max")` 仍可切回去做对比。顺带一个待办: 按波分开看,
A 流斜率反解出的带宽随并发核数变 (28 核 45.3 GB/s, 18 核 37.0 GB/s), 所以 `BW_L1_GM`
不是一个常数 —— 要定它得扫并发数。这条斜率是目前最干净的标定手柄 (几何与并发都固定,
只有 m 变)。
