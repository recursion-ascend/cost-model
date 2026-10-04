# MegaMoE Cost Model

Ascend NPU MegaMoE 流水编排性能评估工具。算子工程师改参数、改编排，模型算出执行时间，对比判断性能收益，不必逐个上板实测。

## 建模原则: 缺省值不引用任何实现

**没有任何参数的取值用"等于某一版 kernel"来定义自己的含义。** 三层分开:

| 层 | 内容 | 谁定 |
| --- | --- | --- |
| 物理 | 带宽、Cube 速率、L1/UB 容量、tile 几何 | 硬件 |
| 编排 | 切分 / stage 边 (`StageLink`: 就绪粒度、落点、缓冲深度) / 分核 / 栅栏 / 波宽 | 算子工程师要扫的决策变量 |
| 实例 | 以上的一组具体取值 + 某一版实现的实测固定开销 | `profiles.MEGAMOE_A8W8` |

所以**缺省值是"最少假设"**: 不声称实现做了任何特殊的事 —— 消费者等齐生产者、中间结果
落 GM、不预设谁取哪些行、依赖之外不加序边、固定开销 0、tile→核 在派发时刻才定。
要复现 MegaMoe A8W8 那份实现 (与实测 trace 对齐就得这样) 就显式引用 profile:

```toml
profile = "megamoe-a8w8"     # 场景文件: 以那份实现的取值为底, 下面写的字段再覆盖它
```

```python
from moe_cost_model import MEGAMOE_A8W8 as P
simulate_routing_counts(..., **P.shape_kw(), options=P.options)
simulate_routing_counts(..., **P.shape_kw(), options=P.with_options(gmm2_k_segments=0))
```

缺省跑出来的数与那份实现的数不同, 这是信息 (差多少 = 那些编排选择值多少), 不是 bug。

### stage 边: 一条边三个问题

stage 之间的编排收在 `StageLink` 里 (`config/links.py`), 对**任意**一条生产者→消费者的边
都是同一组问题 —— 原先这三件事是三个各自为政、只对一条特定边说话的旋钮:

| 字段 | 问的是 | 取值 |
| --- | --- | --- |
| `readiness` | 消费者要等生产者产出多少才能开工 | `1` 等齐 (缺省) / `0` 最细 (每 L1 块一段) / `N` 均分 |
| `location` | 中间结果放哪 | `"gm"` 物化 (缺省) / `"onchip"` 留片上 (不付 GM 字节, 代价是共位) |
| `depth` | 片上能同时存几块 (计数信号量) | `0` 不设限 / `N` 槽数 |
| `colocated_by_hardware` | 同核是硬件强制还是编排选择 | `gmm1→activation` 为真 (Fixpipe 只在绑定对内), 不让 `location` 去推它 |

```python
from moe_cost_model import ModelOptions, StageLink as L
ModelOptions(links=(
    L("gmm1", "activation", location="onchip", depth=1, colocated_by_hardware=True),
    L("activation", "gmm2", readiness=0)))                 # GMM2 逐 K 块就绪
```

场景文件里写成表数组:

```toml
[[options.links]]
producer = "activation"
consumer = "gmm2"
readiness = 0
```

### 扫一遍, 看每个选择值多少钱

```bash
python examples/run_design_space.py
```

```
方案                       时长        Δ       Δ%  关键路径变化 (stage)     最大等待         访存量差          护栏
基线 (最少假设)           196.92    +0.00    +0.0%  -                      - 0us           -               ok (基线)
GMM2 逐 K 块就绪          196.25    -0.67    -0.3%  gmm2-10.1, act+9.4     - 0us           -               ok
UB 深度 2 (交织路径)      174.19   -22.73   -11.5%  gmm2-22.7              - 0us           -               ok
ACT 不物化 (留片上)      1551.78 +1354.85  +688.0%  gmm1+808, gmm2+386     capacity 160us  gm_to_l1-70.78MB ok
波间全核对齐              261.85   +64.92   +33.0%  gmm1+50.5, gmm2-22.7   capacity 9us    -               ok
静态发牌                  243.56   +46.64   +23.7%  gmm1+50.5, act+9.4     capacity 9us    -               违反 AIV1 264
那份实现                  234.95   +38.03   +19.3%  gmm1+50.5, gmm2-25.3   capacity 9us    -               违反 AIC 10
```

每一行回答的不是"多少 us", 而是: **收益落在关键路径的哪个 stage**、改完卡在什么等待上、
少搬多少字节、以及这个方案下模型自己的不变量守住了没有 ——"违反"那一行的时长偏慢,
收益不可比。`design_space()` / `format_design_space()` 给的就是这张表。

一个缺省带来的直接后果: 缺省晚绑定满足模型的不变量 ——
**决不出现"某 tile 前置依赖已完成、又有核空闲, 它却还在等"** (`avoidable_idle_us == 0`)。
静态发牌做不到, 它是一种实现的分核方式, 要评估就显式给 `late_bind_pools=()`。

## 当前能做什么

### 1. 改参数直接评估

一个场景文件承载全部旋钮, 改一个旋钮跑一次, 对比差值就是收益或代价。

```bash
python examples/run_scenario.py
```

场景文件 (`examples/scenario_basic.toml`) 的表名与字段名就是对象属性名:

```toml
h = 6144
hidden_dim = 4096
aic_num = 28
p1_override = 2
p2_override = 1

[workload]
tokens = 64
topk = 8
world = 4
local_experts = 64
routing = "uniform"          # uniform | cyclic | random | explicit | file

[calibration]
cube_mac_per_us = 2.7e7      # 必填, 无缺省; 示例值, 换成实测 Cube 速率

[policy]
dispatch_lookahead = 2
```

```python
from moe_cost_model import load_scenario, simulate

base = load_scenario("examples/scenario_basic.toml")
variant = base.with_overrides({"policy.gmm2_lag_waves": 2, "kernel.tile_n": 128})

for sc in (base, variant):
    res = simulate(sc)
    print(sc.to_dict(defaults=False), res["kernel_total_us"])
```

也可以不用文件, 直接在 Python 里构造:

```python
from moe_cost_model import Calibration, InstancePolicy, Scenario, Workload, simulate

sc = Scenario(
    workload=Workload(tokens=64, world=4, local_experts=64, routing="uniform"),
    p1_override=2, p2_override=1,
    calibration=Calibration(cube_mac_per_us=2.7e7),
    policy=InstancePolicy(gmm2_lag_waves=2),
    wave_packing="balanced_waves",
)
res = simulate(sc)
```

写错字段名、类型不对、策略名不存在都会立即报错并给出提示, 例如
`policy.gmm2_lag_wave: 未知字段, 是否想写 'gmm2_lag_waves'?`。

底层入口 `simulate_routing_counts` 保留, 直接给路由计数 `C[dst][expert][src]` 与公式容器。

可调的全部旋钮：

| 类别           | 旋钮                                    | 取值                 | 作用                          |
| -------------- | --------------------------------------- | -------------------- | ----------------------------- |
| 波宽           | `p1_override` / `p2_override`       | 正整数               | 每波组数                      |
| wave 打包      | `wave_packing`                        | 三种策略             | 专家怎么组成波                |
| 分核           | `core_assignment`                     | 三种策略             | tile 分给哪个核               |
| dispatch 前瞻  | `dispatch_lookahead`                  | 正整数               | 搬运超前几波                  |
| GMM2 滞后      | `gmm2_lag_waves`                      | 正整数               | GMM2 后移几波                 |
| 波偏移组合     | `wave_offsets`                        | `StageWaveOffsets` | 前瞻和滞后任意组合            |
| GMM1→ACT 深度 | `gmm1_activation_depth`               | 正整数               | UB 缓冲深度                   |
| GMM2→COMBINE  | `gmm2_combine_credit`                 | 正整数               | 固定 credit 流控              |
| tile 几何      | `tile_m` / `tile_n` / `l1_tile_k` | 正整数               | 行高、列宽、K 窗              |
| L1 缓冲        | `l1_buf_num`                          | 1 或 2               | 单缓冲串行或双缓冲重叠        |
| B 矩阵复用     | `gmm1_b_reuse`                        | 开或关               | 不影响时长 (B 流不建模)       |
| COMBINE 量化   | `combine_quant_mode`                  | 0 或 1               | BF16 直写或 FP8 加 scale      |
| 相位流水       | `PipelineConstraints`                 | 队列深度             | load 与 cube 跨 tile 重叠     |
| 调度策略       | `scheduling_policy`                   | 三种策略             | 就绪集里谁先跑                |
| 任务转移       | `idle_core_stealing`                  | 重构钩子             | 空闲核拿走忙核的 tile         |
| 编排选择       | `topo_urma`                           | 开或关               | MTE 波循环或 Layered 宏波循环 |

**`topo_urma` 不是平级旋钮，是结构分叉。** 切换后波粒度从 256 行 m-group 变为专家范围，dispatch 从源推变为目的拉，combine 从配对 tile 变为批量 PUT。

| 旋钮                                                     | MTE 路径           | Layered 路径                              |
| -------------------------------------------------------- | ------------------ | ----------------------------------------- |
| `p1_override` / `p2_override`                        | 生效，决定每波组数 | 失效，Layered 按专家数和 token 数自定波数 |
| `wave_packing`                                         | 生效，三种策略     | 失效，Layered 有自己的波规划              |
| `dispatch_lookahead` / `wave_offsets`                | 生效，控制前瞻     | 失效，Layered 固定 recv 后紧跟 combine    |
| `gmm2_lag_waves`                                       | 生效，控制滞后     | 失效，Layered 的 GMM2 总与当前波同跑      |
| `gmm1_activation_depth`                                | 生效               | 生效                                      |
| `gmm2_combine_credit`                                  | 生效               | 生效                                      |
| `core_assignment`                                      | 生效               | 生效                                      |
| `tile_m` / `tile_n` / `l1_tile_k` / `l1_buf_num` | 生效               | 生效                                      |
| `combine_quant_mode`                                   | 生效               | 生效                                      |

`KernelConfig` 的编译期旋钮（`l1_buf_num`、`l1_tile_k`、`combine_quant_mode`）以 `KernelConfig` 为唯一事实源。手工拼 `PrimitiveCosts` 时入口自动按 kernel 重绑公式，任何拼法都生效。`weight_nz` 与 `gmm1_b_reuse` 只描述权重搬运，而权重搬运不建模，所以不影响时长。

策略旋钮用名字引用：

| 旋钮                  | 可选名字                                                          | 管什么                   |
| --------------------- | ----------------------------------------------------------------- | ------------------------ |
| `tile_grid`         | `swizzled` / `split_rows`                                       | GMM1/GMM2 的 tile 怎么切 |
| `wave_packing`      | `sequential_greedy` / `longest_expert_first` / `balanced_waves` | 专家怎么组成波           |
| `core_assignment`   | `static_round_robin` / `greedy_least_busy` / `contiguous_block` | tile 分给哪个核          |
| `scheduling_policy` | `earliest_start` / `critical_path_first` / `priority_by_stage`  | 就绪集里谁先跑           |
| `restructure`       | `idle_core_stealing`                                              | 运行时图重构             |
| `orchestration`     | `mte` / `layered` / `"包.模块:类"`                              | 用哪个建图器             |

带参数时写成表：`{name = "split_rows", parts = 2}`。自定义策略用 `moe_cost_model.register(类别, 名字, 构造函数)` 注册。

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

`orchestration` 决定用哪个建图器。继承 `MteEventBuilder` 改波主循环，注册后从场景引用；也可以直接写 `"包.模块:类"`。

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
  (`ModelOptions.act_to_gmm2="gm"`, 缺省) 下计**: ACT 把量化激活写回 GM, GMM2 的 A
  再从 GM 读回来 (参考 kernel 就是这样: epilogue 写 `activationQuantDataPtr`, GMM2
  从 `Location::GM` 取同一个指针)。`"onchip"` 编排下 A 留在片上 (硬件有 UB→L1 通路),
  A 流不付 GM 字节, 代价是一个 m-group 的 GMM1/ACT/GMM2 必须共位于一个核 ——
  并行度上限变成 m-group 数。单缓冲下 GMM2 同样加 `restart` (每 kL1 块一次)。
- `R_cube` (`cube_mac_per_us`) 必填, 无缺省。**规格峰值由 `cube_mac_per_us("fp8")` 给出
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
| stage 函数 | `builders/gmm1.py`、`activation.py`、`gmm2.py` | 事件生成，与传输协议无关   |
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
| ACT | 一对一跟随 GMM1 tile | 同核的 AIV0 |
| GMM2 | 按输出列切 tile (`ceil(h / tile_n)` 个), 可换 `tile_grid`; 每个 tile 要行范围相交且覆盖整个 K 的 ACT | 每个 tile 一个 AIC 核, 游标轮转 |
| COMBINE | 一对一跟随 GMM2 tile | 同核的 AIV1 |

块号按全局 m-group 序对核数取模决定归属, 与波无关 —— 同一个块落在哪个波, 归属的核都不变。

**块数少于核数时, 多出来的核在 dispatch 阶段没有活。** 块数 = Σ 各专家 `ceil(行数 / tile_m)`。每卡专家少、每专家行数不足 `tile_m` 时 (小 batch) 这一项很小, dispatch 的并行度会成为瓶颈。

### 3. 当前改不了的结构

波粒度只有两种：MTE 路径按 256 行组切波，Layered 路径按专家范围切波。ACT 总与 GMM1 同波。同核程序序没有建成依赖边，靠资源互斥保序。带宽争用不建模（信道模型已停用，见下）。

要突破这些需要改 `builders/` 或 `scheduler/` 的结构，改完重新对实测校准。

## 输出解读

每次仿真返回以下可分析字段：

| 字段                                            | 含义                |
| ----------------------------------------------- | ------------------- |
| `kernel_total_us`                             | 最慢 rank 的执行时间, 记到最后一个 COMBINE 结束 |
| `kernel_dag_end_us`                           | 含尾段的结束时刻 (对比实测整段墙钟时用) |
| `rank_results[r]["total_us"]`                 | 各 rank 执行时间, 记到该 rank 最后一个 COMBINE 结束 |
| `rank_results[r]["dag_end_us"]`               | 各 rank 含尾段的结束时刻 |
| `rank_results[r]["resource_utilization"]`     | 每核利用率          |
| `rank_results[r]["stage_busy_us"]`            | 各 stage 忙碌时长   |
| `rank_results[r]["stage_dependency_wait_us"]` | 各 stage 等数据时长 |
| `rank_results[r]["stage_resource_queue_us"]`  | 各 stage 等引擎时长 |
| `rank_results[r]["critical_path"]`            | 关键路径事件链, 终点是最后一个 COMBINE |
| `rank_results[r]["cursor_trace"]`             | 游标推进轨迹        |
| `provenance`                                  | 全部常数出处报告    |

执行时间不含尾段 (counts_export / core_sync / rank_sync / buffer_init / unpermute / finalize)。尾段事件仍在事件图里照常调度, 只是不计入。共享专家的 GMM2 排在尾段 core_sync 之后, 因此也不在执行时间内。

对比两个变体时，先看 `kernel_total_us` 差值，再看 `stage_busy_us` 哪个 stage 变了，最后看 `critical_path` 上卡在哪种等待。

## 硬件规格 (spec) 与实测 (measured) 分开记

规格是**峰值**: 用它算出来的是时间下界, 所以要配效率系数。实测值已经含了争用与开销,
不该再乘效率。两者混在一个出处标签里早晚用错, 所以 `spec:` 是独立的一类
(`config/provenance.py` 的分类表)。

出处: 《昇腾 950 NPU 架构白皮书》(华为)。本仓 kernel 是 `__NPU_ARCH__==3510`
(第三代达芬奇 / arch35)。950PR 与 950DT 同源同 die, **每核计算峰值相同**, 差别在存储与
互连档位 —— 所以计算侧一组数, 存储侧两组 (`config/platform.py`)。

| 项 | 值 | 说明 |
| --- | --- | --- |
| 合计算力 (32 Cube / 64 Vector 档) | FP8/MXFP8/HiF8 919, BF16/FP16 486, MXFP4 1784, TF32 243 TFLOPS | 含 Vector 部分 |
| Vector 部分 | FP16/BF16 54, FP32 27 TFLOPS | 从合计里减掉它才是 Cube 部分 |
| 每 Cube 核 FP8 峰值 | 864 / 32 / 2 = **1.35e7 MAC/µs** | 除 2 是因为一次 MAC 两个 FLOP |
| 聚合 HBM | 950PR **1.6 TB/s** / 950DT **4 TB/s** | |
| 片间互连 (灵衢 2.0) | 2 TB/s | |

自洽核对: `2×432 + 54 = 918 ≈ 919`, `4×432 + 54 = 1782 ≈ 1784` —— 白皮书的"FP8 同频给
FP16 的 2 倍、MXFP4 给 4 倍"与合计值对得上。

**未逐字核对**: 这些数取自白皮书规格表的转述, 本容器的网络策略拦了 hiascend.com 与华为
OBS, 没能直接打开官方 PDF。拿到原件请核对 Cube/Vector 的算力拆分与 950PR 的 HBM 容量
(转述有 112GB/128GB 两说; 模型只用带宽, 不用容量)。

### 带宽护栏: 单核常数不能突破聚合上界

`BW_L1_GM` 是**单核**实测值 (51.9 GB/s), 聚合 HBM 是规格上界。单核值乘活跃核数不得超过
聚合值, 否则那个时长物理上不可能:

| 活跃核数 | 950PR 占聚合 | 950DT 占聚合 |
| ---: | ---: | ---: |
| 28 (单卡真实可用) | **91%** | 36% |
| 32 | 104% (超) | 42% |
| 36 | 117% (超) | 47% |

所以 **950PR 在 28 核上已经吃掉 HBM 聚合带宽的 91%** —— 任何增加 GM 流量的编排在这档上
几乎没有余量, 而 950DT 有 2.5 倍。`build_analytical_costs(platform=..., active_cores=...)`
会按 `min(单核上限, 聚合/活跃核数)` 压一次; `design_space(..., platform=...)` 则在每行给出
"这个方案需要的聚合带宽占规格的百分比", 超 100% 直接标 `超!`。

## 精度边界

**本节数字是旧公式下的结论。** 搬运口径已于 2026-10-04 按三个实测点定为相加 (见上, GMM1
每 tile 的误差 +0.4%~+3.5%), 但整体墙钟校准还缺两样: 实测的 Cube 速率; 以及按并发核数
分档的 `BW_L1_GM` (A 流斜率显示它随并发变: 28 核 45.3 / 18 核 37.0 GB/s)。

旧公式下: `profiles.MEGAMOE_A8W8` 这组取值 (即那份实现) 在 B≤128 标定域内各 stage busy
误差 ±5%，墙钟偏差 -6~-8%。**注意缺省值不是这组取值** —— 缺省是"最少假设"，与实测对齐要
显式引用 profile (场景文件里写 `profile = "megamoe-a8w8"`)。

偏离该 profile 的取值为未验证取值：模型照常给出预测，但结论需实测抽检。标定域外的已知失效：B=1024 时 COMBINE 偏差 +114~246%（BW_SCATTER 单点标定域外），GMM1 系统性高估 +4~10%（B 矩阵逐 tile 计费）。

### 数据搬运带宽

| 路径                    | 参数                   | 当前值     | 标定方法                   | 不准之处                                                  |
| ----------------------- | ---------------------- | ---------- | -------------------------- | --------------------------------------------------------- |
| GM→L1，GMM 载入        | `BW_L1_GM`           | 51.9 GB/s  | B=64 H 扫描差分单点        | 并发数未扫；激活与权重合并折算未分离                      |
| GM→UB，ACT             | `BW_UB`              | 93 GB/s    | ACT 大 m tile 单点         | 读端口约 123、写端口约 142 B/cyc，速率不同，93 是混合折算 |
| UB→GM，COMBINE 散射写  | `BW_SCATTER`         | 139.5 GB/s | B=64 随机路由反推          | B=1024 实测偏差 +168~246%，域外失效                       |
| GM→L1，dispatch 本地段 | `BW_LOCAL_GM`        | 157 GB/s   | MTE 大尺寸拟合，单核无干扰 | 28 核并发时每核掉到 41~100 GB/s，未折入                   |
| GM→L1，dispatch 远端段 | `BW_REMOTE_GM`       | 31 GB/s    | 1→2 行段差分              | 只在 B=64 域 1~2 行段验证，大段未测                       |
| 跨卡，dispatch 远端     | `BW_WINDOW`          | 33 GB/s    | dispatch 窗排空差分 ×4    | 同上                                                      |
| URMA GET，Layered 接收  | `URMA_GET_BW_SINGLE` | 2.25 GB/s  | pair 两点 OLS 拟合         | drain-chunk=8 下界；并发 ≥5 流未验证                     |
| URMA PUT，Layered 聚合  | `URMA_PUT_BW_SINGLE` | 2.25 GB/s  | 无直测数据                 | 取 GET 对称值，完全假设                                   |
| GM，UNPERMUTE 尾段      | `BW_UNPERMUTE_AGG`   | 0.95 TB/s  | UNPERMUTE 双尺度           | B=64 偏 +18%，B=1024 命中                                 |

### Cube 计算速率

| 参数       | 当前值    | 状态                                                                                                                    |
| ---------- | --------- | ----------------------------------------------------------------------------------------------------------------------- |
| `R_cube` | 0，不启用 | 纯隔离实验 fixpipe 挂死，只有上界 13 TMAC/s。GMM1/GMM2 公式的计算项整个不生效，模型只算载入时间。计算主导的 tile 被低估 |

### Vector 计算参数

| 参数              | 当前值   | 状态                                                    |
| ----------------- | -------- | ------------------------------------------------------- |
| `VEC_REG_WIDTH` | 256 bit  | 来自 kernel 定义，精确                                  |
| `T_STARTUP_VEC` | 1.48 µs | ACT 小 m 截距单点，假设与 m 无关                        |
| ACT 每向量字节数  | 580 B    | 源码逐项计数；实验实测推算约 324 B/向量，两者差异未调和 |

搬运带宽全部是单点标定，没有一个扫过并发数；Cube 计算速率完全没有精确值；Vector 的 ACT 字节口径源码计数与实测不一致。域内 B≤128 可用，域外或参数变了需重新标定。

## 安装与运行

```bash
cd moe-cost-model
pip install -e .          # 或直接 pytest (pyproject 已配 pythonpath)
pytest tests/             # 190 项测试 (5 项需 tiling 真值, 见下), 约 1 分钟
python examples/run_scenario.py    # 场景文件 + 改旋钮对比
python examples/run_basic.py       # 底层入口
```

## GMM2 沿 K 维分段就绪 (编排选择, 非物理约束)

GMM2 的 K 就是 GMM1 切分的那个 N 轴 (`k_gmm2 = hidden_dim / activation_n_half`), 所以一个
ACT tile 只产出 GMM2 在 K 上 1/ceil(k/TILE_N) 的部分, GMM2 要累完整个 K 才有结果。**分几段
独立就绪是编排选择**: L0C 本来就沿 kL1 分块累加 (kernel 的 `ProcessTileL1`), 所以让第 j 段
只等覆盖自己 K 范围的 ACT 在物理上可行。

`ModelOptions.gmm2_k_segments`:

| 值 | 含义 |
| --- | --- |
| `1` (缺省) | 不分段: 等齐覆盖整个 K 的全部 ACT 再开工 —— 最少假设 |
| `2` | 首个 kL1 块一段 (只等 1 个 ACT), 其余合成一段 (`MEGAMOE_A8W8` 用这个) |
| `0` | 每个 kL1 块各一段, 第 j 段只等第 j 块的 ACT —— 最细 |
| `N>2` | 按 kL1 块数均分成 N 段 |

实测 (ep=5, 每专家 256 行全远端, aic=28, `kl1=256` 即 18 个 kL1 块):

| 形状 | 2 段 | 3 段 | 6 段 | 逐块 |
| --- | ---: | ---: | ---: | ---: |
| hidden=9216 专家=3 | 311.7 us | −4.1% | −4.6% | −5.2% |
| hidden=9216 专家=6 | 462.6 us | −4.6% | −5.0% | −5.0% |
| hidden=14336 专家=6 | 679.8 us | −1.0% | −1.9% | −2.4% |
| hidden=18432 专家=6 | 855.2 us | −3.0% | −2.8% | −2.5% |

两点结论: **收益主要在 2→3 段**(现状的两段是 1/18 + 17/18 的极端不均分, 改 3 段就把等待链
打散了); **过细会退化** (hidden=18432 专家=6 逐块反而比 3 段差, 事件数 240→4320 后调度器的
资源排队成为新瓶颈)。

⚠️ **这些数是上界**: kernel 侧每段要多做一次 `WaitUntilGmFlagEquals` (GM 读 + 自旋), 逐块
= 18 次轮询 vs 现在 2 次, 模型未计这项开销。所以 3 段的 −4.1% 比逐块的 −5.2% 更可信 —— 前者
只多 1 次轮询。要算出最优段数需要补一个按段数计费的轮询常数, 实测值可从 trace 的
`WAIT_GMM2_INPUT` 打点反解。

## 带宽争用: 信道模型已停用 (2026-10-03)

原先有一层速率服务器 (`Channel` + 六条通路: `gm_to_l1` / `hbm_write` /
`dispatch_read` / `dispatch_write` / `fab_src:*` / `fab_dst:*`), 事件按申报字节与应得速率
抢带宽, 争用时降速或推迟。**整层机制已移除**, 只保留字节申报:

- `Event.channel_bytes` 仍然由建图器无条件申报, 但**不参与准入、不影响任何时长**。
- 按通路汇总在 `rank_results["traffic_bytes"]` 里, 做访存量核算用。
- 去掉的 API: `Channel`、`default_channels`、`schedule(channels=...)`、
  `ModelOptions.fabric_channels`、`PipelineConstraints.channels`、
  `Scenario.default_channels`、`guardrails.check_channels`、
  `ScheduledEvent.channel_wait_us` / `channel_rate`、`RestructureContext.channel_inflight`。

为什么停用: 这层机制的两个常数本来就不同尺度, 调不出可信的争用。片间通路最明显 ——
聚合 `BW_WINDOW = 33000` B/µs 是**整卡**带宽, 而逐事件速率 `BW_REMOTE_GM = 31000` 是从
28 核并发的真实运行反解的**单核**值 (已含平均争用)。于是单个事件就吃掉整卡 94%, 并发再
叠 26 倍降速, 争用被计了两遍。片内四条则相反: `default_channels` 的聚合 = 每核速率 × 核数,
按构造恰好无争用, 打开和不打开逐位相同 —— 不收紧就没有信息, 收紧又没有整卡带宽的实测可依。

移除的影响: 40 个基准用例里 **38 个时长逐位不变**; 只有两个原本开了信道的变快
(`pipeline_engine_queue2` −11.3%, `pipeline_large_split` −16.9%)。

要重建争用模型, 需要先有**该层级真实的聚合带宽**实测 (单核独占 + 多核并发两组), 字节申报
留着就是为了那一天。

## 搬运口径: A流 + B流 相加, 数据释放不计

GMM tile 的搬运事件时长 = **A流 + B流**。
**数据释放事件 (结果 L0C -> GM/UB) 不计时长**: 闭式公式本就只计搬入, 相位流水的 fix 相位
时长归 0 (节点保留, 仍承载 QUEUE:fix 与归还 L1 缓冲槽的语义), 所以
`PipelineConstraints.phases.fix_bw_bytes_per_us` 不再影响任何时长。

第一条是**从实测定下来的**, 不是口径决定 (2026-10-04)。三个实测点把两个混淆变量分开:

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

## 核空闲分解: 哪些消不掉, 哪些是真浪费

"核不能有空闲"作为绝对约束在逻辑上不可能 —— 事件必须等前置完成, 而 t=0 时 GMM1 还在等
`dispatch_ready` (AIV1 产出), 所以开头所有 AIC 必须空着。能成立的不变量是
**work-conserving**: 核不得在"存在已就绪的活"时空闲。

`rank_results["idle_decomposition"]` 按角色池 (AIC / AIV0 / AIV1) 给出:

| 字段 | 含义 |
| --- | --- |
| `busy_us` | 占用核·us; 与 `resource_busy_us` 的同角色合计逐位一致 |
| `forced_idle_us` | 此刻池里**没有**已就绪未开始的事件 → 消不掉, 只能改 DAG 结构 |
| `avoidable_idle_us` | 有就绪的活却有核空着 → work-conservation 违规, 换 tile→核 绑定方式可回收 |
| `segments` | 每一处 avoidable 的 (时间窗, 空闲核, 等着的就绪事件) |
| `work_conserving` | `avoidable_idle_us == 0` |

`avoidable_idle_us` 是**上界**: 它没有检查那个活是否真能落到那个空核上 (AIC/AIV 共位约束、
队列计数信号量)。当"值不值得动绑定方式"的量级判断用, 不要当承诺。

```bash
python tools/check_work_conservation.py examples/scenario_basic.toml --role AIC:
python tools/check_work_conservation.py --sweep              # 扫几个形状对比
python tools/check_work_conservation.py <场景> --assert-conserving   # CI: 有违规则非零退出
```

实测 (ep=5, 每专家 256 行全远端, aic=28): `forced` 在四个形状上几乎恒定 (~2150 核·us),
那是 dispatch 前段与波间排空的固定代价, 换 wave/tile 编排也消不掉; `avoidable` 占空闲的
22%~54%, 是换绑定方式能动的部分。

### dispatch 每波每核的调用开销

`DispatchMechanisticLatency.t_call_oh_us` 缺省 **0** —— 实测参考值 `T_CALL_OH = 1.006 us`
(`config/hardware.py`, 来自 dispatch 零行调用), 由算子工程师按自己的实现填。

填了非 0 值时, 开销的归属随绑定方式变:

- **静态绑定**: 每波每核一个 `dispatch_call` 事件承担。
- **AIV1 晚绑定**: 改成 `Event.once_per_core` —— 由该核**本波第一段 dispatch** 承担,
  `dispatch_call` 事件时长归 0。这样段可以落任意空闲核, 开销落在真正搬数据的那个核上。
  若改成"dispatch 跟着 dispatch_call 的核走", 反而会让已就绪的段去等一个忙核 (实测 AIV1
  avoidable 9.8~18.6 核·us), 与零空闲冲突。
- 只有**实际做过 dispatch** 的 (波, 核) 组合付这次开销: 9216/6 上静态是 84 个组合各付一次,
  晚绑定下段集中到较早空闲的核, 只有 76 个组合付费。

### 把 avoidable 清零: 晚绑定 + 关键路径打破平手

```python
res = m.simulate_routing_counts(
    ..., scheduling_policy=m.WorkConservingCriticalPath(),
    options=m.ModelOptions(late_bind_pools=("AIC", "AIV1")))
```

| 旋钮 | 作用 |
| --- | --- |
| `ModelOptions.late_bind_pools` | 角色入池: 事件只声明"要一个 AIC", 调度器在**派发时刻**绑最早空闲的成员。缺省 `("AIC", "AIV1")`; `()` = 静态绑定 (某实现的分核方式)。`"AIC"` 入池隐含 `"AIV0"` 入池 —— `GMM1 -> ACT` 同核是物理约束, 整对一起漂移 |
| `WorkConservingCriticalPath` | 排序键 `(start, -remaining_path_us, order, name)`: start 仍排第一位, 核不会为等未就绪的事件空闲; 关键路径只在**同样能立刻开始**的候选之间定先后 |
| `ModelOptions.dispatch_pacing` | 下一波 dispatch 等什么: `"none"` (缺省, 不等, 跨波连续 dispatch) / `"per_core"` (等本核上一波最后一个 combine, `MEGAMOE_A8W8` 用这个) / `"wave"` (等该波全部 combine) |

实测 (hidden=9216 专家=3, rank0), 单位核·us:

| 配置 | AIC busy | 利用率 | forced | avoidable | dag_end |
| --- | ---: | ---: | ---: | ---: | ---: |
| 静态绑定 + 贪心 | 5455.0 | 59.5% | 1693.2 | **2023.8** | 327.6 us |
| 双池晚绑定 + 贪心 | 5455.0 | 61.1% | 3479.5 | **0** | 319.1 us |
| 双池晚绑定 + 关键路径 | 5455.0 | 64.0% | 3071.7 | **0** | **304.5 us** |

6 个形状 x 3 种 `dispatch_pacing` 下 AIC/AIV0/AIV1 的 `avoidable` 全为 0。`busy` 完全不变 ——
晚绑定只换"哪个核做", 不改工作量。dag_end 相对静态基线:

| 形状 | 静态基线 | 双池 + 贪心 | 双池 + 关键路径 |
| --- | ---: | ---: | ---: |
| 9216 / 3 | 327.6 | 319.1 | **304.5** (−7.1%) |
| 9216 / 6 | 488.1 | 479.6 | **456.9** (−6.4%) |
| 14336 / 3 | 380.0 | 378.4 | **369.0** (−2.9%) |
| 14336 / 6 | 705.3 | 711.3 (+0.9%) | **688.6** (−2.4%) |
| 18432 / 3 | 494.3 | 488.2 | **448.5** (−9.3%) |
| 18432 / 6 | 880.7 | 911.4 (+3.5%) | 893.8 / **881.2** (`pacing="none"`) |

**零空闲不等于最快**: 纯贪心在两个形状上把墙钟拖长了 (有就绪的活就立刻上核, 可能把更关键
的 tile 挤后), 关键路径打破平手才把这部分补回来。反过来, 只换策略不开池也不够 —— 静态绑定
下 6 个形状里 4 个仍有 avoidable (AIC 最多 1711.9 核·us)。两件事互相独立。

⚠️ 一处残留保守: `"per_core"` 配速边按建图时的核号连, 晚绑定后可能指向别的核 (占总边数
1.4%~3.0%, 计入 `forced`, 墙钟略高估)。详见 `docs/design_space_gaps.md` 缺口 3。

晚绑定与相位流水**可以同用** (2026-10-04, 原缺口 9): 同一个 tile 的几个相位编成**核组**,
核号由组里最先派发的那个事件选定, 同组其余事件跟随 —— 相位事件不持核资源, 所以不能靠
`colocate_with` (它要求锚点先绑定, 而先跑的恰恰是不持核的那一相)。实测 9216/3: 静态+拆相位
220.83 us, 晚绑定+不拆 196.92 us, 两者叠起来 **183.62 us**, 不变量仍然全 0。

## tiling 真值与实测工件

`examples/*.toml` 的 `[tiling] path` 指向实测 run 的 `raw/tiling_rank0.bin` —— 这是
`tests/test_guardrails.py` 与 `tools/eval_suite.py` 核对"场景文件声称的形状 == 跑出数据的
kernel 配置"的唯一依据, 也是本项目防"手抄参数没人核对"的那一层。

打点工件体积大, `.gitignore` 把 `/data/*/raw/` 整个排除了, 所以**干净克隆里 tiling 真值
缺席**: 5 条护栏测试 skip, `eval_suite` 一个场景都跑不了。tiling 真值本身只是十来个整数,
用导出器写成几百字节的 JSON 旁置文件入库即可永久解决:

```bash
python tools/export_tiling.py --all        # 在有 raw/*.bin 的采集机上跑一次
git add data/*/tiling_rank0.json           # 不在 gitignore 里
```

`parse_tiling` 在 `raw/*.bin` 缺失时自动回落到上一级同名 `.json`, 所以 `examples/*.toml`
一字不用改。两者都没有时报错会指明这条命令。

## 回归保护

`tests/test_golden.py` 对 40 个配置核对调度指纹: 每个事件的起止时刻、等待归因、关键父事件取 sha256, 任何一位浮点差异都会失败。覆盖 MTE / Layered 两条路径、波偏移、编译期旋钮、三种调度策略、分核与打包策略、相位流水、计数信号量、任务转移。

```bash
python tools/gen_golden.py --check     # 核对, 不写文件
python tools/gen_golden.py             # 重新生成快照
```

重构与提速必须通过 `--check`。只有在有意改变模型行为时才重新生成快照, 并在提交说明里写明哪些用例变了、为什么。

## 仿真耗时

| 配置 | 事件数 | 耗时 |
| --- | --- | --- |
| MTE, 4 rank × 64 专家, B=64 (`examples/run_basic.py`) | 2.7 万 | 约 1.5 s |
| Layered, 同上 | 1.9 万 | 约 1.1 s |
| 相位流水, 4 rank × 64 专家, B=1024 | 3.5 万 | 约 4 s |

无重构钩子、使用内置调度策略时, 各 rank 独立调度。`idle_core_stealing` 每次提交都扫描全部未提交事件, 耗时随事件数平方增长: 6800 事件约 30 s。

## 项目结构

```
moe-cost-model/
├── pyproject.toml
├── src/moe_cost_model/
│   ├── __init__.py              # 显式导出
│   ├── scenario.py              # 统一入口: Scenario / load_scenario / simulate
│   ├── registry.py              # 策略名注册表
│   ├── api.py                   # simulate_routing_counts 底层入口
│   ├── config/                  # 第 0 层: 纯参数
│   │   ├── hardware.py          #   硬件常数 + KernelConfig
│   │   ├── policy.py            #   InstancePolicy + StageWaveOffsets
│   │   ├── pipeline.py          #   PipelineConstraints + QueueDepths
│   │   └── provenance.py        #   常数出处标签系统
│   ├── shape.py                 # 第 1 层: MegaMoeShape / ModelOptions
│   ├── costs.py                 # 第 1 层: 各 stage 物理公式
│   ├── scheduler/               # 第 2 层: 通用离散事件调度引擎
│   │   ├── events.py            #   Event / Channel / 速率服务器
│   │   ├── engine.py            #   MultiResourceScheduler
│   │   └── policies.py          #   EarliestStart / WorkConservingCriticalPath / PriorityByStage
│   ├── planning/                # 第 3 层: wave 规划 + tile 网格
│   │   ├── waves.py             #   plan_waves / swizzle / Layered 波规划
│   │   ├── core_assignment.py   #   StaticRoundRobin / GreedyLeastBusy / ContiguousBlock
│   │   └── wave_packing.py      #   SequentialGreedy / LongestExpertFirst / BalancedWaves
│   │   └── tile_grid.py         #   TileGrid: 行范围 x 列范围, 可自定义切分
│   ├── builders/                # 第 4 层: 事件图构建
│   │   ├── base.py              #   公共基类
│   │   ├── context.py           #   BuildContext
│   │   ├── gmm1.py + activation.py  #  GMM1 tile + ACT tile
│   │   ├── gmm2.py              #   GMM2 head/tail
│   │   ├── comm/                #   通信协议接口
│   │   │   ├── base.py          #     DispatchTransport / CombineTransport
│   │   │   ├── mte.py           #     MTE: DataCopyPad 直写 + 配对 tile combine
│   │   │   └── urma.py          #     URMA: 批量 GET/PUT + AIV1 程序序链
│   │   ├── mte.py               #   MTE 编排
│   │   ├── layered.py           #   Layered 编排
│   │   └── pipeline_expand.py   #   相位拆分
│   ├── model.py                 # 第 5 层: A8W8WaveCostModel 编排
│   └── analysis/                # 第 6 层: 关键路径 / 空闲核任务转移
├── tests/                       # 155 项 (引擎 / 建图 / API 锚点 / Layered / golden 指纹 / 场景)
├── examples/                    # scenario_basic.toml + run_scenario.py + run_basic.py
└── tools/                       # 分析脚本 (审计 / 诊断 / 全量对比 / golden 生成 / HTML 报告)
```
