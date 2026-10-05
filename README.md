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
from moe_cost_model import MEGAMOE_A8W8 as P, StageLink
simulate_routing_counts(..., **P.shape_kw(), options=P.options)
# 以那份实现为底, 只改一条 stage 边 (GMM2 改成逐 kL1 块就绪)
simulate_routing_counts(..., **P.shape_kw(), options=P.with_options(links=(
    StageLink("gmm1", "activation", location="onchip", depth=1,
              colocated_by_hardware=True),
    StageLink("activation", "gmm2", readiness=0))))
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
| COMBINE 数据格式 | `combine_quant_mode`                | 0 或 1               | BF16 直写或 FP8 加 scale (只管字节, 见¹) |
| COMBINE 元数据   | `combine_meta_bytes_per_row`        | 12 / 16 / 32         | 每行搬几个路由字段 (算法只需 3 项) |
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
| `StageLink("gmm1","activation").depth` (原 `gmm1_activation_depth`, 已删) | 生效 | 生效 |
| `gmm2_combine_credit`                                  | 生效               | 生效                                      |
| `core_assignment`                                      | 生效               | 生效                                      |
| `tile_m` / `tile_n` / `l1_tile_k` / `l1_buf_num` | 生效               | 生效                                      |
| `combine_quant_mode`                                   | 只有字节宽度生效¹  | 只有字节宽度生效¹                         |

¹ `combine_quant_mode` 只管**数据格式** (写侧每元素字节: BF16 2B → FP8 1B + 1/32 scale)。
"combine 跑在哪个角色、什么粒度"是**编排**, 分别由 `ModelOptions.roles` 与
`ModelOptions.combine_granularity` 给 (2026-10-04 补齐, 原缺口 11)。参考实现把数据格式与
这两件事绑在同一个模板参数上, 那是那份实现的耦合, 不是物理。

`KernelConfig` 的编译期旋钮（`l1_buf_num`、`l1_tile_k`、`combine_quant_mode`）以 `KernelConfig` 为唯一事实源。手工拼 `PrimitiveCosts` 时入口自动按 kernel 重绑公式，任何拼法都生效。

> **2026-10-05 订正**: 本段原先写"`weight_nz` 与 `gmm1_b_reuse` 只描述权重搬运, 而权重搬运
> 不建模, 所以不影响时长"。**那是错的** —— 权重 (B 流) 现在是载入项里**更大**的那一股
> (`b_load = wb·K·cols / bw_b`), 两个旋钮都显著改时长。实测同一个 tile (m=256, K=6144,
> cols=256): 基线 90.917us → `weight_nz` + NZ 带宽 80000 得 **69.627us (−23%)**;
> `gmm1_b_reuse_frac=0.53` 得 **62.430us (−31%)**。
> 另外字段名已变: `gmm1_b_reuse` → **`gmm1_b_reuse_frac`** (比例, 不是布尔)。

策略旋钮用名字引用：

| 旋钮                  | 可选名字                                                          | 管什么                   |
| --------------------- | ----------------------------------------------------------------- | ------------------------ |
| `tile_grid`         | `row_major` (缺省) / `swizzled` / `split_rows`                   | GMM1/GMM2 的 tile 怎么切 |
| `wave_packing`      | `sequential_greedy` / `longest_expert_first` / `balanced_waves` | 专家怎么组成波           |
| `core_assignment`   | `static_round_robin` / `greedy_least_busy` / `contiguous_block` | tile 分给哪个核          |
| `scheduling_policy` | `earliest_start` / `critical_path_first` / `priority_by_stage`  | 就绪集里谁先跑           |
| `restructure`       | `idle_core_stealing`                                              | 运行时图重构             |
| `orchestration`     | `mte` / `layered` / `"包.模块:类"`                              | 用哪个建图器             |

带参数时写成表：`{name = "split_rows", parts = 2}`。自定义策略用 `moe_cost_model.register(类别, 名字, 构造函数)` 注册。

### 事件粒度: 五个 stage 共有的一个维度 (2026-10-04 补齐, 原缺口 12)

**粒度与 tile 几何是两件事**, 补齐前被混成一件:

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

补齐前只有 `combine_granularity` 一个旋钮 (缺口 11 的产物, 为回答一个具体问题就地加的),
dispatch 的叫 `dispatch_rows_per_item`, 而 GMM1/SwiGLU/GMM2 的写死为 1。那是提问历史留下的
洞, 不是物理 —— 这与"缺省不引用任何实现"是同一类毛病的另一种形态: **只有被问到的那一个
维度才被抽象出来**。两个历史旋钮现在降级为兼容视图, 两边矛盾会报错。

三条必须知道的物理耦合:

1. **ACT 的粒度 > 1 可能静默无效。** ACT 必须与产它的 GMM1 同核 (L0C→UB 的 Fixpipe),
   所以只在"喂它的 tile 既同核又 n 相邻"时才合并。轮转分核把相邻 n-tile 散到不同核,
   这时它是空操作 —— 查事件 meta 的 `gmm1_events_in_event` 确认它有没有生效。
2. **ACT 粒度 g 要求 UB 槽数 ≥ g** (`StageLink("gmm1","activation").depth`), 否则死锁,
   构造时直接报错。
3. **combine 的攒批不按核分组**: combine 从 GM 读 GMM2 的输出, 同核不是物理约束。
   按核分组会让这个旋钮在轮转/晚绑定下静默失效。

**粗粒度不是收益开关, 符号随形状翻转**: 28 核确定性夹具 (40 个 GMM1 tile, 填不满核) 上
四种粗粒度全部变慢 (`combine=0` 从 245.97 慢到 596.47); 而 `scenario_basic.toml`
(4 卡 × 64 专家, tile 数远多于核数) 上 `gmm2=2` 从 1751.48 快到 1741.46。**必须扫, 不能
照搬取值。**

### 队列深度 ≠ 执行单元数 (这个混淆是一个真 bug 的根源)

`QueueDepths` 的五个字段是**在飞上限 / 缓冲槽数** —— 能提前多少发起, 由 L1/UB 槽数决定,
是编排选择, 所以它是旋钮。而**执行单元数**是硬件事实, 每核每种单元恒为 1:

| 队列深度 (旋钮) | 执行单元 (事实, 恒 1) | 怎么表达 |
| --- | --- | --- |
| `mte_aic` | 一个 AIC 一条 MTE2 (GM→L1) | `MTE2:c{core}` 容量 1 |
| `fix` | 一个 AIC 一条 FixPipe (L0C→UB/GM) | `FIXPIPE:c{core}` 容量 1 (见下) |
| `mte_aiv` | 一个 AIV 一条 MTE (GM↔UB) | `MTE_AIV:{eng}:c{core}` 容量 1 |
| `cube` | — | `.cb` 相位本身独占 AIC 核资源 |
| `vec` | — | ACT/COMBINE 主事件本身独占 AIV 核资源 |

深度恒为 1 时两者重合, 所以缺省下看不出区别。**深度 >1 时只有队列深度、没有执行单元约束,
等于给每个核凭空多出几条管道** —— 那正是下界断言抓到的那个洞 (载入可无限并行, 墙钟低于
带宽下界 26.6%)。写成**容量 1 的计数信号量**而不是独占资源, 是因为晚绑定只改写
`acquires`/`releases` 的核后缀 (`:c7` → `:c*`), 独占资源会把相位钉在建图时的占位核号上。

`FIXPIPE` 现在**量不出来**: 结果写出 (数据释放事件) 按口径忽略不计, fix 相位时长恒为 0。
它是一条预置护栏 —— 不花代价, 口径改了自动生效。`PhaseRates.fix_bw_bytes_per_us` 给了值会
**直接报错**而不是静默无效 (它是全项目唯一"声明了却没有读者"的参数, 一个会静默吞掉用户
输入的旋钮比没有这个旋钮更糟)。

顺带查出一个**什么都没测到的 golden case**: `pipeline_fix_phase` 原先靠给那个死参数来
"覆盖 fix 相位", 于是它一直与 `pipeline_split` 逐位相同 —— 占着名分却没有鉴别力。
现在去掉那个参数, case 保留 (它仍覆盖 fix 相位的事件结构: 占 `QUEUE:fix` 与 `FIXPIPE`),
而且哪天 fix 口径改成计时长, 差异会在这里显形。

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
| ACT | 一对一跟随 GMM1 tile | 同核的 AIV0 |
| GMM2 | 按输出列切 tile (`ceil(h / tile_n)` 个), 可换 `tile_grid`; 每个 tile 要行范围相交且覆盖整个 K 的 ACT | 每个 tile 一个 AIC 核, 游标轮转 |
| COMBINE | 一对一跟随 GMM2 tile | 同核的 AIV1 |

块号按全局 m-group 序对核数取模决定归属, 与波无关 —— 同一个块落在哪个波, 归属的核都不变。

**块数少于核数时, 多出来的核在 dispatch 阶段没有活。** 块数 = Σ 各专家 `ceil(行数 / tile_m)`。每卡专家少、每专家行数不足 `tile_m` 时 (小 batch) 这一项很小, dispatch 的并行度会成为瓶颈。

### 3. 当前改不了的结构

波粒度只有两种：MTE 路径按 256 行组切波，Layered 路径按专家范围切波。ACT 总与 GMM1 同波。同核程序序没有建成依赖边，靠资源互斥保序。带宽争用不建模（信道模型已停用，见下）。

要突破这些需要改 `builders/` 或 `scheduler/` 的结构，改完重新对实测校准。

## 四层架构: 实现 / 编译点 / 运行期 / 事件图 (2026-10-05)

这个项目原先只能表达"某一份 A8W8 Wave 实现": 换 kernel 变体是翻一个布尔
(`KernelConfig.topo_urma`), 标定常数是一套覆盖所有编排的全局量, 编译参数在 C++ 与 Python
里各写一份而没有机制对上。现在分成四层, 每层有自己的入口与护栏。

```
Workload (token/专家/路由)
   -> Runtime  (拓扑 + 波推进策略)        implementations/runtime.py
   -> Compile  (编译轴 + 编译指纹)        implementations/compile.py
   -> Lowering (每份实现一个适配器)        implementations/megamoe.py
   -> DAG      (类型化事件图)              ir/
   -> Scheduler/Timing (与 kernel 无关)    scheduler/
```

### 实现身份: 换 kernel 是换适配器, 不是翻布尔

```python
>>> m.MEGAMOE_A8W8.implementation          # ascend950.megamoe.a8w8_wave.v1
>>> m.MEGAMOE_A8W8.compile_config.describe()
'66f41072c6312c23 (combine_meta_bytes_per_row=32)'
```

仓内两份实现各有身份与**源码依据** (`source_refs` 指向真实文件, 有测试核对路径存在):

| 实现 id | 源码 | 建图器 |
| --- | --- | --- |
| `ascend950.megamoe.a8w8_wave.v1` | `mega_moe_wave_a8w8.h` | `builders/mte.py` |
| `ascend950.megamoe.layered.v1` | `mega_moe_layered.h` | `builders/layered.py` |

场景文件里 `orchestration = "ascend950.megamoe.layered.v1"` 或旧名 `"layered"` 都可以, 两种
拼法现在**同值** —— 之前 `"layered"` 只换建图器而波计划仍按 `topo_urma` 分支, 得到"Layered
建图器 + m-group 波宽"这种错配组合。

**每条结果自带身份**: `rank_results[i]["implementation"]` 给出实现 id、源码依据、编译指纹、
编译点的一行描述、执行时间记到哪个 stage、以及运行拓扑。一个时长数字不再能脱离"哪份 kernel、
哪个编译点、几张卡几个核"而存在 —— 这是把标定值按域分开的前提。

### 编译指纹: 为什么不能只靠 tiling key

kernel 自己的 tiling key 只编码 5 个轴 (`mega_moe_tiling_key.h`), 而 `TILE_M`/`TILE_N`/
`L1_BUF_NUM`/`IsGmm1Interleaved`/`TOPK_PREFETCH` 都在 key 之外 —— **同一个 key 可以对应多个
二进制**。所以编译点用 17 个轴的指纹表达, 形状与拓扑**不进**指纹 (它们每次运行都变, 混进来
指纹就失去"同一个二进制"的含义; 形状域与拓扑另有 `ShapeDomain` / `RuntimeTopology`)。

有测试逐轴扫: 任何一个声明的轴不进指纹就红。

### 编译清单: 与 C++ 源码对账

```bash
python tools/compile_manifest.py --check     # 失配则退出码 1
```

从 `mega_moe/include/CMakeLists.txt` 的 `MEGAMOE_*` cache 变量、两行 `#ifndef/#define` 宏缺省、
白名单 `constexpr` 常数、以及 `BlockSchedulerSwizzle<Offset, Direction>` 的模板实参抽出 25 项,
再与 Python 侧逐项对账 (18 项)。**只报告, 不改常数。**

为什么需要它: 这个 bug 类已经真实发生过 —— `KernelConfig.swizzle_direction` 缺省 1、注释声称
kernel 用 `<3, 1>`, 而 `common/mega_moe_gmm_common.h:33` 写的是 `<3, 0>`; m 组 > 1 时模型的
GMM tile 遍历顺序相对 kernel 是 M/N 转置的, 实测墙钟差 **+5.0%**。注释不会报错, 对账会。
测试里有一条把源码树复制出去只改那一个模板实参, 断言对账能抓到 —— 那是真实会发生的方向。

派生关系不丢: `L1_TILE_M_256 = MEGAMOE_TILE_M` 解到 256, `248U * 1024U` 折成 253952。
Python 把 `URMA_FLAG_WINDOW_TOKENS` 抄成字面量 256 而 kernel 从 `tile_m` 派生, 这种脱钩因此
查得出来。

清单还记下**标定语料那份实例化**: `include/kernel.cpp` 写死 `CombineQuantMode=COMBINE_NO_QUANT`
与 `IsGmm1Interleaved=false`, 所以全部实测常数来自**一个**编译点 —— 这就是标定要按指纹分域的
具体理由。

### 类型化事件图 (IR)

`Event` 能表达依赖/资源/信号量/字节, 但表达方式是**字符串约定**: `"MTE2:c7"` 是执行单元,
`"QUEUE:mte_aic:c7"` 是 L1 缓冲槽, 方向藏在 `"gm_to_l1"` 这个名字里, 而数据依赖与程序序边
在 `deps` 里长得一模一样。约定能跑但不可查询, 换 kernel 时不会报错, 只会悄悄对不上。

`ir/` 把约定提升为类型 (`Engine` / `Pipe` / `MemorySpace` / `TokenKind` / `DependencyKind` /
`TransferDirection`), 并且是**只读视图** —— 不改 `Event`, 不改调度, 所以 39 个 golden 指纹
逐位不变。关键的区分: 执行单元 (容量恒 1 的硬件事实) 与缓冲槽 (容量是编排选择) 用的是同一个
`acquires/releases` 机制, 不分型就没法说"这个容量能不能调" —— 那正是 `EngineQueueDepths` 当初
被当成旋钮的根因。

**表达不了的东西写成明文** (`ir.UNREPRESENTABLE`, 有测试要求每条都讲清为什么):

| 缺口 | 现状 |
| --- | --- |
| 异步发射 vs 完成 | 只有一个 `duration_us`; 用拆相位近似重叠, 真的 issue 开销没有标定 |
| 硬件 flag 身份 | flag 只是某条边上的延迟; 没有身份, 没有 set/wait 配对; kernel 侧 20 多个 flag 与三个 `SyncLatency` 字段的对应关系无记载 |
| 带宽域争用 | 只有标签, 没有共享速率的后果 (信道模型 2026-10-03 停用) |
| 跨核 flag 等待 | 只以依赖边出现; `avoidable_idle_us` 因此只是上界 |

留白会被当成"已经建模了", 所以宁可写出来。

### 结构校验与实测对账

```bash
python tools/compare_trace_structure.py --run bs128
```

`validation/invariants.py` 用 IR 的词表写了 7 条结构不变量 (缓冲槽取还配对且同核、执行单元
容量为 1、共位同核、名字唯一/边存在/无自环、搬运两端已知、零时长不占执行单元), 每条都带
**反例会怎样** —— 因为这些失效是静默的: 取还不配对会让台账漂, 约束悄悄失效, 表现是更快的
排程而不是报错。每条都有反例测试, 两份实现的真实图都过。

`validation/trace.py` + `compare.py` 读实测 trace 并做**结构**对账。先说清能比什么:

| 维度 | 能否比 |
| --- | --- |
| 波数 / 逐专家分布形状 / 核覆盖 / 条数比是否逐专家一致 | 能 |
| 搬运字节 | **不能** —— trace 的 args 只有 rank/local_id/payload/cycles/wave/expert |
| buffer 生命周期 | **不能** —— 只有等待事件这个影子, 没有槽位取/还 |

两个数据事实必须知道:

* **8 个 trace 文件被截断** (两个 bs8192 run 的全部 rank, 都在 7602176 字节处断在记录中间 ——
  同一个字节数, 是采集侧写入上限)。读取器按记录边界救回前面的完整记录并**标记**截断,
  否则"事件数比模型少"会被当成模型的问题。
* **实测 tile 数是模型的 4 倍 (GMM1/ACT) 与 2 倍 (GMM2/COMBINE)**, 逐专家一致, 波数两边都对。
  两边都按 tile 计数 (kernel 的 `MOE_PROFILE_BEGIN` 带 `ProfileTile(mLoc,nLoc)`), 而模型的
  每 m-group tile 数与 kernel 自己的公式**完全一致** (hidden=4608 时 GMM1 是 9, h=5120 时
  GMM2 是 20), 所以差在"每专家几个 m-group"。候选: 采集含多轮 (`config.json5` 里 warmup: 3,
  且 gmm2/combine 恰好分成 3 段各 40 条), 或每专家行数真的更多 (但 `run.log` 的
  `ROUTING_SLICE sent_total=768` 支持模型的 256)。**没解释清之前, 拿这些 trace 对时长没有意义。**

对账工具报的是事实与"需要解释", 不是"口径不同所以没事"。时间段数只作证据不做归一化 ——
同一个 run 的不同 stage 用间隔启发式切出来是 7/10/3/5/14 段, 彼此矛盾; 一个看着精确的错数
比不给数更糟。

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
| `rank_results[r]["idle_decomposition"]`       | 核空闲分解 (forced / avoidable + 逐段给出哪些核在空、当时哪些就绪事件在等) |
| `rank_results[r]["traffic_bytes"]`            | 各通路访存量 (见下) |
| `rank_results[r]["bounds"]`                   | **三个下界与谁绑定** + `violation` (穿透就是模型漏算了代价) |
| `provenance`                                  | 全部常数出处报告    |

`traffic_bytes` 的通路名有语义, 不能混用 —— 塞错地方等于悄悄废掉别处的护栏
(2026-10-05 踩过: 把 COMBINE 的**读**申报到了 `hbm_write` 上, 直接撞掉
"不物化就不写 GM" 那条断言):

| 通路 | 是什么 |
| --- | --- |
| `gm_to_l1` | GMM1/GMM2 的 A 流 + B 流 (与 `bounds` 的算法必搬字节同口径) |
| `hbm_write` | ACT 的量化输出写出 + COMBINE 目的卡是本卡的那些行 |
| `combine_read` | COMBINE 读回 GMM2 tile + 路由元数据 (GM→UB, 既不是 `gm_to_l1` 也不是写) |
| `dispatch_read` / `dispatch_write` | dispatch 的本卡读写 |
| `fab_src:{r}` / `fab_dst:{r}` | 片间: 流量离开本卡 / 到达对端 (跨 rank 共享, 不带 rank 前缀) |

**字节由建图器按算法逐项申报, 相位展开只做重新分配, 绝不从时长倒推** —— 否则换一个
编排旋钮就会改变"搬了多少字节"。这条是不变量, 有测试钉住。

执行时间不含尾段 (counts_export / core_sync / rank_sync / buffer_init / unpermute / finalize)。尾段事件仍在事件图里照常调度, 只是不计入。共享专家的 GMM2 排在尾段 core_sync 之后, 因此也不在执行时间内。

对比两个变体时，先看 `kernel_total_us` 差值，再看 `stage_busy_us` 哪个 stage 变了，最后看 `critical_path` 上卡在哪种等待。

## 常数的出处: 这个数是谁定的

每个常数带一个机器可读的出处标签 (`config/provenance.py`)。类别按**谁定的**分, 不按它
写在哪里:

| 类别 | 谁定的 | 换一份实现会变吗 | 例 |
| --- | --- | --- | --- |
| `spec:` | 硬件规格 / 格式标准 | 不会 | L1/UB/L0C 容量、向量寄存器位宽、MX 量化格式、聚合 HBM 带宽 |
| `algo:` | 算法定义 | 不会 (换算法才变) | SwiGLU 的 gate+up 两个投影 |
| `impl:` | **某一份实现的选择** | **会** | tile 几何、缓冲槽数、档位阈值、每行搬几个元数据字段 |
| `measured:` | 实测 (含争用与开销) | 看标定域 | 各路带宽、固定延迟 |
| `assumed:` | 假设值 | — | 报告里高亮 |

原先 `spec`/`algo`/`impl` 三类**共用一个 `kernel:` 标签**。于是读到一个 `kernel:` 常数时
分不出"物理上只能这样"还是"那份实现这么选的" —— 这正是把实现取值当成"应该的值"的来源
(本仓两次踩过: ACT→GMM2 的 A 流、COMBINE 的元数据字节)。2026-10-04 拆开, 并由
`tests/test_provenance_taxonomy.py` 守住。

**`impl:` 类的数必须能被参数覆盖** (`KernelConfig` / `InstancePolicy` / `ModelOptions`),
模块常数只是那份实现的缺省来源; 模型的缺省值不引用它 (见上文分层)。

### 有读者的常数与只作参考的常数要分开看 (2026-10-05 审计)

出处标签说"这个数是谁定的", 但不说"它现在有没有进公式"。审计发现 **12 个常数没有任何
读者**, 分三类:

| 类 | 常数 | 为什么没读者 |
| --- | --- | --- |
| spec 容量, 只作参考 | `TOTAL_L1_SIZE` / `TOTAL_L0C_SIZE` / `TOTAL_UB_SIZE` / `VEC_REG_WIDTH` | 容量检查走 `KernelConfig.l1_size` 等可覆盖字段 |
| 前导/尾段, 不计入执行时间 | `T_INIT_US` / `T_INPUT_QUANT_FIXED_US` / `T_INPUT_QUANT_PER_TOKEN_US` / `T_CALL_OH` | 这些阶段不在 `kernel_total_us` 口径内 |
| 被参数化之后的孤儿 | `T_FILL_GMM1` (=0) / `L1_TILE_K` / `SCALE_TRANSFER_BYTES` / `GMM2_LAG_MIN_TOKEN_NUM` | 实际取值走 `Calibration` / `KernelConfig` / `InstancePolicy` 的同名字段 |

**改这些常数不会改变任何结果** —— 要改行为得改对应的参数。另外 `BW_SCATTER` 已退役
(2026-10-05): 它曾被用来从**时长倒推**COMBINE 的字节, 那条已删, 现在它不进任何公式、
不进任何申报, 只留复现记录。

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

## 下界: 模型怎么证伪自己 (2026-10-05)

只会推演的模型**无法证伪自己** —— 吐出的数不管对错都长得一样。`analysis/bounds.py` 把
三类事实变成可检查的断言:

| 事实 | 内容 | 与编排的关系 |
| --- | --- | --- |
| 算法事实 | GMM1 乘加 = Σ m_e·h·hidden_dim (hidden_dim = 2I, SwiGLU 的 gate 与 up 都算); GMM2 = Σ m_e·I·h; 必搬字节 = 激活 + 每专家权重至少一次 | **无关**, 只由形状决定 |
| 物理事实 | 一次乘加占 Cube 一拍; 一个字节占带宽一次; 依赖链上的事不能并行 | 无关 |
| 硬件事实 | 每核 Cube 速率、每核载入带宽、聚合 HBM、可用核数 (规格值) | 无关 |

    墙钟 >= max(算力下界, 带宽下界, 依赖下界)

`check_bounds` 缺省 **True**: 穿透就抛 `BoundViolation`。给 `False` 可降级为只记录
(`rank_results[i]["bounds"]["violation"]`), 那是排查用的, 不是出结论用的。

### 它立刻抓出了三处漏账 (全是模型自己的)

1. **GMM2 的权重流进了时长公式却没进字节申报** —— 申报 1660.9MB < 算法必搬 2420.1MB。
   申报量低于算法下界在物理上不可能。
2. **载入相位不占任何资源** —— 墙钟低于带宽下界 26.6%。根因: 把 L1 **缓冲槽数**
   (能提前多少发起) 当成了 **MTE2 管道** (同时能搬几笔, 每核恒 1 条)。
3. **从时长倒推字节** (三处: COMBINE 用退役常数 `BW_SCATTER`, GMM1/GMM2 覆盖掉建图器
   已算对的值) —— 于是开不开相位流水会改变"搬了多少字节"。

### 两条被它推翻的结论

| 先前报的 | 实际 |
| --- | --- |
| 相位流水 **-30.24%**, 最大的杠杆 | **0.00%** (与基线逐位相同)。这个形状带宽绑定, 把载入与计算重叠不会让载入变快 |
| Cube 绑定, 可挖 3.2% | **带宽绑定**, 可挖 5.17% (算力下界只有 25.57us, 带宽下界 1665.37us)。先前拿 `AIC busy/核数` 当算力界, 但 `busy` 含载入时间 |

### 这一层改变了项目的用法

`scenario_basic` 上**95% 的时间在搬字节**, 编排再怎么调最多碰到那 5.17%; 要降那
1665us 只能**少搬** (换 dtype、提高 L2 复用、改物化编排)。所以先看下界再看旋钮 ——
这个判断只用算法 + 物理 + 硬件事实, 不依赖任何未标定系数, 现在就站得住。
而旋钮对比那部分还欠标定: 见下节与 `examples/run_uncertainty.py`。

## 结论的区间: 不能判定的比较要说出来

`analysis/sensitivity.py` 把未标定输入分成两类, **这个区分是关键**:

* **有区间** (实测给了范围只是没定到点) → 可以传播成 Δ 的区间
  `bw_combine_remote` 8600 [4500, 9500]、`bw_l1_gm` 51900 [49200, 54400]
* **无区间** (连范围都没有) → **不编一个范围传播**, 那是把无知包装成精度; 只如实标注
  "这个结论依赖某个没测过的量"。`late_bind_fetch_us` (R7)、`cube_mac_per_us` (R1)

于是 `Interval.decidable` 要求两件事: 符号确定**且**不依赖任何无区间输入。实测
(`examples/run_uncertainty.py`, scenario_basic):

```
GMM2 攒 2 个 tile     Δ = -0.57% [-0.57, -0.55]   可判定
combine 逐专家 (整片)  Δ = +0.51% [+0.47, +0.94]   可判定
晚绑定 (AIC+AIV1)     Δ = -1.34% [-1.39, -1.25]  依赖未测量 late_bind_fetch_us  **不可判定**
UB 深度 2            Δ = -0.30% [-0.31, -0.28]   可判定
```

两点反直觉的事实:

1. **带宽区间接近 2 倍, 推到 Δ 上只有 ±0.02 个百分点** —— 比较的两边共用同一个带宽,
   大部分抵消了。所以比较类结论比绝对时长稳健得多。
2. **唯一不可判定的那条, 区间反而最窄**。窄区间不等于可信 —— 这正是把两类未知分开的
   价值: 只看区间宽度会把晚绑定误判成最稳的结论之一。

口径: one-at-a-time 局部敏感度, 不覆盖输入之间的交互。

## 精度边界

**本节数字是旧公式下的结论。** 搬运口径已于 2026-10-04 按三个实测点定为相加 (见上, GMM1
每 tile 的误差 +0.4%~+3.5%), 但整体墙钟校准还缺两样: 实测的 Cube 速率; 以及按并发核数
分档的 `BW_L1_GM` (A 流斜率显示它随并发变: 28 核 45.3 / 18 核 37.0 GB/s)。

旧公式下: `profiles.MEGAMOE_A8W8` 这组取值 (即那份实现) 在 B≤128 标定域内各 stage busy
误差 ±5%，墙钟偏差 -6~-8%。**注意缺省值不是这组取值** —— 缺省是"最少假设"，与实测对齐要
显式引用 profile (场景文件里写 `profile = "megamoe-a8w8"`)。

**要哪些上板 run 才能把系数定下来**: 见 `docs/calibration_runs.md` —— 每项写清固定什么、
扫什么、从 trace 取哪个量、能定哪个常数、怎么验。现在结构基本完整, 瓶颈在系数:
Cube 效率、`BW_L1_GM` 按并发分档、COMBINE 的落点跨度、`BW_REMOTE_WRITE`。

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

> **怎么用** 看 [`docs/USAGE.md`](docs/USAGE.md) —— 入口、场景文件怎么写、输出怎么读、
> 旋钮速查、精度边界。本文件讲的是**为什么这样建模**。

六个入口, 按"你想干什么"选:

| 想干什么 | 跑什么 |
| --- | --- |
| 看一个形状跑多久 | `python examples/run_basic.py` |
| 写场景文件、改旋钮对比 | `python examples/run_scenario.py` ← 日常 |
| 扫一片编排, 看每个选择值多少钱 | `python examples/run_design_space.py` |
| **流水编排逐旋钮** (空闲分解 + 关键路径归因) | `python examples/run_pipeline_study.py` |
| **结论还站不站得住** (未标定输入 → Δ 的区间) | `python examples/run_uncertainty.py` |
| 与实测 run 逐 stage 对账 | `python tools/compare_measured.py <run_dir> <场景.toml>` |

```bash
cd moe-cost-model
pip install -e .          # 或直接 pytest (pyproject 已配 pythonpath)
pytest tests/             # 356 项测试 (5 项需 tiling 真值, 见下), 约 15 分钟
python examples/run_scenario.py    # 场景文件 + 改旋钮对比
python examples/run_basic.py       # 底层入口
```

## GMM2 沿 K 维分段就绪 (编排选择, 非物理约束)

GMM2 的 K 就是 GMM1 切分的那个 N 轴 (`k_gmm2 = hidden_dim / activation_n_half`), 所以一个
ACT tile 只产出 GMM2 在 K 上 1/ceil(k/TILE_N) 的部分, GMM2 要累完整个 K 才有结果。**分几段
独立就绪是编排选择**: L0C 本来就沿 kL1 分块累加 (kernel 的 `ProcessTileL1`), 所以让第 j 段
只等覆盖自己 K 范围的 ACT 在物理上可行。

`StageLink("activation", "gmm2", readiness=N)` —— 2026-10-04 起由这条边给出
(原先是 `ModelOptions.gmm2_k_segments`, 已删除; 三个各自为政的旋钮
`gmm2_k_segments` / `act_to_gmm2` / `InstancePolicy.gmm1_activation_depth`
合并成"一条边三个问题", 见 `config/links.py`):

| `readiness` | 含义 |
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

## 核空闲分解: 哪些消不掉, 哪些是某个编排选择的标价

"核不能有空闲"作为绝对约束在逻辑上不可能 —— 事件必须等前置完成, 而 t=0 时 GMM1 还在等
`dispatch_ready` (AIV1 产出), 所以开头所有 AIC 必须空着。能成立的不变量是
**work-conserving**: 核不得在"存在已就绪的活"时空闲。

`rank_results["idle_decomposition"]` 按角色池 (AIC / AIV0 / AIV1) 给出:

| 字段 | 含义 |
| --- | --- |
| `busy_us` | 占用核·us; 与 `resource_busy_us` 的同角色合计逐位一致 |
| `forced_idle_us` | 此刻池里**没有**已就绪未开始的事件 → 消不掉, 只能改 DAG 结构 |
| `avoidable_idle_us` | 有就绪的活却有核空着 → **换 tile→核 绑定方式可回收**。注意它**不是"实现有 bug"** —— 静态分派下这是那个编排选择的标价 (见下) |
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

### `avoidable` 不等于"实现做错了" (2026-10-05 订正措辞)

本节原来的小标题是"哪些是**真浪费**", 那个说法会让人读成"kernel 有 bug"。不是。
静态分派 (`late_bind_pools=()`, 复现那份实现的 `startBlockIdx` 轮转) 下, 每个核按 block 号
算出自己该干哪些 tile, 干完就收工 —— 核空着的时候它**名下没有活**, 那些就绪的 tile 不归它。
从 kernel 自己的视角这不是"本不应空闲而空闲", 是**尾部负载不均**。

实测 trace 证明真实 kernel 确实不是工作守恒的 (20260930, rank0, AIC 事件):

| run | 每核忙碌 min / 中位 / max | max/min | 每核最后事件结束的散布 | 尾部空闲 |
| --- | --- | --- | --- | --- |
| bs36 | 122.6 / 138.6 / 171.0 us | **1.39x** | 28.5us (窗口 11.6%) | 245.2 核·us = 3.6% |
| bs128 | 177.8 / 206.9 / 266.3 us | **1.50x** | 47.5us (窗口 9.9%) | 389.9 核·us = 2.9% |

**1.4~1.5 倍的每核忙碌差本身就是证据**: 如果核之间能互相抢活, 忙碌时间会被抹平到一个
任务时长之内。差 1.5 倍只能是静态分派 + 分派不均。

所以 `avoidable` 的正确读法是"**换绑定方式能回收多少**", 而且回收要付代价 ——
晚绑定的取活开销 (原子加/核间同步) 现在可以计费 (`PrimitiveCosts.late_bind_fetch_us`),
缺省 0 表示本模型没有声称它是多少, 要定它见 `docs/calibration_runs.md` 的 **R7**。
在那之前"开晚绑定更快"这个结论是**不可判定**的。

两类已声明的虚报 (`avoidable` 是上界): 共位约束 (GMM1 落核 X 则 ACT 必须落 AIV0:X) 与
相位组绑定 (cube 相位只能在 load 相位填过 L1 的那个核上算)。见 `analysis/idle.py`。

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
| `ModelOptions.combine_layout` | 写出落点布局: `"token_scatter"` (缺省, 落点由 token 全局编号定, UNPERMUTE 顺序读) / `"expert_contiguous"` (按专家连续写, 读侧改 gather)。布局决定申报的**跨度**; 跨度的代价系数 (`AnalyticalCombineCosts.scatter_us_per_row`) 缺省 0, 要由扫 token 数的 run 定。读侧代价尚未建模 —— 见 `docs` 缺口 10 |
| `ModelOptions.combine_granularity` | 一个 combine 事件覆盖多少工作: `"per_tile"` (缺省, 与 GMM2 tile 1:1 配对、与计算交错) / `"per_expert"` (一个专家切片一个事件、等自己那片 GMM2 做完)。与角色正交。实测攒批省 0.5% 元数据字节但墙钟 +30.5% —— 不过模型还算不出它的主要好处 (写侧跨度, 见 `docs` 缺口 10) |
| `ModelOptions.roles` | `stage -> 执行角色` 映射 (`RoleAssignment`): 哪个 stage 跑在 AIC / AIV0 / AIV1 上。矩阵乘只能在 AIC (物理), ACT 与它的 GMM1 必须同核 (Fixpipe), 其余可换。**实测在 A8W8 主路径上换角色不改墙钟** —— 两个向量核利用率都不到 6%, 关键路径在 AIC |
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

## 旋钮覆盖: 每个可调参数都必须能动模型 (2026-10-05)

这个项目是给算子工程师改**编排 / 编译期 / 运行期**参数用的, 所以一个旋钮扫出
**0 收益**必须能分清是哪一种 0。四种意思, 指示完全相反:

| 判定 | 意思 | 下一步 |
| --- | --- | --- |
| 生效 | 每个形状上都动 (与形状无关) | 这个取舍可以照着做 |
| 生效* | 至少一个形状上动, 本形状没有作用对象 | 换形状再扫: 只有一波谈不上超前几波, 只有一个 K 块谈不上逐块就绪 |
| 被拒 | 模型显式拒绝该取值 (缺标定 / 这条路径没实现) | 拒绝是诚实的 |
| 动不了 | 模型里**没有可表达的后果** | **陷阱**: 这个 0 是模型的空白, 不是硬件的事实 |

```bash
python tools/knob_audit.py --quiet    # 五个互补形状 x 全部旋钮
```

旋钮树自动走 (`dataclasses.fields` + `scenario._NESTED`), 所以**新加的字段自动进审计**;
全量判定钉在 `knob_audit.EXPECTED`, `tests/test_knob_coverage.py` 守着它:
新旋钮忘了接线、老旋钮被改没了、"动不了"的声明过期了, 三种都会红。

扫法三条 (为什么结论能信):

1. 从**本场景的生效值**出发扰动, 不是从 dataclass 缺省值出发 —— 场景带 `profile` 时
   两者不同, 拿缺省值当基线会把"值根本没变"误判成"没有读者"。
2. 一个旋钮给**一串**候选取值 —— 翻倍常落在无语义的档上 (`l1_buf_num` 2→4 与 2 同构,
   2→1 才是关 ping-pong; `swizzle_direction` 只有 0/1 两档)。
3. 比对五项: 墙钟 / 事件数 / 事件名集合 / 逐事件时长 / 逐信道字节。只看墙钟会把
   "结构变了而两边恰好等长"当成没动。

建立这一层当天抓出三件事, 全是它要防的那一类:

* **`EngineQueueDepths` 任何取值都无后果, 已删。** 持核事件独占 AIC/AIV0/AIV1,
  同核在途数恒 ≤ 1; 相位拆分后的 load 相位又刻意不继承 `Q:*` (继承会让容量 1 的引擎
  信号量卡死 L1 缓冲深度)。证据: golden 的 `pipeline_engine_queue2` 与 `pipeline_split`
  指纹**逐位相同** —— 那个 case 从来什么都没测到。容量现在写死 1, 要表达"更深的队列"
  得先有发射开销这类物理后果, 模型里没有, 给个旋钮只会让扫描得出"深了也没用"的假结论。
* **`KernelConfig.topk_weights_prefetch` 没有读者, 已删** (硬门查的是
  `ModelOptions.topk_weights_prefetch`)。
* **`options.roles` 与 `options.epilogue_overheads` 在场景文件这条日常路径上写不出来**
  (报"应为数值"), 只能在 Python 里构造对象 —— 于是"哪个 stage 跑在哪个核上"这一类编排
  在场景扫描里根本到不了。已接上, 文件里写 `[options.roles]` 下 `combine = "AIV0"`。

目前唯一标为"动不了"的是 `options.combine_layout`: 写侧只经 `scatter_us` 的
`(spread_slots/m) ** scatter_exponent`, 而 `scatter_exponent` 缺省 0 使指数项恒 1,
两种布局算出同一个数 —— 0 不是保守, 是实测把"落点跨度"这个机制否掉了; 读侧 UNPERMUTE
从顺序读变 gather 的代价完全没建模 (缺口 10)。

## 回归保护

`tests/test_golden.py` 对 40 个配置核对指纹, 2026-10-05 起锁**四类**东西:

| 锁什么 | 字段 | 为什么 |
| --- | --- | --- |
| 时长与排程 | `total_us` / `dag_end_us` / `stage_busy_us` / `schedule_sha256` (每个事件的起止、等待归因、关键父事件) | 任何一位浮点差异都会失败 |
| **访存量** | `traffic_bytes` (逐通路) | 字节口径的改动不改时长, 原先完全抓不到 |
| **下界** | `bounds` (三个下界 + `binding` + `violation`) | 穿透物理下界要立刻显形, 不等全套 |
| **出处** | `provenance_summary` (类别计数) + `provenance_sha256` (全部常数的名字/取值/**完整标签文本**) | 标签文本是承诺 (域限制、待重标), 和数值一样该被保护 |

后三类是补上来的, 起因是一个实际踩到的盲区: 原指纹只锁时长, 于是连着两次拿
"40 个 golden 零 diff" 当提交依据, 却都漏掉了同样两条失败 —— 一条查 `traffic_bytes`,
一条查出处标签文本, 两者都不在指纹里。补上之后用注入法验证过:

* 悄悄不申报 COMBINE 的读回 → 30 秒的 golden 报 **2/40 DIFF** (原先 0);
* 把某个常数出处里的"域受限"三个字删掉 → 报 **34/40 DIFF** (原先 0)。

两层出处都要: 类别计数只在**分类**变了时变, 同类别内的文本改动得靠 sha256。
反馈从 18 分钟压到 30 秒。覆盖 MTE / Layered 两条路径、波偏移、编译期旋钮、三种调度策略、分核与打包策略、相位流水、计数信号量、任务转移。

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
