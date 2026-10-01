# MegaMoE Cost Model

Ascend NPU MegaMoE 流水编排性能评估工具。算子工程师改参数、改编排，模型算出执行时间，对比判断性能收益，不必逐个上板实测。

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
| GMM1  | `max(A流/BW, 计算/R_cube)`   | `A流/BW + 计算/R_cube + restart` |
| GMM2  | `计算/R_cube`                | `计算/R_cube`                    |

- A 流 = m·K 字节; GMM1 计算 = 2·m·cols·K MACs; GMM2 计算 = m·cols·K2 MACs。
- GMM2 只有计算: A 从 UB 直达 L0A, 片上带宽约 500 GB/s 且每核独占, 搬运时间忽略不计。restart 是 L1 换块的停顿, A 不经 L1, 所以 GMM2 没有 restart, 时长与 L1 缓冲数无关。
- B 流 (权重搬运) 不建模: 认为被其他任务的执行掩盖。
- `R_cube` (`cube_mac_per_us`) 必填, 无缺省。仓库里没有标定过的 Cube 速率, 示例与测试里的 `2.7e7` 只是占位值。

相位流水 (`options.pipeline`, 可选) 与闭式同口径:

| stage | MTE 队列 (L1 缓冲槽) | `gm_to_l1` 信道 | Cube 队列 |
| ----- | -------------------- | ----------------- | --------- |
| GMM1  | 占                   | A 流字节 m·K      | 占        |
| GMM2  | 占 (B 权重)          | 不占              | 占        |

占用与计时是两回事: A 与 B 都经 L1 进 L0, 所以两个 stage 的 tile 都占 L1 缓冲槽; B 流搬运不计时、不计信道流量, 但权重仍在 L1 里占着位置。

- `queues.mte_aic > 1` 时 GMM1 拆相位: tile 内 load 与 cube 并行, 单 tile 时长 = `max(A流, 计算)`; 后一个 tile 的 A 流可在前一个 tile 计算时预取。
- load / cube 相位时长取自 GMM 公式的分解, 载入带宽与 Cube 速率只有公式这一个来源。
- `l1_buf_num = 1` 与 `queues.mte_aic > 1` 互相矛盾, 同时给会报错。
- `gmm1_tile` 换成自定义函数后没有 A 流/计算分解, 拆相位或开 `gm_to_l1` 信道会报错。

### 共享专家

`[workload]` 里设 `shared_expert_num = 1` 打开。事件与依赖:

| 事件 | 占用 | 时长 | 依赖 |
| --- | --- | --- | --- |
| 共享 GMM1 tile | AIC 核 | GMM1 公式 | 无, 从 0 时刻开始 |
| 共享 ACT tile | AIV0 核 | ACT 公式 | 同 tile 的共享 GMM1 |
| 门控 | 无 | 0 | 全部共享 ACT |
| MoE dispatch_call | AIV1 核 | — | 门控 (MTE: 每个波; Layered: 仅首波接收) |
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

波粒度只有两种：MTE 路径按 256 行组切波，Layered 路径按专家范围切波。ACT 总与 GMM1 同波。同核程序序没有建成依赖边，靠资源互斥保序。片间 fab 信道占位关闭，跨卡争用不建模。

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

## 精度边界

**GMM 公式于 2026-09-29 换口径 (见上), 本节数字是旧公式下的结论, 新公式尚未对实测校准。** 校准需要两样东西: 实测的 Cube 速率; 按新口径重新标定的 `BW_L1_GM` (现值 51.9 GB/s 是在 A 流与 B 流一起计费的旧口径下反解的)。

旧公式下: 默认配置对应当前 kernel 行为，在 B≤128 标定域内各 stage busy 误差 ±5%，墙钟偏差 -6~-8%。

偏离默认的取值为未验证取值：模型照常给出预测，但结论需实测抽检。标定域外的已知失效：B=1024 时 COMBINE 偏差 +114~246%（BW_SCATTER 单点标定域外），GMM1 系统性高估 +4~10%（B 矩阵逐 tile 计费）。

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
pytest tests/             # 177 项测试 (5 项需 tiling 真值, 见下), 约 1 分钟
python examples/run_scenario.py    # 场景文件 + 改旋钮对比
python examples/run_basic.py       # 底层入口
```

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

`tests/test_golden.py` 对 44 个配置核对调度指纹: 每个事件的起止时刻、等待归因、关键父事件取 sha256, 任何一位浮点差异都会失败。覆盖 MTE / Layered 两条路径、波偏移、编译期旋钮、三种调度策略、分核与打包策略、相位流水、容量与信道、片间信道、任务转移。

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
| 相位流水 + 信道, 4 rank × 64 专家, B=1024 | 3.5 万 | 约 4 s |

片间信道关闭、无重构钩子、使用内置调度策略时, 各 rank 独立调度。`idle_core_stealing` 每次提交都扫描全部未提交事件, 耗时随事件数平方增长: 6800 事件约 30 s。

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
│   │   └── policies.py          #   EarliestStart / CriticalPathFirst / PriorityByStage
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
