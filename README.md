# MegaMoE Cost Model

Ascend NPU MegaMoE 流水编排性能评估工具。算子工程师改参数、改编排，模型算出执行时间，对比判断性能收益，不必逐个上板实测。

## 它解决什么问题

算子的执行时间由三类决策共同决定: 硬件参数、编译期模板参数、运行期编排 (切分、就绪
粒度、分核、配速)。上板实测回答"这一版多快", 回答不了另外两个问题:

- **为什么是这个数** —— 哪条依赖链最长, 链上每一段在等什么;
- **改哪一项能变快多少** —— 换一个编排选择的差值, 不重新编译、不重新上板。

本工具把一次执行展开成显式的事件图再排程, 所以能回答后两个。相对实测, 它多给四样
东西, 每一样都能在输出里直接查到:

**1. 逐事件归因。** 每个事件带 `dependency_wait_us` / `resource_queue_us` /
`capacity_wait_us` 与 `critical_parent` / `critical_reason`。"为什么慢"可以逐层追问:
哪条路径最长 → 路径上谁在等 → 等的是数据依赖、资源独占, 还是缓冲容量。

**2. 反事实对比。** 同一形状改一个编排选择跑两次, 差值就是那个选择的代价或收益。
一条边一行、一个旋钮一列的批量扫描见「扫一遍, 看每个选择值多少钱」; 沿 K 分段就绪的
实测收益见「GMM2 沿 K 维分段就绪」。

**3. 自我证伪。** 每次运行同时给出两项一致性检查, 结果不自洽时模型报告自己错了,
而不是给一个看起来合理的数:

- **物理下界**: 算力 / 带宽 / 最长依赖链三条下界取最大, 墙钟低于它直接抛
  `BoundViolation`。`examples/scenario_basic.toml` 上三条分别是 25.57 / 1665.37 /
  68.06 µs, 带宽绑定, 模型给 1751.48 µs —— 高于下界 5.2%, 这是合理区间。
- **工作守恒度量**: 把核空闲分成 `forced` (没有就绪的活, 或容量不允许) 与
  `avoidable` (有就绪的活、又有核空着)。同一场景上 `forced` 909.6、`avoidable`
  1022.8 核·µs —— 后者是那份实现的静态分核方式的标价, 不是建模误差; 换成派发时刻
  绑定 (缺省) 后它归 0。

**4. 可信度标注。** 每个常数带出处标签, `scenario_basic` 上的分布是
`measured 24 / impl 23 / user-supplied 18 / spec 7 / assumed 3 / algo 1 / derived 1`,
并有 `provenance_sha256` 锁住标签文本。没有标定到点的输入单列在
`analysis.sensitivity.UNCERTAIN_INPUTS` 里: 有实测区间的参与区间传播, 连区间都没有的
(晚绑定取活开销、Cube 可达效率、分段同步开销) 会让依赖它的结论判为**不可判定**,
而不是给一个数。换编译点时标定域 (`CalibrationDomain`) 报 `out_of_domain` 而不是
沿用旧系数外推。

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
pytest tests/             # 547 项测试 (5 项需 tiling 真值, 见下), 约 25 分钟
python examples/run_scenario.py    # 场景文件 + 改旋钮对比
python examples/run_basic.py       # 底层入口
```

## 工作原理

### 被排程的对象

一次执行展开成一张事件图。每个事件 $e$ 带五样东西:

| | 含义 |
| --- | --- |
| $d_e$ | 时长, 由闭式物理公式给出 |
| $R_e$ | **独占**的资源集合, 占用区间严格是 $[s_e,\; s_e+d_e)$ |
| $D_e$ | 前驱集合, 每条边带同步延迟 $\lambda(e,p)$ |
| $A_e$ | 计数信号量需求 $\{(\tau,k)\}$, **可跨事件持有** (取与还不必是同一个事件) |
| 落核约束 | `colocate_with` (与某个具名事件同核) / `core_group` (同组同核) |

这五样就是模型的表达能力上界: 表达不出来的约束, 模型不声称。$R_e$ 与 $A_e$ 的区别在
持有时长 —— 前者严格等于事件自身那一段, 后者可以跨事件, 所以只有后者能表达"缓冲槽
被下游占着、核空着也进不来新活"。

### 排程递推

贪心表调度。ready 集合里每个事件先算基点

$$t_{\text{base}}(e)=\max\Big(\max_{p\in D_e}\big(\mathrm{end}_p+\lambda(e,p)\big),\;\;\max_{r\in R_e}\mathrm{free}(r)\Big)$$

再做容量准入 —— 不是判真假, 而是沿未来的归还时刻找最早可行点:

$$s_e=\min\Big\{\,t\ \ge\ t_{\text{base}}(e)\ :\ \forall (\tau,k)\in A_e,\ \ \mathrm{outstanding}(\tau,t)+k\le \mathrm{cap}(\tau)\Big\}$$

$\mathrm{end}_e=s_e+d_e$。然后从 ready 集合里取使策略键最小的那个提交 (缺省
`EarliestStart` 的键是 $(s_e,\ \text{order},\ \text{name})$), 提交时更新
$\mathrm{free}(r)$、写信号量台账、递减后继入度。墙钟 $=\max_e \mathrm{end}_e$。

$R_e$ 里允许池占位符 `ROLE:*`: 这时成员在**提交那一刻**才选定, 取使
$\max(\mathrm{free}(c),\ \text{该核信号量的最早可行时刻})$ 最小的核。两个约束必须
一起取最小 —— 分别取最小会指向不同的核, 算出的 $s_e$ 没有任何单核真能满足。

### 时长的来源

$d_e$ 与并发无关, 由闭式公式给出, 所有并发、排队、空闲都是上面那个递推的产物:

- GMM tile: $d=\max(\text{载入},\ \text{计算})$, 载入 $=(A\text{ 字节}+B\text{ 字节})/\mathrm{BW_{L1\_GM}}$,
  计算 $=\mathrm{MAC}/\texttt{cube\_mac\_per\_us}$;
- 搬运: 字节 / 带宽 + 启动开销;
- combine: 字节 + 可选的逐行散射系数。

### 两项检查的定义

$$\mathrm{LB}=\max\Big(\frac{\sum \mathrm{MAC}}{P\cdot \text{rate}},\ \ \frac{\sum \text{bytes}}{P\cdot \mathrm{BW}},\ \ \text{最长依赖链}\Big)$$

工作守恒度量要先定义"可动时刻", 它**不含**"自己要的资源空出来"那一关:

$$\mathrm{actionable}(e)=\max\big(\text{依赖就绪},\ \text{信号量可准入}\big)$$

$$\text{avoidable\_idle}=\int \Big|\big\{\,r\ \text{在}\ t\ \text{空闲}\ :\ \exists e,\ \mathrm{actionable}(e)\le t< s_e,\ e\ \text{可落到}\ r \big\}\Big|\,\mathrm{d}t$$

缺省配置 (派发时刻绑定) 下要求它恒为 0。这个定义对"$\mathrm{actionable}$ 里混进资源
等待"极其敏感, 所以图里起不了约束的计数信号量必须删掉 —— 判据与实测见
`scheduler/normalize.py`。

### 这个方法本身的边界

贪心表调度**对输入不单调**: 缩短一个时长或放松一个约束, 墙钟可能变大。实测两例 ——
一条依赖边上加 0.01 µs 让墙钟变化 −2.81%; 就绪粒度均分 3 段比 2 段差 1.0%, 而 4 段又
回到 2 段的值。所以**低于几个百分点的差值不可读**, 除非同一绑定方式、同一调度策略,
且差值大于这类抖动。这条写成了测试, 不用容差掩盖。

## 使用方式

### 三条输入通道

| 通道 | 适用 | 入口 |
| --- | --- | --- |
| 场景文件 + 点分路径覆盖 | 日常 | `load_scenario("x.toml")` / `sc.with_overrides({"policy.gmm2_lag_waves": 1})` |
| profile 起步 | 要以某份实现的编排为底再改 | `MEGAMOE_A8W8.with_options(...)` / 文件里写 `profile = "megamoe-a8w8"` |
| Python API | 要全部旋钮, 或程序化扫描 | `simulate_routing_counts(...)` |

写错字段名、类型不对、策略名不存在都立即报错并给出提示, 例如
`policy.gmm2_lag_wave: 未知字段, 是否想写 'gmm2_lag_waves'?`。

### 九个运行入口

| 命令 | 回答什么 |
| --- | --- |
| `python examples/run_scenario.py` | 一个场景 + 改几个旋钮的对比 |
| `python examples/run_basic.py` | 底层 API 的最小例子 |
| `python examples/run_design_space.py` | 编排扫描: 每个方案一行, 给 Δ / Δ% / 关键路径变化 / 最大等待 / 访存量差 |
| `python examples/run_pipeline_study.py` | 同上, 按"单核内 / stage 之间 / 波之间 / 工作落核"分组 |
| `python examples/run_uncertainty.py` | 每条结论标成稳定或不可判定 |
| `python tools/knob_audit.py` | 旋钮审计: 每个旋钮在五个形状上分成生效 / 需对的形状 / 被拒 / 未建模 |
| `python tools/diagnose.py` | 单形状诊断: 瓶颈在哪、能改什么、预计收益范围 |
| `python tools/compare_trace_structure.py` | 与实测 trace 做结构比对 (波数、逐专家分布、核数、条数比) |
| `python tools/gen_golden.py --check` | 回归: 40 个调度指纹逐位比对, 35 秒 |

### 一次运行给出什么

不是一个数, 而是一组可以继续追问的量: 时间 (`kernel_total_us` / 每 rank
`total_us` / `stage_busy_us`)、事件级明细 (起止与三类等待、关键父事件)、结构
(`critical_path` / `wave_count`)、物理量 (`traffic_bytes` 逐通路字节申报、`bounds`
三条下界与是否穿透)、自检 (`idle_decomposition`)、可信度 (`provenance`)。字段逐个
解释见「输出解读」, 那张表由测试保证覆盖 `simulate()` 真实返回的每个字段。

### 规模与耗时

| 规模 | 时间 |
| --- | --- |
| 2308 事件 (典型形状, 静态分核) | 0.2 s |
| 6599 事件 (`scenario_basic`) | 3.5 s |
| 49308 事件 (`pipeline_large_split`) | 8.3 s |
| 同一形状改成派发时刻绑定 | 约慢 72 倍 (瓶颈在选核) |
| 40 个 golden 指纹全跑 | 35.5 s |
| 全套测试 (547 项) | 约 25 分钟 |

给定输入**逐位可复现**: 没有任何随机来源不带种子 (`routing = "random"` 用
`random.Random(seed + src)`), 所以调度结果能用 sha256 锁住做回归。

### 什么情况下结论不再成立

- 只改编排 / 编译期 / 运行期**参数** (`ModelOptions` / `KernelConfig` /
  `InstancePolicy`) —— 模型自动给结果, 准确性在标定域内有保障;
- 改了 **C++ 控制流、同步协议、buffer 复用方式, 或流水阶段结构** —— 必须重新生成
  实现描述或 trace schema, 仅靠几个 Python 参数保证不了精确。

### 场景文件: 完整示例

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
cube_mac_per_us = 2.7e7      # 占位示例; 仓里没有标定值。缺省 0 = 计算项不生效

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

### 扫一遍, 看每个选择值多少钱

```bash
python examples/run_design_space.py
```

```
== Ascend 950PR  (聚合 HBM 1.6 TB/s, 28 核)
方案                          时长          Δ       Δ%  关键路径变化 (stage)              总忙碌变化               最大等待              访存量差            不变量
基线 (最少假设)               326.64      +0.00    +0.0%  -                           (工作量不变)             - 0us             -               ok | HBM 68% (基线)
GMM2 首块先开工              326.64      +0.00    +0.0%  -                           (工作量不变)             - 0us             -               ok | HBM 68%
GMM2 均分 4 段就绪           315.87     -10.77    -3.3%  gmm2-20.2, activation+9.4   (工作量不变)             - 0us             -               ok | HBM 70%
GMM2 逐 K 块就绪            315.87     -10.77    -3.3%  gmm2-20.2, activation+9.4   (工作量不变)             - 0us             -               ok | HBM 70%
UB 深度 2 (交织路径)          296.34     -30.31    -9.3%  gmm1-75.8, gmm2+45.5        (工作量不变)             - 0us             -               ok | HBM 75%
UB 不设限 (上界)             296.34     -30.31    -9.3%  gmm1-75.8, gmm2+45.5        (工作量不变)             - 0us             -               ok | HBM 75%
ACT 不物化 (留片上)          2017.39   +1690.74  +517.6%  gmm1+1212.2, gmm2+318.2     gmm2-1364           capacity 160us    gm_to_l1-70.78MB  ok | HBM 9%
波间全核对齐                  405.12     +78.48   +24.0%  gmm1+75.8, gmm2-45.5, combine+16.1  (工作量不变)             capacity 9us      -               ok | HBM 55%
静态发牌                    375.81     +49.17   +15.1%  gmm1+75.8, gmm2-45.5, activation+9.4  (工作量不变)             capacity 9us      -               违反 AIV1 264 | HBM 59%
GMM1 攒 2 个 tile         334.59      +7.95    +2.4%  activation+7.9              activation-40       - 0us             -               ok | HBM 66%
GMM2 攒 2 个 tile         391.89     +65.24   +20.0%  gmm1+75.8, gmm2-45.5, combine+16.1  combine-1           capacity 19us     combine_read-0.12MB  ok | HBM 56%
combine 攒 2 个 tile      342.72     +16.08    +4.9%  combine+16.1                combine-1           - 0us             combine_read-0.12MB  ok | HBM 65%
combine 逐专家 (整片)        632.08    +305.44   +93.5%  combine+305.4               combine-1           - 0us             combine_read-8.11MB  ok | HBM 35%
ACT 攒 2 个 (轮转下空操作)      296.34     -30.31    -9.3%  gmm1-75.8, gmm2+45.5        (工作量不变)             - 0us             -               ok | HBM 75%
那份实现 (MEGAMOE_A8W8)     364.67     +38.03   +11.6%  gmm1+75.8, gmm2-50.5, activation+9.4  dispatch+119, epilogue+7  capacity 9us      -               违反 AIC 1645 | HBM 61%
   不变量成立的最快方案: UB 深度 2 (交织路径)  296.34 us (-9.3%), 需要聚合带宽 1.19 TB/s = 规格的 75%
```

每一行回答的不是"多少 us", 而是: **收益落在关键路径的哪个 stage**、改完卡在什么等待上、
少搬多少字节、以及这个方案下模型自己的不变量守住了没有 ——"违反"那一行的时长偏慢,
收益不可比。`design_space()` / `format_design_space()` 给的就是这张表。

一个缺省带来的直接后果: 缺省晚绑定满足模型的不变量 ——
**决不出现"某 tile 前置依赖已完成、又有核空闲, 它却还在等"** (`avoidable_idle_us == 0`)。
静态发牌做不到, 它是一种实现的分核方式, 要评估就显式给 `late_bind_pools=()`。

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
| `rank_results[r]["resource_busy_us"]` / `resource_idle_us` / `resource_span_us` | 逐资源 (如 `R0.AIC:7`) 的忙 / 闲 / 跨度 |
| `rank_results[r]["stage_first_start_us"]` / `stage_last_end_us` | 各 stage 的首个开始 / 最后结束时刻 |
| `rank_results[r]["events"]`                   | 全部 `ScheduledEvent` (名字、起止、落核、meta) |
| `rank_results[r]["waves"]` / `wave_count` / `m_groups_per_wave` | 波计划: 逐波范围、波数、每波组数 |
| `rank_results[r]["gmm2_lag_active"]`          | 本次运行 GMM2 滞后有没有真的生效 |
| `rank_results[r]["dispatch_ready_tiles"]`     | 每个 (波, 专家, m-group) 的就绪时刻与依赖 |
| `rank_results[r]["implementation"]`           | 这条结果的身份: 实现 id / 编译指纹 / 运行拓扑 / 计时终点 stage |
| `slowest_rank` / `scenario`                   | 最慢 rank 的号; 本次运行的场景对象 |
| `provenance`                                  | 全部常数出处报告    |

`traffic_bytes` 的通路名有语义, 不能混用: 申报到错误的通路上会使别处的一致性检查失效
(例如把 COMBINE 的**读**申报到 `hbm_write` 上, 会触发"不物化就不写 GM"那条断言):

| 通路 | 是什么 |
| --- | --- |
| `gm_to_l1` | GMM1/GMM2 的 A 流 + B 流 (与 `bounds` 的算法必搬字节同口径) |
| `hbm_write` | ACT 的量化输出写出 + COMBINE 目的卡是本卡的那些行 |
| `combine_read` | COMBINE 读回 GMM2 tile + 路由元数据 (GM→UB, 既不是 `gm_to_l1` 也不是写) |
| `act_readback` | **只在 `topk_weights_prefetch` 开着时存在**: ACT 从 GM 读回 GMM1 的输出 + 本行块的 topk 权重 (GM→UB, 同理单列一条) |
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

`spec` / `algo` / `impl` 必须分成三类而不是合成一个"来自 kernel": 合在一起就分不出
"物理上只能这样"与"那份实现这么选的", 而后者是可以改的决策变量。
分类由 `tests/test_provenance_taxonomy.py` 守住。

**`impl:` 类的数必须能被参数覆盖** (`KernelConfig` / `InstancePolicy` / `ModelOptions`),
模块常数只是那份实现的缺省来源; 模型的缺省值不引用它 (见「建模原则」的三层划分)。

### 模块常数: 哪些有读者, 改了会发生什么

出处标签说"这个数是谁定的", 不说"它现在有没有进公式"。按"改了它会发生什么"分三类:

**一、改了什么都不会发生**

| 常数 | 实际取值走哪里 |
| --- | --- |
| `TOTAL_L1_SIZE` / `TOTAL_L0C_SIZE` / `VEC_REG_WIDTH` | 容量检查走 `KernelConfig.l1_size` 等可覆盖字段 |
| `T_INIT_US` / `T_INPUT_QUANT_FIXED_US` / `T_INPUT_QUANT_PER_TOKEN_US` / `T_CALL_OH` | 这些阶段不在 `kernel_total_us` 口径内 |
| `T_FILL_GMM1` (=0) | `Calibration` 的同名字段 |
| `BW_SCATTER` | 不进任何公式、不进任何申报, 只留复现记录 |

**二、公式读不到, 但 `tools/compile_manifest.py --check` 读得到** —— 它们是对 C++ 源码的
断言, 改了对账失配 (退出码 1):

| 常数 | 对账的 C++ 项 | 公式侧实际走哪里 |
| --- | --- | --- |
| `TOTAL_UB_SIZE` | `LAYERED_USABLE_UB_BYTES` | — |
| `L1_TILE_K` | `L1_TILE_K` | `KernelConfig.l1_tile_k` (模块常数只是 `select_kl1` 的缺省实参, 调用点都显式覆盖) |
| `GMM2_LAG_MIN_TOKEN_NUM` | `GMM2_LAG_MIN_TOKEN_NUM` | `InstancePolicy.gmm2_lag_threshold` |

**三、有真读者**

`SCALE_TRANSFER_BYTES` 进 `select_kl1` 的容量判据 (`units * scale_a <= SCALE_TRANSFER_BYTES`
与同式的 `scale_b`)。改小它, 部分 tile 的 kL1 从 512 掉回 256 (`select_kl1(120, 2048)`),
GMM2 沿 K 的分段数随之翻倍 —— 一个分段就绪的形状上事件数 563 → 947, 墙钟不变
(段多了但依赖都已满足)。后果在事件图里, 不在 `total_us` 上。

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
from moe_cost_model import (MEGAMOE_A8W8 as P, StageLink,
                            simulate_routing_counts)
# 本函数**只收关键字参数**。routing_counts / costs 见「使用方式」的"场景文件: 完整示例"
simulate_routing_counts(routing_counts=C, costs=costs,
                        **P.shape_kw(), options=P.options)
# 以那份实现为底, 只改一条 stage 边 (GMM2 改成逐 kL1 块就绪)
simulate_routing_counts(routing_counts=C, costs=costs,
                        **P.shape_kw(), options=P.with_options(links=(
    StageLink("gmm1", "activation", location="onchip", depth=1,
              colocated_by_hardware=True),
    StageLink("activation", "gmm2", readiness="per_chunk"))))
```

缺省跑出来的数与那份实现的数不同, 这是信息 (差多少 = 那些编排选择值多少), 不是 bug。

### stage 边: 一条边三个问题

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

#### 哪条边能分段 (写得出就必须买账)

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

## 可调的旋钮与覆盖审计

### 可调的全部旋钮

(数据源: `knob_audit.EXPECTED` + `knob_audit.WHAT`), `tests/test_knob_coverage.py`
核对 README 里这份与生成结果逐字一致, 所以它不会和代码分叉。

判定的含义: **生效** = 五个形状上都动了模型; **生效\*** = 只在有作用对象的形状上动
(只有一波就谈不上超前几波); **被拒** = 模型显式拒绝该取值 (缺标定常数); **动不了** =
有旋钮但当前建模下没有可表达的后果。

<!-- BEGIN knob-table (generated: python tools/knob_audit.py --markdown) -->
| 旋钮 | 判定 | 管什么 |
| --- | --- | --- |
| `core_assignment` | 生效* | tile 分给哪个核 (三种策略) |
| `kernel.activation_n_half` | 生效 | SwiGLU 的投影数 (gate+up) |
| `kernel.combine_meta_bytes_per_row` | 生效 | COMBINE 每行搬几字节路由元数据 |
| `kernel.combine_quant_mode` | 生效 | COMBINE 的数据格式 (BF16 / FP8+scale) |
| `kernel.gmm1_b_reuse_frac` | 生效* | 非首个 m-group 的 tile 付几成 B 流 |
| `kernel.gmm1_interleaved` | 生效 | GMM1 的 gate/up 是否在 tile 内按列交织 |
| `kernel.l1_buf_num` | 生效 | L1 ping-pong 缓冲块数 (1 = 关) |
| `kernel.l1_size` | 生效* | L1 容量 (进 select_kl1 的容量判据) |
| `kernel.l1_tile_k` | 生效* | K 窗基线 |
| `kernel.swizzle_direction` | 生效* | tile 遍历的外层维 (0 = M 在外) |
| `kernel.swizzle_offset` | 生效* | swizzle 的分组宽度 |
| `kernel.tile_m` | 生效* | 一个 m-group 的行数 |
| `kernel.tile_n` | 生效 | 一个 N-tile 的列数 |
| `kernel.topk_weights_prefetch` | 生效 | topk 权重在 epilogue 里乘; 行块 256->128 且 GMM1 输出走 GM 往返 |
| `kernel.topo_urma` | 生效 | 通信路径: MTE 波循环 / URMA Layered 宏波循环 (换建图器) |
| `kernel.weight_nz` | 被拒 | 权重 GM 布局 Z / NZ (开启须显式给 NZ 带宽) |
| `options.barriers` | 生效 | 全核栅栏: 不加 / 波间 / 波内每 stage 后 |
| `options.combine_granularity` | 生效* | 一个 COMBINE 事件覆盖几个 GMM2 tile |
| `options.combine_layout` | 动不了 | COMBINE 写出的落点跨度 |
| `options.dispatch_pacing` | 生效* | dispatch 的发起配速 |
| `options.dispatch_partition` | 生效* | dispatch 的行按核预切还是不预切 |
| `options.dispatch_rows_per_item` | 生效* | 一份 dispatch 工作覆盖多少行 |
| `options.epilogue_overheads` | 生效 | 尾段五项固定开销 |
| `options.gmm2_kl1` | 生效 | GMM2 的 kL1 (不给则自适应) |
| `options.granularity` | 生效* | 每个 stage 一个事件覆盖多少个单元 |
| `options.late_bind_pools` | 生效* | 哪些引擎晚绑定 (派发时刻才定核) |
| `options.links` | 生效* | stage 边: 就绪粒度 / 落点 / 片上槽数 |
| `options.m_groups_per_wave` | 生效* | 波宽: 每波装几个 m-group |
| `options.pipeline` | 生效 | 相位拆分 (load/cube/fix) + 每核队列深度 |
| `options.roles` | 生效* | 哪个 stage 跑在哪个引擎角色上 |
| `options.serialize_dispatch_comm` | 生效* | 跨卡搬运是否串行化 |
| `policy.cursor_resonance_fix` | 生效* | 游标共振修正 |
| `policy.dispatch_lookahead` | 生效* | dispatch 超前几波 |
| `policy.gmm2_combine_credit` | 生效* | GMM2->COMBINE 的固定 credit |
| `policy.gmm2_lag_threshold` | 生效* | GMM2 滞后生效的 token 阈值 |
| `policy.gmm2_lag_waves` | 生效* | GMM2 滞后几波 |
| `policy.wave_offsets` | 生效* | 各 stage 的波偏移组合 |
| `scheduling_policy` | 生效* | 就绪集里谁先跑 (三种策略) |
| `wave_packing` | 生效* | 专家怎么组成波 (三种策略) |
<!-- END knob-table -->

```bash
python tools/knob_audit.py          # 五个形状逐旋钮扫一遍, 打印判定与差值
python tools/knob_audit.py --emit   # 重新生成 EXPECTED
```

**`topo_urma` 不是平级旋钮，是结构分叉。** 切换后波粒度从 256 行 m-group 变为专家范围，dispatch 从源推变为目的拉，combine 从配对 tile 变为批量 PUT。

| 旋钮                                                     | MTE 路径           | Layered 路径                              |
| -------------------------------------------------------- | ------------------ | ----------------------------------------- |
| `p1_override` / `p2_override`                        | 生效，决定每波组数 | 失效，Layered 按专家数和 token 数自定波数 |
| `wave_packing`                                         | 生效，三种策略     | 失效，Layered 有自己的波规划              |
| `dispatch_lookahead` / `wave_offsets`                | 生效，控制前瞻     | 失效，Layered 固定 recv 后紧跟 combine    |
| `gmm2_lag_waves`                                       | 生效，控制滞后     | 失效，Layered 的 GMM2 总与当前波同跑      |
| `StageLink("gmm1","activation").depth` | 生效 | 生效 |
| `gmm2_combine_credit`                                  | 生效               | 生效                                      |
| `core_assignment`                                      | 生效               | 生效                                      |
| `tile_m` / `tile_n` / `l1_tile_k` / `l1_buf_num` | 生效               | 生效                                      |
| `combine_quant_mode`                                   | 只有字节宽度生效¹  | 只有字节宽度生效¹                         |

¹ `combine_quant_mode` 只管**数据格式** (写侧每元素字节: BF16 2B → FP8 1B + 1/32 scale)。
"combine 跑在哪个角色、什么粒度"是**编排**, 分别由 `ModelOptions.roles` 与
`ModelOptions.combine_granularity` 给。参考实现把数据格式与这两件事绑在同一个模板参数上,
那是那份实现的耦合, 不是物理。

`KernelConfig` 的编译期旋钮（`l1_buf_num`、`l1_tile_k`、`combine_quant_mode`）以 `KernelConfig` 为唯一事实源。手工拼 `PrimitiveCosts` 时入口自动按 kernel 重绑公式，任何拼法都生效。

权重 (B 流) 是载入项里**更大**的那一股 (`b_load = wb·K·cols / bw_b`), 所以
`weight_nz` 与 `gmm1_b_reuse_frac` 都显著改时长。同一个 tile (m=256, K=6144, cols=256)
实测: 基线 90.917 µs; `weight_nz` 配 NZ 带宽 80000 得 **69.627 µs (−23%)**;
`gmm1_b_reuse_frac=0.53` 得 **62.430 µs (−31%)**。`gmm1_b_reuse_frac` 是比例, 不是布尔。

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

### 覆盖审计: 每个旋钮都必须能动模型

这个项目是给算子工程师改**编排 / 编译期 / 运行期**参数用的, 所以一个旋钮扫出
**0 收益**必须能分清是哪一种 0。四种意思, 指示完全相反:

| 判定 | 意思 | 下一步 |
| --- | --- | --- |
| 生效 | 每个形状上都动 (与形状无关) | 这个取舍可以照着做 |
| 生效* | 至少一个形状上动, 本形状没有作用对象 | 换形状再扫: 只有一波谈不上超前几波, 只有一个 K 块谈不上逐块就绪 |
| 被拒 | 模型显式拒绝该取值 (缺标定 / 这条路径没实现) | 拒绝是诚实的 |
| 动不了 | 模型里**没有可表达的后果** | **最需要分辨的一类**: 这个 0 是模型的空白, 不是硬件的事实 |

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

审计当前给出的几项判定:

* **引擎队列深度不是旋钮。** 持核事件独占 AIC/AIV0/AIV1, 同核在途数恒 ≤ 1, 所以每核
  引擎队列的容量固定为 1; 相位拆分后的载入相位刻意不继承 `Q:*`, 否则容量 1 的引擎信号量
  会限制 L1 缓冲深度。要表达"更深的队列"必须先有发射开销这类可观测的物理后果, 模型里
  没有。图里起不了约束的计数信号量由 `scheduler/normalize.py` 按一条定理删掉 —— 判据、
  八类 token 的分类表与实测见 `docs/design_space_gaps.md` 的「空约束」一节。
* **`topk_weights_prefetch` 的唯一出处是 `KernelConfig`。** 它是编译期宏
  `MEGAMOE_TOPK_PREFETCH`, 属于编译点而不是编排选项。它有后果: epilogue 行块 256→128、
  GMM1 的输出改走 GM 往返、每个行块多一次 topk 权重读 (见下节)。
* **`options.roles` 与 `options.epilogue_overheads` 在场景文件里写得出**: 文件里写
  `[options.roles]` 下 `combine = "AIV0"` 即可, 不必在 Python 里构造对象。

目前唯一标为"动不了"的是 `options.combine_layout`: 写侧只经 `scatter_us` 的
`(spread_slots/m) ** scatter_exponent`, 而 `scatter_exponent` 缺省 0 使指数项恒 1,
两种布局算出同一个数 —— 0 不是保守, 是实测把"落点跨度"这个机制否掉了; 读侧 UNPERMUTE
从顺序读变 gather 的代价完全没建模 (缺口 10)。

## 模型表达了哪些具体选择

### 事件粒度: 五个 stage 共有的一个维度

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
3. **combine 的攒批不按核分组**: combine 从 GM 读 GMM2 的输出, 同核不是物理约束。
   按核分组会让这个旋钮在轮转/晚绑定下静默失效。

**粗粒度不是收益开关, 符号随形状翻转**: 28 核确定性夹具 (40 个 GMM1 tile, 填不满核) 上
四种粗粒度全部变慢 (`combine=0` 从 245.97 慢到 596.47); 而 `scenario_basic.toml`
(4 卡 × 64 专家, tile 数远多于核数) 上 `gmm2=2` 从 1751.48 快到 1741.46。**必须扫, 不能
照搬取值。**

### 队列深度与执行单元数是两件事

`QueueDepths` 的五个字段是**在飞上限 / 缓冲槽数** —— 能提前多少发起, 由 L1/UB 槽数决定,
是编排选择, 所以它是旋钮。而**执行单元数**是硬件事实, 每核每种单元恒为 1:

| 队列深度 (旋钮) | 执行单元 (事实, 恒 1) | 怎么表达 |
| --- | --- | --- |
| `mte_aic` | 一个 AIC 一条 MTE2 (GM→L1) | `MTE2:c{core}` 容量 1 |
| `fix` | 一个 AIC 一条 FixPipe (L0C→UB/GM) | `FIXPIPE:c{core}` 容量 1 (见下) |
| `mte_aiv` | 一个 AIV 一条 MTE (GM↔UB) | `MTE_AIV:{eng}:c{core}` 容量 1 |
| `cube` | — | `.cb` 相位本身独占 AIC 核资源 |
| `vec` | — | ACT/COMBINE 主事件本身独占 AIV 核资源 |

深度恒为 1 时两者重合, 所以缺省下看不出区别。**深度 >1 时若只有队列深度、没有执行单元
约束, 等于给每个核多出几条管道**: 载入可无限并行, 墙钟会低于带宽下界。所以执行单元单独
写成**容量 1 的计数信号量**, 而不是独占资源 —— 后者不行是因为晚绑定只改写
`acquires`/`releases` 的核后缀 (`:c7` → `:c*`), 独占资源会把相位钉在建图时的占位核号上。

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
| ACT | 一对一跟随 GMM1 tile; `topk_weights_prefetch` 开着时按 128 行块切成两个 (见上文 EPILOGUE_TILE_M) | 同核的 AIV0 |
| GMM2 | 按输出列切 tile (`ceil(h / tile_n)` 个), 可换 `tile_grid`; 每个 tile 要行范围相交且覆盖整个 K 的 ACT | 每个 tile 一个 AIC 核, 游标轮转 |
| COMBINE | 一对一跟随 GMM2 tile | 同核的 AIV1 |

块号按全局 m-group 序对核数取模决定归属, 与波无关 —— 同一个块落在哪个波, 归属的核都不变。

**块数少于核数时, 多出来的核在 dispatch 阶段没有活。** 块数 = Σ 各专家 `ceil(行数 / tile_m)`。每卡专家少、每专家行数不足 `tile_m` 时 (小 batch) 这一项很小, dispatch 的并行度会成为瓶颈。

### 3. 当前改不了的结构

波粒度只有两种：MTE 路径按 256 行组切波，Layered 路径按专家范围切波。ACT 总与 GMM1 同波。同核程序序没有建成依赖边，靠资源互斥保序。带宽争用不建模（`channel_bytes` 只申报字节，不影响时长）。

要突破这些需要改 `builders/` 或 `scheduler/` 的结构，改完重新对实测校准。

## 四层架构: 实现 / 编译点 / 运行期 / 事件图

"这是哪份 kernel 的哪个编译点"是一个可查询的身份, 而不是散落在布尔开关与全局常数里。
四层各有自己的入口与校验: 换实现是换适配器, 标定常数按编译指纹分域, 编译参数与 C++
源码逐项对账。

```
Workload (token/专家/路由)
   -> Runtime  (拓扑 + 波推进策略)        implementations/runtime.py
   -> Compile  (编译轴 + 编译指纹)        implementations/compile.py
   -> Lowering (每份实现一个适配器)        implementations/megamoe.py
   -> DAG      (类型化事件图)              ir/
   -> Scheduler/Timing (与 kernel 无关)    scheduler/
```

### 实现身份: 换 kernel 是换适配器, 不是改一个布尔开关

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

### TopkWeightsPrefetch (`MEGAMOE_TOPK_PREFETCH`)

这个编译期模板参数在 kernel 里有三个**结构**后果, 模型逐条建了:

| kernel 的事实 | 出处 | 模型里的落点 |
|---|---|---|
| `EPILOGUE_TILE_M = TopkWeightsPrefetch ? 128 : 256` | `mega_moe_arch35.h:161` | `config.hardware.epilogue_tile_m`; ACT 按行块拆成两个事件 |
| AIC 落 GM + 置 `gmm1TileStatus`, AIV 等 GM 标志再 `CopyGM2UB` | `stage/mega_moe_gmm1_activation.h:618-645, 405-460` | `config.links.effective_gmm1_act_link`: 这条边 location="gm"、depth=0、同核不再是硬件强制 |
| 每个行块一次 topk 权重 GM→UB 读 (m × `META_INFO_SIZE` × int32) | 同文件 349/419 | `AnalyticalActCosts.readback_bytes`, 申报到 `act_readback` 通路 |

行块减半的原因是 UB 容量: prefetch 要多留一块 topk 权重缓冲
(`block_epilogue_activation_mx_quant.h:184` 的 `weightUb_`, 只在 prefetch 下分配)。

依赖键不变: 通知用的 flag 下标仍是 `subMLoc / L1_TILE_M_256` (m-group), 所以
`ctx.activation_ready` 的键还是 (专家, m-group), 同一个键下多一条行范围更窄的记录,
GMM2 按行相交把两个行块都取到。

**时长口径**: 读回与向量计算串行相加 (kernel 在 `CopyGM2UB` 之后紧跟
`SetFlag/WaitFlag<MTE2_V>` 才进 epilogue), 带宽取 `BW_LOCAL_GM`。**仓内没有 prefetch
路径的实测**, 所以这一项是按物理口径算的, 不是标定值。GMM1 侧的 Fixpipe 写出在两种落点下
都不进时长公式 —— 只申报字节, 不动时长。手工拼的 `PrimitiveCosts` 不描述读回时, 开 prefetch
会直接报错而不是按"读回免费"算。

模型给出的差值 (golden 的 `mte_topk_prefetch` vs 同形状的 `mte_3wave_lag2`):

| | 墙钟 | 事件数 | `hbm_write` | `act_readback` | AIC forced idle |
| --- | --- | --- | --- | --- | --- |
| 关 | 600.93 µs | 1252 | 25.4 MB | — | 1669.2 核·µs |
| 开 | 591.50 µs (−1.6%) | 1348 | 50.5 MB | 26.0 MB | 462.2 核·µs |

两种配置的 avoidable 空闲都是 0。提速来自 AIC 不再等配对 AIV 读走 UB; 翻倍的 HBM 写**在
时长上不计**, 因为带宽争用不建模。所以 −1.6% 是解耦收益的上界, 不是对实测的预测;
要收紧它需要整卡访存带宽与 prefetch 路径的实测, 两者仓内都没有。

`tests/test_topk_prefetch.py` 的 15 个测试钉住这条路。

### 编译清单: 与 C++ 源码对账

```bash
python tools/compile_manifest.py --check     # 失配则退出码 1
```

从 `mega_moe/include/CMakeLists.txt` 的 `MEGAMOE_*` cache 变量、两行 `#ifndef/#define` 宏缺省、
白名单 `constexpr` 常数、以及 `BlockSchedulerSwizzle<Offset, Direction>` 的模板实参抽出 23 项,
再与 Python 侧逐项对账 (20 项)。**只报告, 不改常数。**

为什么需要它: 注释与源码脱钩不会报错, 对账会。以 `BlockSchedulerSwizzle` 的模板实参为例,
`common/mega_moe_gmm_common.h:33` 写的是 `<3, 0>`, 若 `KernelConfig.swizzle_direction` 与它
不一致, m 组 > 1 时模型的 GMM tile 遍历顺序相对 kernel 是 M/N 转置的, 墙钟差 **+5.0%**。
测试里有一条把源码树复制出去只改那一个模板实参, 断言对账能抓到。

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
`TransferDirection`), 并且是**只读视图**: 不改 `Event`, 不改调度, 所以 40 个 golden 指纹
逐位不变。关键的区分是执行单元 (容量恒 1 的硬件事实) 与缓冲槽 (容量是编排选择) —— 两者用
同一个 `acquires/releases` 机制表达, 不分型就说不清"这个容量能不能调"。

**表达不了的东西写成明文** (`ir.UNREPRESENTABLE`, 有测试要求每条都讲清为什么):

| 缺口 | 现状 |
| --- | --- |
| 异步发射 vs 完成 | 只有一个 `duration_us`; 用拆相位近似重叠, 真的 issue 开销没有标定 |
| 硬件 flag 身份 | flag 只是某条边上的延迟; 没有身份, 没有 set/wait 配对; kernel 侧 20 多个 flag 与三个 `SyncLatency` 字段的对应关系无记载 |
| 带宽域争用 | 只有标签, 没有共享速率的后果 (带宽争用不建模) |
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

### 标定值按域登记: 一个数只在它量过的地方有效

```bash
python tools/calibration_domain.py --scenario examples/scenario_basic.toml
```

为什么要分域: 这些常数自己的注释就写明了它们在不同条件下不是一个数 ——
一套全局值覆盖所有实现/编译点/形状/拓扑是不成立的:

| 常数 | 它自己记下的离散 |
| --- | --- |
| `BW_L1_GM` 51900 | 按并发核数重拟: 28 核 45300 / 18 核 37000 (**1.40 倍**) |
| `BW_REMOTE_WRITE` 8600 | 三个形状各自反扣: 9.5 / 7.8 / 4.5 GB/s 每核 (**2.11 倍**) |
| `BW_UNPERMUTE_AGG` 950000 | h6144 比语料高 **18%** |
| `T_RANK_SYNC_RTT_US` 2.2 | 缺省形状 1.6-2.0, h6144 是 2.2-2.5 |
| `URMA_GET_LAT_US` 8.5 | 域是"4 卡 / 3 条流", 超出 world-1 > 3 未验证 |

`implementations/calibration.py` 给每个值配一个域 (实现 id + 编译指纹 + 形状域 + 拓扑),
查表给三种答案, 并且**三种要分开**:

* `in_domain` —— 实际运行落在量过的范围内;
* `out_of_domain` —— 哪几维越界、当时量的范围是什么、同一个量在别的条件下的其它观测;
* `undeclared` —— 这一维**从没声明过范围**。与越界是两种不同的不确定性: `BW_LOCAL_GM`
  的注释只说"单核大块 MTE 无竞争", 没给任何形状范围, 所以它在任何形状上都是 undeclared ——
  报 in_domain 会谎称量过, 报 out_of_domain 会谎称量过且超了。

换编译指纹报 `wrong_key`, 而不是拿另一个二进制上量的值顶上。

**不做自动外推**: 越域时不给"修正值"。那些依赖关系只有两三个点 (核数两个、形状三个),
凭它们造一条曲线再外推, 比直接说"超出标定域"更坏。

这一层立刻查出一件事: **项目自己的缺省场景 `scenario_basic.toml` (h=6144 / hidden_dim=4096 /
topk=8) 跑在全部带宽常数的标定域之外** —— 实测都是在 h=5120 / hidden=4608 / topk=6 上做的。
不是说结果没用, 而是读结论时要知道这些数的来源条件与它不同。

种子数据里的一个坑也是这层自己照出来的: 语料的编译指纹最初按模型缺省的
`combine_meta_bytes_per_row=16` 登记, 而打点跑的是 kernel (搬满 `META_INFO_SIZE` 8 个 int32
= 32B), 于是"复现那份实现"的场景查标定时全部报 `wrong_key`。现在语料按 32 登记, 与
`profiles.MEGAMOE_A8W8` 的指纹一致, 有测试钉住。

### 第三份实现: 声明了, 但会拒绝

`ascend950.megamoe.a8w4_wave.v1` 有身份、有源码依据, `accepts()` 会抛 `Unsupported` 并说清
差哪一步。为什么要有这样一个适配器: 使用者问"支持 A8W4 吗", 三种答案信息量完全不同 ——
没有这个名字 (像没想过)、有名字但凭空给个数 (最坏)、有名字且说清差什么 (可以照着补)。

从源码能确定的 (所以 DAG 的结构部分写得出来):

* 独立 kernel 类 `MegaMoeA8W4Wave`, 7 个模板参数 (没有 `IsGmm1Interleaved`);
* 多一段 **AIV 上的权重反量化前段**, A8W8 完全没有: `BlockPrologue` 只在 `IsA8W4` 时非 void,
  三步是 `CopyGmToUb` (4bit GM→UB) → `WeightAntiQuantComputeNzNk` (4bit→8bit 展开) →
  `CopyWeightToL1`, L1 双缓冲 384 KiB;
* B 矩阵分形与布局都不同 (`C0_SIZE_B = 32`, `LayoutB = Te::ZNLayoutPtn`);
* 角色分工不同 (AIV0 跑 prologue、AIV1 跑 combine), 而模型的角色表是全局的。

**差的是一个量, 不是一个旋钮**: `WeightAntiQuantComputeNzNk` 的向量吞吐。它的地位与 ACT 的
`ACT_BYTES_PER_VEC` / `BW_UB` 相同 —— 要实测。仓内没有 A8W4 的打点 (`data/` 下六个 run 的
`dtype` 都是 `fp8_e5m2`), 所以现在给不出。

这正是本项目的边界: 改同一个 variant 的参数可以自动出结果; 改了 C++ 控制流 / 同步协议 /
缓冲复用 / 流水阶段结构, 就必须重新生成实现描述并重新标定。

## 物理下界: 模型如何证伪自己

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

### 下界检查覆盖的三类错误

断言不通过时, 问题一定在模型这一侧, 而且只有三种成因:

1. **字节申报少于算法必搬的量** —— 物理上不可能, 说明某一股流进了时长公式却没进申报;
2. **墙钟低于带宽下界** —— 说明某条搬运没有占住它该占的执行单元 (典型是把 L1 缓冲槽数
   当成了 MTE2 管道: 前者是能提前多少发起, 后者是同时能搬几笔, 每核恒 1 条);
3. **字节随编排旋钮变化** —— 说明某处从时长倒推字节, 而不是按算法逐项申报。

第三类由单独的不变量钉住: 字节由建图器按算法申报, 相位展开只做重新分配。

### 先读下界, 再读旋钮

`scenario_basic` 上算力下界只有 25.57 µs, 带宽下界 1665.37 µs, 模型给 1751.48 µs ——
**95% 的时间在搬字节**, 编排再怎么调最多碰到剩下的 5.17%。要降那 1665 µs 只能**少搬**:
换 dtype、提高 L2 复用、改物化编排。

下界这个判断只用算法 + 物理 + 硬件事实, 不依赖任何未标定系数; 旋钮对比那部分依赖标定,
见「结论的区间」与 `examples/run_uncertainty.py`。

## 核空闲分解: 哪些消不掉, 哪些是某个编排选择的代价

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

### `avoidable` 不等于"实现做错了"

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
import moe_cost_model as m

res = m.simulate_routing_counts(
    routing_counts=C, costs=costs, token_num_per_rank=..., h=..., hidden_dim=...,
    aic_num=...,                      # 以上四项是必填关键字参数
    scheduling_policy=m.WorkConservingCriticalPath(),
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

晚绑定与相位流水**可以同用**: 同一个 tile 的几个相位编成**核组**,
核号由组里最先派发的那个事件选定, 同组其余事件跟随 —— 相位事件不持核资源, 所以不能靠
`colocate_with` (它要求锚点先绑定, 而先跑的恰恰是不持核的那一相)。实测 9216/3: 静态+拆相位
220.83 us, 晚绑定+不拆 196.92 us, 两者叠起来 **183.62 us**, 不变量仍然全 0。

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

整体墙钟的校准还缺两个系数: 实测的 Cube 速率; 以及按并发核数分档的 `BW_L1_GM`
(A 流斜率显示它随并发变: 28 核 45.3 / 18 核 37.0 GB/s)。所以下面这组误差是**在现有系数
之下**的量级, 不是定论。

`profiles.MEGAMOE_A8W8` 这组取值 (即那份实现) 在 B≤128 标定域内各 stage busy 误差 ±5%,
墙钟偏差 −6~−8%。GMM1 逐 tile 的搬运误差是 +0.4%~+3.5%。**缺省值不是这组取值** ——
缺省是"最少假设", 与实测对齐要显式引用 profile (场景文件里写 `profile = "megamoe-a8w8"`)。

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
| `ACT_BYTES_PER_VEC` | 722 B | 源码逐项计数 (读 520 + 写 202); 两个不同 m 的 run 定出的斜率 0.007760 µs/向量 与 722/`BW_UB` 相差 +0.05% |

搬运带宽全部是单点标定，没有一个扫过并发数；Cube 计算速率完全没有精确值；Vector 的 ACT 字节口径源码计数与实测不一致。域内 B≤128 可用，域外或参数变了需重新标定。

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

### 带宽上界校验: 单核常数不能突破聚合上界

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

## 带宽争用: 不建模

`Event.channel_bytes` 由建图器按算法无条件申报, 但**不参与准入、不影响任何时长**; 按通路
汇总在 `rank_results["traffic_bytes"]` 里做访存量核算。模型里没有"事件之间抢带宽"这件事。

为什么不建: 现有常数不在同一个尺度上, 建出来的争用不可信。片间通路最清楚 —— 聚合
`BW_WINDOW = 33000` B/µs 是**整卡**带宽, 而逐事件速率 `BW_REMOTE_GM = 31000` 是从 28 核
并发的真实运行反解的**单核**值 (已含平均争用)。用前者当容量、后者当速率, 单个事件就吃掉
整卡 94%, 并发再叠 26 倍降速, 争用被计两遍。片内四条相反: 每核速率 × 核数按构造恰好无
争用, 开与不开逐位相同, 不收紧就没有信息, 收紧又没有整卡带宽的实测可依。

**不建模的后果写在明处**: 带宽绑定形状上的墙钟是下界方向偏乐观的 (见「物理下界」),
而字节申报是齐的 —— 所以"少搬多少字节"这类结论可用, "争用让它慢多少"这类结论给不出。
要建争用模型, 需要先有该层级的聚合带宽实测 (单核独占 + 多核并发两组)。

## tiling 真值与实测工件

`examples/*.toml` 的 `[tiling] path` 指向实测 run 的 `raw/tiling_rank0.bin` —— 这是
`tests/test_guardrails.py` 与 `tools/eval_suite.py` 核对"场景文件声称的形状 == 跑出数据的
kernel 配置"的唯一依据, 也是本项目防"手抄参数没人核对"的那一层。

打点工件体积大, `.gitignore` 把 `/data/*/raw/` 整个排除了, 所以**干净克隆里 tiling 真值
缺席**: 5 条校验测试 skip, `eval_suite` 一个场景都跑不了。tiling 真值本身只是十来个整数,
用导出器写成几百字节的 JSON 旁置文件入库即可永久解决:

```bash
python tools/export_tiling.py --all        # 在有 raw/*.bin 的采集机上跑一次
git add data/*/tiling_rank0.json           # 不在 gitignore 里
```

`parse_tiling` 在 `raw/*.bin` 缺失时自动回落到上一级同名 `.json`, 所以 `examples/*.toml`
一字不用改。两者都没有时报错会指明这条命令。

## 回归保护

`tests/test_golden.py` 对 40 个配置核对指纹, 锁**四类**东西:

| 锁什么 | 字段 | 为什么 |
| --- | --- | --- |
| 时长与排程 | `total_us` / `dag_end_us` / `stage_busy_us` / `schedule_sha256` (每个事件的起止、等待归因、关键父事件) | 任何一位浮点差异都会失败 |
| **访存量** | `traffic_bytes` (逐通路) | 字节口径的改动不改时长, 只锁时长抓不到 |
| **下界** | `bounds` (三个下界 + `binding` + `violation`) | 穿透物理下界要立刻显形, 不等全套 |
| **出处** | `provenance_summary` (类别计数) + `provenance_sha256` (全部常数的名字/取值/**完整标签文本**) | 标签文本是承诺 (域限制、待重标), 和数值一样该被保护 |

后三类的鉴别力用注入法验证过: 去掉 COMBINE 读回的字节申报报 **2/40 DIFF**, 删掉某个
常数出处里的"域受限"三个字报 **34/40 DIFF** —— 两者都不改时长, 只锁时长的指纹看不见。
两层出处都要: 类别计数只在**分类**变了时变, 同类别内的文本改动靠 sha256 抓。
全跑 35 秒, 覆盖 MTE / Layered 两条路径、波偏移、编译期旋钮、三种调度策略、分核与打包策略、相位流水、计数信号量、任务转移。

```bash
python tools/gen_golden.py --check     # 核对, 不写文件
python tools/gen_golden.py             # 重新生成快照
```

重构与提速必须通过 `--check`。只有在有意改变模型行为时才重新生成快照, 并在提交说明里写明哪些用例变了、为什么。

## 仿真耗时

本容器单核实测, 事件数是**每 rank**:

| 配置 | 事件数/rank | 耗时 | 墙钟 |
| --- | --- | --- | --- |
| `examples/run_basic.py` —— MTE, 4 rank × 64 专家, B=64, **缺省选项 (晚绑定)** | 4423 | 230.7 s | 1737.51 µs |
| 同上, 只改 `ModelOptions(late_bind_pools=())` (静态分核) | 4423 | **2.71 s** | 1754.94 µs |
| `examples/scenario_basic.toml` (profile `megamoe-a8w8`, 其 `late_bind_pools=()`) | 6599 | 3.8 s | 1751.48 µs |
| 同形状改 Layered (`topo_urma=True`), 缺省选项 | 3210 | 286.0 s | 2181.65 µs |

**晚绑定是主要开销, 同形状 85 倍**: 池化资源下调度器要为每个事件在池里挑核, 静态分核在
建图时就定了。缺省是晚绑定 (`late_bind_pools=("AIC", "AIV1")`), 理由是它把 avoidable 空闲
清零 (见「核空闲分解」), 代价就是这 85 倍。要快就显式给 `late_bind_pools=()`, 但要连带
接受那部分空闲: 同一形状静态分核的 avoidable 是 AIC 1043.6 / AIV0 0.0 / AIV1 685.3 核·µs,
而晚绑定下三个池都是 0。

绝对耗时随容器负载浮动, **比值才是可比的量**; 墙钟那一列与负载无关, 可以用来核对这张表
是不是还对得上代码。

分段 (`run_basic` 那一行, 4 rank 合计 17692 事件): `build_events` 0.68 s,
`idle_decomposition` (1 rank, AIC 池) 0.09 s, 其余 **229 s 全在调度与后处理**。
所以慢的是调度本身, 不是建图, 也不是空闲分解。无重构钩子、使用内置调度策略时各 rank
独立调度 (结果与合并调度逐位一致, 见 `model._ranks_independent`)。

`idle_core_stealing` 是另一笔: 它每次提交都扫描全部未提交事件。golden 里
`stealing_gmm1` (1674 事件) 2.5 s、`stealing_pipeline` (2826 事件) 6.6 s, 而同规模
不带转移的用例 (`core_greedy_least_busy`, 2308 事件) 是 0.2 s。

## 项目结构

`tests/test_readme_structure.py` 核对这棵树与 `src/moe_cost_model/` 下的实际文件一一对应。

```
moe-cost-model/
├── pyproject.toml
├── src/moe_cost_model/
│   ├── __init__.py              # 显式导出
│   ├── scenario.py              # 统一入口: Scenario / load_scenario / simulate
│   ├── guardrails.py            # 校验: tiling 真值核对 / 信道尺度 / 路由守恒
│   ├── registry.py              # 策略名注册表
│   ├── profiles.py              # 复现某份实现用的成套取值 (MEGAMOE_A8W8 等)
│   ├── api.py                   # simulate_routing_counts 底层入口
│   ├── config/                  # 第 0 层: 纯参数, 不含逻辑
│   │   ├── hardware.py          #   硬件常数 + KernelConfig (编译点) + select_kl1
│   │   ├── platform.py          #   白皮书规格 (spec): 峰值算力/带宽, 与实测分开记
│   │   ├── policy.py            #   InstancePolicy + StageWaveOffsets
│   │   ├── pipeline.py          #   PipelineConstraints + QueueDepths + tiling 解析
│   │   ├── links.py             #   StageLink: 一条 stage 边的就绪/落点/槽数 + EDGE_AXES
│   │   ├── readiness.py         #   Readiness: 消费者沿共享轴分几段就绪 (段界/余数规则)
│   │   ├── roles.py             #   RoleAssignment: 哪个 stage 跑在哪个引擎角色
│   │   ├── granularity.py       #   StageGranularity: 一个事件覆盖多少个单元
│   │   └── provenance.py        #   常数出处标签系统 (SourcedValue / SourcedInt)
│   ├── shape.py                 # 第 1 层: MegaMoeShape / ModelOptions
│   ├── costs.py                 # 第 1 层: 各 stage 物理公式
│   ├── implementations/         # 第 1.5 层: "这是哪份 kernel 的哪个编译点"
│   │   ├── identity.py          #   ImplementationId = hardware.implementation.variant
│   │   ├── compile.py           #   CompileConfig: 17 个编译轴 + 指纹
│   │   ├── runtime.py           #   RuntimeTopology: 几卡几核
│   │   ├── adapter.py           #   适配器接口 (plan / lower / accepts) + Unsupported
│   │   ├── megamoe.py           #   三份实现: a8w8_wave / layered / a8w4 (已声明未建图)
│   │   ├── manifest.py          #   编译清单: 从 C++/CMake 抽参数并与 Python 对账
│   │   └── calibration.py       #   标定域: 一个数只在它量过的 (实现, 编译点, 形状, 拓扑) 里有效
│   ├── scheduler/               # 第 2 层: 通用离散事件调度引擎
│   │   ├── events.py            #   Event / Channel / 速率服务器
│   │   ├── engine.py            #   MultiResourceScheduler
│   │   ├── normalize.py         #   删掉起不了约束的计数信号量 (判据是一条定理)
│   │   └── policies.py          #   EarliestStart / WorkConservingCriticalPath / PriorityByStage
│   ├── planning/                # 第 3 层: wave 规划 + tile 网格
│   │   ├── waves.py             #   plan_waves / swizzle / Layered 波规划
│   │   ├── core_assignment.py   #   StaticRoundRobin / GreedyLeastBusy / ContiguousBlock
│   │   ├── wave_packing.py      #   SequentialGreedy / LongestExpertFirst / BalancedWaves
│   │   └── tile_grid.py         #   TileGrid: 行范围 x 列范围, 可自定义切分
│   ├── builders/                # 第 4 层: 事件图构建
│   │   ├── base.py              #   公共基类 (_event / 共享专家 / 尾段)
│   │   ├── context.py           #   BuildContext: 各 stage 之间的共享状态
│   │   ├── gmm1.py              #   GMM1 tile (ACT 由 activation.ActBatcher 一起发)
│   │   ├── activation.py        #   ACT tile: 攒批 + epilogue 行块拆分
│   │   ├── gmm2.py              #   GMM2 head/tail + K 段就绪
│   │   ├── tiling.py            #   tile 合并 (事件粒度) 与标签
│   │   ├── barriers.py          #   全核栅栏 (融合 vs 分段)
│   │   ├── comm/                #   通信协议接口 (dispatch 与 combine 都在这里)
│   │   │   ├── base.py          #     DispatchTransport / CombineTransport
│   │   │   ├── mte.py           #     MTE: DataCopyPad 直写 + 配对 tile combine
│   │   │   └── urma.py          #     URMA: 批量 GET/PUT + AIV1 程序序链
│   │   ├── mte.py               #   MTE 编排
│   │   ├── layered.py           #   Layered 编排
│   │   └── pipeline_expand.py   #   相位拆分 + 信道/容量
│   ├── ir/                      # 事件图的类型化只读视图
│   │   ├── vocabulary.py        #   Engine / Pipe / MemorySpace / TokenKind / ...
│   │   └── graph.py             #   classify_resource / classify_token / 图视图
│   ├── validation/              # 校验与实测对账
│   │   ├── invariants.py        #   7 条结构规则 (建图器必须满足的)
│   │   ├── trace.py             #   Chrome Trace 读取 (容忍截断)
│   │   └── compare.py           #   预测 DAG vs 实测 trace 的结构比对
│   ├── model.py                 # 第 5 层: A8W8WaveCostModel 编排
│   └── analysis/                # 第 6 层: 解读
│       ├── critical_path.py     #   关键路径与等待归因
│       ├── idle.py              #   核空闲分解 (forced vs avoidable)
│       ├── stealing.py          #   空闲核任务转移
│       ├── bounds.py            #   三个物理下界 (模型怎么证伪自己)
│       ├── design_space.py      #   一次扫一组编排选择
│       └── sensitivity.py       #   标定值不确定度 -> 结论区间
├── tests/                       # 42 个文件 / 547 项 (引擎 / 建图 / 实现层 / IR / 校验 / golden / 场景)
├── examples/                    # 场景文件 + 六个实测 run 的复现脚本 + 设计空间扫描
└── tools/                       # 16 个脚本: 清单对账 / 标定域 / 旋钮审计 / trace 比对 / golden / 报告
```
