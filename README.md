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
一条边一行、一个旋钮一列的批量扫描见「使用方式」的"扫一遍"; 逐个编排维度的语义与实测
收益见 `docs/modelling.md`。

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

```bash
cd moe-cost-model
pip install -e .                   # 或直接 pytest (pyproject 已配 pythonpath)
pytest tests/                      # 547 项, 约 25 分钟 (5 项需 tiling 真值旁置文件)
python examples/run_scenario.py    # 场景文件 + 改旋钮对比 (日常入口)
```

本文件讲**原理与用法**。细节分到 `docs/`:

| 文档 | 内容 |
| --- | --- |
| [`docs/USAGE.md`](docs/USAGE.md) | 怎么用: 场景文件怎么写、旋钮速查、完整示例 |
| [`docs/knobs.md`](docs/knobs.md) | 全部可调参数的清单与判定 (生效 / 需对的形状 / 被拒 / 未建模) |
| [`docs/modelling.md`](docs/modelling.md) | 逐个编排维度的语义、物理耦合、实测收益 |
| [`docs/architecture.md`](docs/architecture.md) | 四层架构: 实现身份、编译指纹、编译清单对账、类型化事件图 |
| [`docs/idle_decomposition.md`](docs/idle_decomposition.md) | `forced` / `avoidable` 的定义与已声明上界 |
| [`docs/provenance.md`](docs/provenance.md) | 常数的出处分类、规格与实测分开记、模块常数改了会发生什么 |
| [`docs/design_space_gaps.md`](docs/design_space_gaps.md) | 已知缺口与已定选择, 逐条带实测 |
| [`docs/calibration_runs.md`](docs/calibration_runs.md) | 要哪些上板 run 才能把系数定下来 (R1–R8) |
| [`docs/why_aic_idles.md`](docs/why_aic_idles.md) | AIC 为什么空闲: 七个成因与消除手段 |

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
# 本函数**只收关键字参数**。routing_counts / costs 见 docs/USAGE.md
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
都是同一组问题: 消费者沿共享轴分几段独立就绪 (`readiness`)、中间结果放哪 (`location`)、
片上能同时存几块 (`depth`), 加上每多一段付多少同步开销 (`segment_sync_us`) 与同核是否为
硬件强制 (`colocated_by_hardware`)。

每条边的共享轴、自然块、以及"分段在这条边上有没有意义"逐条写在 `config/links.EDGE_AXES`,
校验照着它**拒绝**写不出后果的取值, 而不是静默忽略。字段取值表、四条边的共享轴表、以及
`readiness` 与 `granularity` 的分工见 [`docs/modelling.md`](docs/modelling.md)。

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
清零 (定义与上界见 `docs/idle_decomposition.md`), 代价就是这 85 倍。要快就显式给 `late_bind_pools=()`, 但要连带
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
