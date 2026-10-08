# MegaMoE Cost Model

**在 CPU 上模拟 NPU 执行的 cost model。** 给它一组流水编排与编译/运行期参数, 它按昇腾的
硬件条件算出这套参数在 NPU 上会怎么跑: 执行时间、每个事件什么时候开始、在等什么、搬了
多少字节、哪些核在空转。

用途是让算子工程师**自己把参数调出来** —— 不上板、不等编译、不排队占机器。一次运行 0.2
到 4 秒, 所以可以一次扫几十组参数。它不是用来解释某一次已有运行为什么慢的, 那是
profiler 的事。

## 它模拟了哪些硬件条件

这一节决定"在模拟器上调出来的参数, 拿到 NPU 上还成不成立"。

| 硬件条件 | 怎么模拟的 |
| --- | --- |
| 每核的执行单元 | 一个 AI Core 一条 MTE2 (GM→L1)、一条 Cube、一条 FixPipe; 一个 AIV 一条 MTE、一条 Vector。各为容量 1, 所以同核上这些单元不会被凭空并行 |
| 片上容量 | L1 / UB / L0C 容量与缓冲槽数。GMM1 到激活的 UB 槽按"取了才能用、还了才空出来"记账: 槽没还回来, 新的 GMM1 落不进去, 即使核是空的 |
| 搬运带宽 | 七条通路分开计时, 取值见下表 |
| 跨卡通信 | dispatch 的窗口排空、combine 的跨卡写, 按字节与带宽计时; 通路可选串行化 |
| 同步 | stage 之间三条依赖边上可填同步延迟; 片上缓冲的取与还跨事件持有 |
| 工作到核的分配 | 编译期静态分核 (复现某实现的 `startBlockIdx` 轮转) 与派发时刻动态取活, 两种都能模拟 |
| 波推进 | 分波、波偏移、波间配速、分段栅栏 |

搬运带宽的当前取值 (单位 GB/s, 代码里是 B/µs):

| 通路 | 常数 | 取值 | 怎么定的 |
| --- | --- | ---: | --- |
| GM→L1, GMM 载入 | `BW_L1_GM` | 51.9 | B=64 的 H 扫描差分, 单点 |
| GM→UB, 激活 | `BW_UB` | 93 | 大 m tile 单点; 读写端口速率不同, 这是混合折算 |
| UB→GM, combine 散射写 | `BW_SCATTER` | 139.5 | B=64 随机路由反推 |
| GM→L1, dispatch 本地段 | `BW_LOCAL_GM` | 157 | 大尺寸拟合, 单核无干扰 |
| GM→L1, dispatch 远端段 | `BW_REMOTE_GM` | 31 | 1→2 行段差分 |
| 跨卡, dispatch 远端 | `BW_WINDOW` | 33 | 窗口排空差分 |
| URMA GET / PUT | `URMA_GET_BW_SINGLE` | 2.25 | 两点拟合; PUT 取 GET 的对称值 |
| GM, unpermute 尾段 | `BW_UNPERMUTE_AGG` | 950 | 双尺度 |

向量侧: `VEC_REG_WIDTH` 256 bit (来自 kernel 定义), `T_STARTUP_VEC` 1.48 µs (小 m 截距),
`ACT_BYTES_PER_VEC` 722 B (源码逐项计数, 读 520 + 写 202)。

**Cube 速率缺省是 0, 后果是计算项整个不参与计时** —— 仓里没有标定过的值, 要算计算时间
必须自己给 `cube_mac_per_us`。详见「还没模拟的硬件条件」。

## 它能让你调什么

全部可调项与判定在 [`docs/knobs.md`](docs/knobs.md)。几个实际的例子, 数字都是模拟器跑出来的:

| 想试的事 | 怎么写 | 模拟结果 |
| --- | --- | --- |
| GMM1 的片上缓冲单槽改双槽 | `StageLink("gmm1","activation", depth=2)` | 326.64 → 296.34 µs (−9.3%): GMM1 少等 75.8, GMM2 多花 45.5 |
| GMM2 不等齐整个 K 就开工 | `StageLink("activation","gmm2", readiness="per_chunk")` | 326.64 → 315.87 µs (−3.3%); 换成 hidden=18432 逐块反而比 3 段差, 事件数 240→4320 后排队成了新瓶颈 |
| 激活结果不落 GM、留片上 | `location="onchip"` | 少搬 70.78 MB, 但 2017 µs (+517%): 消费者被迫与生产者同核, 一组工作固定在一个核上 |
| 工作到核改成运行时抢活 | `late_bind_pools=("AIC","AIV1")` | 1754.94 → 1737.51 µs; 静态分核下有 1729 核·µs 的工作已就绪却没核接, 动态取活后归 0 |
| 权重走 NZ 格式 / B 流复用 | `weight_nz` / `gmm1_b_reuse_frac=0.53` | 同一个 tile 90.917 → 69.627 (−23%) / 62.430 µs (−31%) |
| 开 topk 权重预取 | `topk_weights_prefetch=True` | 591.50 vs 600.93 µs; AIC 的被迫空闲 1669 → 462 核·µs, 代价是 HBM 写 25.4 → 50.5 MB |
| 相位流水队列深度 | `queues.mte_aic=2` | 该形状 0.00%: 它是带宽绑定的, 把载入与计算重叠不会让载入变快 |

## 怎么跑

```bash
pip install -e .                      # 或直接 pytest (pyproject 已配 pythonpath)
python examples/run_scenario.py       # 场景文件 + 改参数对比 (日常入口)
python examples/run_design_space.py   # 一次扫一组编排, 每个方案一行
python examples/run_pipeline_study.py # 同上, 按"单核内 / stage 之间 / 波之间 / 工作落核"分组
python examples/run_uncertainty.py    # 每条结论标成稳定或不可判定
python examples/run_basic.py          # 底层 API 的最小例子
pytest tests/                         # 547 项, 约 25 分钟 (5 项需 tiling 真值旁置文件)
```

场景文件的表名与字段名就是对象属性名, 写错字段会立即报错并给出提示
(`policy.gmm2_lag_wave: 未知字段, 是否想写 'gmm2_lag_waves'?`):

```toml
profile = "megamoe-a8w8"     # 以某份实现的取值为底; 不写 = 最少假设
h = 6144
hidden_dim = 4096
aic_num = 28

[workload]
tokens = 64
topk = 8
world = 4
local_experts = 64

[options.granularity]
gmm2 = 2                     # 一个 GMM2 事件覆盖 2 个相邻 n-tile

[[options.links]]
producer = "activation"
consumer = "gmm2"
readiness = "per_chunk"      # whole (缺省) / N>=2 均分 / per_chunk / first_chunk
```

细节分到 `docs/`:

| 文档 | 内容 |
| --- | --- |
| [`docs/USAGE.md`](docs/USAGE.md) | 场景文件怎么写、参数速查、完整示例 |
| [`docs/knobs.md`](docs/knobs.md) | 全部可调参数的清单与判定 (生效 / 需对的形状 / 被拒 / 未建模) |
| [`docs/modelling.md`](docs/modelling.md) | 逐个编排维度的语义、物理耦合、实测收益 |
| [`docs/architecture.md`](docs/architecture.md) | 实现身份、编译指纹、与 C++ 源码对账、类型化事件图 |
| [`docs/idle_decomposition.md`](docs/idle_decomposition.md) | 核空闲怎么分解, 这个量的已声明上界 |
| [`docs/provenance.md`](docs/provenance.md) | 常数出处分类、规格与实测分开记 |
| [`docs/design_space_gaps.md`](docs/design_space_gaps.md) | 已知缺口与已定选择, 逐条带实测 |
| [`docs/calibration_runs.md`](docs/calibration_runs.md) | 要哪些上板 run 才能把系数定下来 (R1–R8) |
| [`docs/why_aic_idles.md`](docs/why_aic_idles.md) | AIC 空闲的七个成因与消除手段 |

## 工作原理

一次执行展开成事件图, 每个事件带五样东西:

```
resources        本事件独占的资源 (某个核的某个角色), 占用区间就是事件自身那一段
deps             前置事件 + 每条边上的同步延迟
acquires/releases 计数信号量: 可跨事件持有, 用来表达缓冲槽
colocate_with    必须与哪个具名事件落同一个核 (如 Fixpipe 直给配对 AIV0)
duration_us      时长, 由闭式物理公式给出
```

这五样就是模拟器的表达能力上界: 表达不出来的约束, 它不声称。

排程是贪心表调度。每个就绪事件先取 `max(前置都结束了, 我要的资源空出来了)`, 再沿信号量
未来的归还时刻找最早可准入点, 从就绪集里取最早能开始的提交; 墙钟是最后一个事件的结束
时刻。事件时长由公式给 (GMM tile = `max(载入, 计算)`), **与并发无关** —— 并发、排队、
空闲全是排程的产物。

资源名里允许写池占位符, 这时具体哪个核在**提交那一刻**才定, 取"核空出来"与"该核的槽可用"
两者取最小的那个核。这就是动态取活的模拟方式。

每次运行同时做两项检查:

- **物理下界**: 算力、带宽、最长依赖链三条取最大, 墙钟低于它直接抛 `BoundViolation`。
  `examples/scenario_basic.toml` 上三条分别是 25.57 / 1665.37 / 68.06 µs, 带宽绑定,
  模拟结果 1751.48 µs。
- **工作守恒**: 核空闲分成"没有就绪的活"与"有就绪的活却有核空着"。后者在动态取活下要求
  恒为 0; 不为 0 说明这个方案的时长偏慢, 该方案的收益不能与别的方案直接比。

## 输出解读

每次仿真返回以下字段:

| 字段 | 含义 |
| --- | --- |
| `kernel_total_us` | 最慢 rank 的执行时间, 记到最后一个 combine 结束 |
| `kernel_dag_end_us` | 含尾段的结束时刻 (与实测整段墙钟对比时用) |
| `slowest_rank` / `scenario` | 最慢 rank 的号; 本次运行的场景对象 |
| `provenance` | 全部常数的出处报告 |
| `dispatch_ready_tiles` | 每个 (波, 专家, m-group) 的就绪时刻与依赖 |
| `rank_results[r]["total_us"]` / `dag_end_us` | 该 rank 的执行时间 / 含尾段结束时刻 |
| `rank_results[r]["events"]` | 全部事件: 名字、起止、落核、三类等待、关键父事件 |
| `rank_results[r]["critical_path"]` | 关键路径事件链, 终点是最后一个 combine |
| `rank_results[r]["bounds"]` | 三条下界、哪条绑定、`violation` |
| `rank_results[r]["idle_decomposition"]` | 核空闲分解, 并逐段给出当时哪些核在空、哪些就绪事件在等 |
| `rank_results[r]["traffic_bytes"]` | 逐通路访存量 |
| `rank_results[r]["stage_busy_us"]` | 各 stage 忙碌时长 |
| `rank_results[r]["stage_dependency_wait_us"]` / `stage_resource_queue_us` | 各 stage 等数据 / 等引擎的时长 |
| `rank_results[r]["stage_first_start_us"]` / `stage_last_end_us` | 各 stage 的首个开始 / 最后结束时刻 |
| `rank_results[r]["resource_utilization"]` | 每核利用率 |
| `rank_results[r]["resource_busy_us"]` / `resource_idle_us` / `resource_span_us` | 逐资源 (如 `R0.AIC:7`) 的忙 / 闲 / 跨度 |
| `rank_results[r]["waves"]` / `wave_count` / `m_groups_per_wave` | 波计划: 逐波范围、波数、每波组数 |
| `rank_results[r]["cursor_trace"]` | 游标推进轨迹 |
| `rank_results[r]["gmm2_lag_active"]` | 本次运行 GMM2 滞后有没有真的生效 |
| `rank_results[r]["implementation"]` | 这条结果的身份: 实现 id / 编译指纹 / 运行拓扑 / 计时终点 |

执行时间不含尾段 (counts_export / core_sync / rank_sync / buffer_init / unpermute /
finalize)。尾段事件仍在图里照常排程, 只是不计入。

通路名有语义, 不能混用: `gm_to_l1` 是 GMM 的 A 流 + B 流, `hbm_write` 是激活的量化输出
与本卡 combine 行, `combine_read` 是 combine 读回 GMM2 tile 与路由元数据, `act_readback`
只在开了 topk 预取时存在, `dispatch_read` / `dispatch_write` 是 dispatch 的本卡读写,
`fab_src` / `fab_dst` 是片间。字节由建图器按算法逐项申报, **绝不从时长倒推**。

对比两个方案时: 先看 `kernel_total_us` 差值, 再看 `stage_busy_us` 哪个 stage 变了, 最后看
`critical_path` 上卡在哪种等待。

## 常数的出处: 这个数是谁定的

每个常数带一个标签, 回答"换一份 kernel 它会不会变":

| 标签 | 是什么 | 换实现会变吗 |
| --- | --- | --- |
| `spec:` | 硬件规格 / 格式标准 | 不会 |
| `algo:` | 算法定义 | 不会 (换算法才变) |
| `impl:` | **某一份实现的选择** (tile 几何、缓冲槽数、档位阈值) | **会** |
| `measured:` | 实测, 含争用与开销 | 看标定域 |
| `assumed:` | 假设值 | 报告里高亮 |

`impl:` 类的数必须能被参数覆盖; 模型的缺省值不引用任何实现。标签文本由
`provenance_sha256` 锁住 —— 改掉一句"域受限"也会让回归失败。

没有标定到点的输入单列在 `analysis.sensitivity.UNCERTAIN_INPUTS`: 有实测区间的参与区间
传播; 连区间都没有的 (动态取活开销、Cube 可达效率、分段同步开销) 使依赖它的结论判定为
**不可判定**, 而不是给一个数。换编译点时标定域报超域, 不沿用旧系数外推。

```bash
python tools/calibration_domain.py --scenario examples/scenario_basic.toml
python tools/compile_manifest.py --check     # 编译参数与 C++ 源码对账, 失配退出码 1
```

## 保真度

| 手段 | 当前结果 |
| --- | --- |
| 与 profiler trace 的结构比对 | GMM1 54/54、激活 54/54、GMM2 60/60、combine 60/60, 条数比 1.00 且逐专家一致; 2/2 波、28/28 核 |
| 编译参数与 C++ 源码对账 | 从 CMake、宏、`constexpr`、模板实参抽 23 项, 与模型侧对账 20 项 |
| 时长误差 (标定域内 B≤128) | 逐 stage 忙碌 ±5%, 墙钟 −6% 至 −8% |
| 标定域外已知失效 | B=1024 时 combine 偏差 +114% 至 +246%, GMM1 高估 +4% 至 +10% |
| 回归 | 40 个配置的调度指纹逐位锁定 (时长与排程、访存量、三条下界、常数出处), `python tools/gen_golden.py --check` 35 秒 |
| 参数覆盖 | 每个可调参数在五个形状上分成生效 / 需对的形状 / 被拒 / 未建模, `python tools/knob_audit.py` |

**一条方法限制**: 贪心表调度对输入不单调 —— 实测在一条依赖边上加 0.01 µs 让墙钟变化
−2.81%; 就绪粒度均分 3 段比 2 段差 1.0% 而 4 段又回到 2 段的值。所以**几个百分点以下的
差值不能当有效差异读**, 必须同一绑定方式、同一调度策略, 且差值大于这类抖动。

## 还没模拟的硬件条件

| 没模拟的 | 后果: 哪类参数调不准 |
| --- | --- |
| **Cube 实际算力** (`cube_mac_per_us` 缺省 0, 计算项不参与计时) | 计算密集的形状整体偏低, 且没有提示。改 tile 几何、改量化位宽这类影响计算量的参数, 结果不可用。需要 R1 |
| **多核抢 HBM 带宽** | 靠"多搬字节换少等待"的参数会显得免费。topk 预取那 1.6% 是上界 (HBM 写翻倍没计入时长) |
| **带宽随并发核数变化** | 单点标定 51.9, 而实测 28 核 45.3、18 核 37.0, 差 1.4 倍。改核数、改并发度的参数偏乐观。需要 R2 |
| **运行时取活的开销** | 动态取活只有收益没有代价。实测这笔开销到 0.15 µs 就把 1.34% 的收益吃光, 所以"动态取活更快"现在不能下结论。需要 R7 |
| **分段就绪的每段同步开销** | 分段只有收益 (开工更早) 没有代价 (每段多等一次标志), 所以"分得越细越好"是上界。需要 R8 |
| **指令异步发射与完成的分离** | 每个事件只有一个时长, 用相位拆分近似重叠; 真实发射开销没有 |
| **同步标志的身份与配对** | kernel 侧二十余个标志, 模型只有三个可填的延迟字段, 没有 set/wait 配对 |
| **跨核标志等待** | 只以依赖边出现, 所以"有就绪的活却空闲"是上界 (已声明两类高报: 硬件共位、相位组绑核) |
| **跨 Server 中继** | 单 Server 假设 |
| **A8W4 路径** | 激活的字节口径与带宽无实测 (仓内六个 run 全是 fp8), 给不出预测 |

一条使用边界: 只改编排 / 编译期 / 运行期**参数**时模拟器自动给结果; 改了 C++ 控制流、
同步协议、buffer 复用方式或流水阶段结构, 必须重新生成实现描述或 trace schema。

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
│   │   ├── activation.py        #   ACT tile: 合并 + epilogue 行块拆分
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
└── tools/                       # 16 个脚本: 清单对账 / 标定域 / 参数审计 / trace 比对 / golden / 报告
```
