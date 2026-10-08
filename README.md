# MegaMoE Cost Model

在 CPU 上模拟昇腾 NPU 的执行过程，算出一套编排/tiling 参数下的执行时间、各核利用率与瓶颈所在。
面向 MegaMoE 算子开发人员：改一个参数跑一次，差值就是那个选择的代价，不必上板、不必编译、不占板卡。

---

# 一、建模理论方法

## 1. 本体：事件图 + 多资源表调度

一次执行被展开成有限个事件。一个事件只有五样东西：

```
resources        本事件独占的资源 (某个核的某个执行角色)，占用区间就是事件自身那一段
deps             前置事件 + 每条边上的同步延迟 λ
acquires/release 计数信号量，可跨事件持有 —— 用来表达片上缓冲槽
colocate_with    必须与哪个具名事件落同一个核 (如 L0C→UB 的 Fixpipe 只在绑定对内)
duration_us      时长，由闭式公式给出
```

调度递推（`scheduler/engine.py`）：

```
t_base(e) = max( max_p(end_p + λ(e,p)),  max_r free(r) )
再按计数信号量未来的归还时刻向前搜索一个可准入时刻
从就绪集合里选"最早能开始"的那个提交
资源名写成池占位符时，具体落哪个核在提交那一刻才定
```

即 **P\|prec\|Cmax 的表调度**。核心简化也是最大的假设：**事件时长由公式给，与并发无关**；
并发、排队、空闲都是调度的结果，不回写进时长。

**贪心是模型的仿真机制，不是硬件行为。** 这份 kernel 里 tile 到核的分配是一个随程序推进的
滚动游标（`mega_moe/op_kernel/arch35/mega_moe_arch35.h` 的 `startBlockIdx_`，跨 stage 与专家连续
滚动），由程序的块索引算术决定；Ascend 上哪个 block 跑哪个 tile 也由程序决定，硬件不重分配工作。
所以没有一个"会做贪心决策的硬件调度器"。

## 2. 一般性：换规则不改引擎

引擎只做一件与规则无关的事——在多资源、计数信号量、共位约束下把事件摆到时间轴上。
具体派发规则由三条正交的轴表达：

| 轴 | 在哪 | 可选值 |
| --- | --- | --- |
| tile→核 的**绑定时刻** | `CoreAssignment` / `ModelOptions.late_bind_pools` | 编译期静态（`static_round_robin` 复现那个滚动游标 / `greedy_least_busy` / `contiguous_block`）；派发时刻绑定（池内选能最早开始的核） |
| **就绪集选择规则** | `SchedulingPolicy` | `earliest_start`（缺省）/ `critical_path_first` / `work_conserving_critical_path` / `priority_by_stage` |
| **核内程序序** | 依赖边 | URMA 路径的 AIV1 程序序链、`dispatch_pacing` |

策略可以声明 `wants_view = True`，引擎每步给它一个**只读状态视图**（`scheduler/view.py`）：
能问谁就绪、各自最早何时能开始、某个资源何时空出，也能做一步试探
`start_if(事件, {资源: 被占到何时})`。视图没有提交、没有回退，策略改不了状态。
`LookaheadOneStep` 是这个契约上的第一个实现（按"选了它之后整张图最早可能何时结束"的下界破平）。
**实测它不如缺省策略**：三个形状上持平 / +0.40% / +1.26%，而且慢 100–200 倍；把那个下界放到排序
首项（即允许为了更小的下界推迟开工）更差，+47%。原因是最小化一个松的下界不等于最小化墙钟，
而贪心的"最早开始"虽然短视却保证了工作守恒。它留在仓里的作用是把这条契约跑通，并作为后续多步
搜索的对照基线——要用它做结论得先有新证据。

尚缺的：**多步试探**（beam）需要能复制整个调度状态；派发时刻绑定的取活开销还没标定；
核内程序序只能手工加边，没有声明式原语。

## 3. 时长公式：机制性闭式，不是回归

每个 stage 一条公式，**项的结构**来自算法必做的事，**系数**来自标定或规格：

```
GMM tile       = max(载入, 计算)
dispatch 一段  = max(λ, 槽内重叠的行) + 多出来的行 × 每行节拍      ← 行级软流水
combine 一个窗 = 读回(tile + 路由元数据) + 本卡行写 + 跨卡行写
```

每个常数带出处标签（`spec:` / `algo:` / `impl:` / `measured:` / `assumed:`），并绑**标定域四元组**
（实现 id、编译指纹、形状域、拓扑）；域外的结果带 `out_of_domain` 判定。

## 4. 可证伪性：三条物理下界

```
算力下界   = 总乘加 / (核数 × Cube 速率)
带宽下界   = 算法必搬字节 / min(单核带宽 × 核数, 聚合 HBM 规格)
依赖链下界 = 必经链上每个 stage 最小一份的和
```

下界**只由形状决定，与编排无关**——它是尺子，不是预测。时长低于三者之最大即抛 `BoundViolation`：
穿透下界意味着漏算了某项代价，那个数不该拿去做决策。`examples/scenario_basic.toml` 上三条是
25.57 / 1665.37 / 68.06 µs，带宽绑定。

## 5. 工作守恒：模型自己的不变量

核空闲分两类（`analysis/idle.py`）：

- **forced**：此刻全局没有就绪的活，是 DAG 逼出来的，换绑定方式消不掉；
- **avoidable**：有就绪的活却有核空着。

"就绪"的定义刻意**不含**"我自己的资源空着"，只含依赖已满足 + 信号量可准入 + 非核独占资源空出。

护栏在模型层（`simulate_multi`，与下界同一个位置——护栏不能被入口绕过），缺省开：
**派发时刻绑定的角色池里出现 avoidable 就抛 `WorkConservationViolation`**，不返回这个时长。
三处例外，每处都有理由：

| 不检查 | 为什么 |
| --- | --- |
| 静态钉核的池 | 工作钉死在某个核上，那个核忙而别处空着时搬不过去。那是**那种分核方式的代价**（量出来就是结论，换绑定能回收），不是调度器没做到位 |
| 跑 ACT 的那个角色 | ACT 与它的 GMM1 必须同核，是成对漂移而不能独立落核；现在的测量分不开"配对逼出来的"与"真可回收的" |
| 声明 `work_conserving = False` 的策略 | 按优先级排序的策略**主动**让核空着去等高优先级的 stage，那是它的语义 |

实测：40 个 golden 形状强制改成缺省绑定后扫一遍，**0 违反**（两个曾报出的形状正是上表后两条）。

## 6. 表达力的上界（显式声明，不含糊）

- 一个事件内部**只有一个时长**：没有指令、没有周期、不区分发射与完成。能分辨的最小差异是
  某个 tile 的某个相位提前或推迟多少；只改 tile 内部指令行为的参数（指令调度、循环展开、
  双缓冲写法）在此评估不了。
- **不建模带宽争用**：`channel_bytes` 只累加成访存量，不参与准入、不影响任何时长。
- **写得出但表达不出后果的取值报错**，不静默忽略（stage 边的 `readiness` 校验就是这条）。

## 7. 已知最弱的四处（按影响排）

1. **Cube 速率缺省 0** → 计算项整个不参与计时，改 tile 几何、改量化位宽这类影响计算量的参数结果不可用；
2. **时长与并发无关 + 不建争用** → 靠"多搬字节换少等待"的方案显得免费；
3. **表调度对输入不单调**（Graham 1969 的时序异常）→ 实测一条依赖边加 0.01 µs 让总时长变化
   −2.81%，就绪均分 3 段比 2 段差 1.0%。所以**几个百分点以下的差值不能当有效差异读**，
   必须同一绑定方式、同一策略下比较；
4. **跨卡写无聚合上限** → `BW_REMOTE_WRITE` 是从最快 tile 反扣的单核值，而 combine 的主导项正是跨卡写。

---

# 二、项目如何使用

## 安装与跑一次

```bash
pip install -e .                      # 或直接 pytest (pyproject 已配 pythonpath)
python examples/run_scenario.py       # 跑 examples/scenario_basic.toml
```

```python
import moe_cost_model as m

scenario = m.load_scenario("examples/scenario_basic.toml")
result = m.simulate(scenario)
print(result["kernel_total_us"])                       # 执行时间 (到最后一个 combine 结束)
```

## 场景文件

表名与字段名就是对象属性名，写错字段名直接报错并给提示。

```toml
h = 6144                      # 专家 FFN 中间维
hidden_dim = 4096             # 隐藏维
aic_num = 28                  # 核数
profile = "megamoe-a8w8"      # 以某份实现的取值为底; 不写 = 最少假设, 不复现任何实现

[workload]
tokens = 64                   # 每 rank token 数
topk = 8
world = 4                     # rank 数
local_experts = 64
routing = "uniform"           # uniform | cyclic | random | explicit | file

[kernel]                      # 编译期参数 (tile 几何、量化模式、通信拓扑 ...)
tile_m = 256
tile_n = 256

[policy]                      # 波推进 (前瞻/滞后)
dispatch_lookahead = 2

[calibration]
cube_mac_per_us = 2.7e7       # 缺省 0 = 计算项不参与计时, 见"已知最弱的四处"
```

## 比较若干方案

方案描述是 Scenario 的点分路径，覆盖面与场景文件一致——**编排参数、tile 几何、分核/打包/调度
策略都能换**：

```python
rows = m.compare_variants(scenario, {
    "基线":          {},
    "tile_n 128":    {"kernel.tile_n": 128},
    "GMM2 事件粒度 2": {"options.granularity": {"gmm2": 2}},
    "逐 K 块就绪":    {"options.links": [{"producer": "activation",
                                          "consumer": "gmm2",
                                          "readiness": "per_chunk"}]},
    "派发时刻绑定":   {"options.late_bind_pools": ["AIC", "AIV1"]},
    "最闲核优先":     {"options.late_bind_pools": [],
                      "core_assignment": "greedy_least_busy"},
})
print(m.format_design_space(rows))
```

```
方案            时长        Δ       Δ%   利用率                瓶颈           关键路径变化         不变量
基线         1751.48    +0.00    +0.0%  AIC98% AIV02% AIV14%  bandwidth +5%  -                   ok | 静态钉核可回收 AIC 1023核·us
派发时刻绑定  1727.97   -23.51    -1.3%  AIC99% AIV02% AIV14%  bandwidth +4%  gmm2-39.1, act+10.4  ok
```

列的含义：**时长/Δ** 执行时间与相对基线的差；**利用率** 每个执行角色的忙碌和÷跨度和；
**瓶颈** 三条物理下界里哪条绑定、时长比它高多少；**关键路径变化** 关键路径上各 stage 的时长差；
**不变量** 工作守恒守住没有，以及静态钉核留下多少可回收空闲。

注意两件事：核·µs 的可回收空闲 **不等于** 墙钟收益（上例 1023 核·µs 只换来 23.5 µs，因为这个形状
带宽绑定，空闲多半不在关键路径上）；几个百分点以下的差值不能当有效差异读（见理论第 7 条）。

## 一次运行能读到什么

| 字段 | 含义 |
| --- | --- |
| `kernel_total_us` / `kernel_dag_end_us` | 执行时间（到最后一个 combine 结束）/ 含尾段 |
| `rank_results[r]["events"]` | 每个事件的开始、结束、等了谁、等了多久、落在哪个核 |
| `resource_busy_us` / `resource_idle_us` / `resource_utilization` | 逐资源的忙碌、空闲、利用率 |
| `idle_decomposition` | 核空闲分解（forced / avoidable，含违规区间样本） |
| `bounds` | 三条下界、哪条绑定、是否穿透 |
| `traffic_bytes` | 逐通路搬了多少字节 |
| `critical_path` | 关键路径逐事件，含"因什么而关键" |
| `implementation` | 实现 id、源码依据、编译指纹、运行拓扑 |
| `provenance` | 本次用到的常数及其出处标签 |

## 命令行工具

```bash
python tools/gen_golden.py --check                     # 40 个配置的调度指纹逐位核对
python tools/knob_audit.py                             # 每个参数在五个形状上: 生效/换形状才动/被拒/未建模
python tools/check_work_conservation.py <scenario> --assert-conserving
python tools/compile_manifest.py --check               # 编译参数与 C++ 源码对账
python tools/calibration_domain.py --scenario <scenario>   # 这次结果在不在标定域内
python tools/compare_trace_structure.py                # 预测事件图 vs 实测 trace 的结构比对
```

---

# 三、项目结构

```
moe-cost-model/
├── src/moe_cost_model/
│   ├── __init__.py              # 显式导出
│   ├── scenario.py              # 统一入口: Scenario / load_scenario / simulate
│   ├── api.py                   # simulate_routing_counts 底层入口
│   ├── model.py                 # 驱动: 选适配器 → 建图 → 调度 → 后处理 + 两道护栏
│   ├── shape.py                 # MegaMoeShape (工作量) / ModelOptions (编排参数)
│   ├── costs.py                 # 各 stage 的时长公式
│   ├── guardrails.py            # 路由守恒 / tiling 真值核对 / 信道尺度
│   ├── registry.py              # 策略名注册表
│   ├── profiles.py              # 复现某份实现用的成套取值 (MEGAMOE_A8W8 等)
│   ├── config/                  # 第 0 层: 纯参数, 不含逻辑
│   │   ├── hardware.py          #   硬件常数 + KernelConfig (编译点) + select_kl1
│   │   ├── platform.py          #   白皮书规格 (峰值算力/带宽), 与实测分开记
│   │   ├── stages.py            #   StageVocabulary: 一份实现有哪些 stage 与边
│   │   ├── links.py             #   StageLink: 一条 stage 边的就绪/落点/槽数
│   │   ├── readiness.py         #   Readiness: 消费者沿共享轴分几段就绪
│   │   ├── roles.py             #   RoleAssignment: 哪个 stage 跑在哪个执行角色
│   │   ├── granularity.py       #   StageGranularity: 一个事件覆盖多少个单元
│   │   ├── policy.py            #   InstancePolicy + StageWaveOffsets (波前瞻/滞后)
│   │   ├── pipeline.py          #   PipelineConstraints + QueueDepths + tiling 解析
│   │   └── provenance.py        #   常数出处标签系统
│   ├── implementations/         # 第 1.5 层: 这是哪份 kernel 的哪个编译点
│   │   ├── identity.py          #   ImplementationId = hardware.implementation.variant
│   │   ├── compile.py           #   CompileConfig: 编译轴 + 指纹
│   │   ├── runtime.py           #   RuntimeTopology: 几卡几核
│   │   ├── adapter.py           #   适配器接口 (stages / accepts / plan / lower)
│   │   ├── megamoe.py           #   三份实现: a8w8_wave / layered / a8w4 (已声明未建图)
│   │   ├── megamoe_stages.py    #   MegaMoE 的 stage 词汇表 (五个 stage + 四条边)
│   │   ├── manifest.py          #   编译清单: 从 C++/CMake 抽参数并与 Python 对账
│   │   └── calibration.py       #   标定域: 一个数只在它量过的四元组里有效
│   ├── scheduler/               # 第 2 层: 通用调度引擎 (不认识任何 stage 名)
│   │   ├── events.py            #   Event / ScheduledEvent / 池占位符
│   │   ├── engine.py            #   MultiResourceScheduler
│   │   ├── view.py              #   SchedulerView: 策略用的只读状态视图 + 一步试探
│   │   ├── policies.py          #   就绪集选择规则 (含 LookaheadOneStep)
│   │   └── normalize.py         #   删掉起不了约束的计数信号量 (判据是一条定理)
│   ├── planning/                # 第 3 层: 波规划 + tile 网格
│   │   ├── waves.py             #   plan_waves / swizzle / Layered 波规划
│   │   ├── core_assignment.py   #   static_round_robin / greedy_least_busy / contiguous_block
│   │   ├── wave_packing.py      #   sequential_greedy / longest_expert_first / balanced_waves
│   │   └── tile_grid.py         #   TileGrid: 行范围 × 列范围, 可自定义切分
│   ├── builders/                # 第 4 层: 事件图构建
│   │   ├── base.py              #   公共基类 (_event / 共享专家 / 尾段)
│   │   ├── context.py           #   BuildContext: 各 stage 之间的共享状态
│   │   ├── gmm1.py              #   GMM1 tile
│   │   ├── activation.py        #   ACT tile: 合并 + epilogue 行块拆分
│   │   ├── gmm2.py              #   GMM2 + 沿 K 分段就绪
│   │   ├── tiling.py            #   tile 合并 (事件粒度) 与标签
│   │   ├── barriers.py          #   全核栅栏 (融合 vs 分段)
│   │   ├── comm/                #   通信后端 (dispatch 与 combine)
│   │   │   ├── base.py          #     DispatchTransport / CombineTransport
│   │   │   ├── peerwrite.py     #     直写对端对称窗口 + 配对 tile combine
│   │   │   └── urma.py          #     URMA: 批量 GET/PUT + AIV1 程序序链
│   │   ├── mte.py               #   融合波编排
│   │   ├── layered.py           #   Layered 编排
│   │   └── pipeline_expand.py   #   相位拆分 + 信道/容量
│   ├── ir/                      # 事件图的类型化只读视图
│   │   ├── vocabulary.py        #   Engine / Pipe / MemorySpace / TokenKind
│   │   └── graph.py             #   classify_resource / classify_token / 图视图
│   ├── validation/              # 校验与实测对账
│   │   ├── invariants.py        #   建图代码必须满足的结构规则
│   │   ├── trace.py             #   Chrome Trace 读取 (容忍截断)
│   │   └── compare.py           #   预测事件图 vs 实测 trace 的结构比对
│   └── analysis/                # 第 6 层: 解读
│       ├── bounds.py            #   三条物理下界 (模型怎么证伪自己)
│       ├── idle.py              #   核空闲分解 + 工作守恒护栏
│       ├── critical_path.py     #   关键路径与等待归因
│       ├── design_space.py      #   compare_variants: 多方案一张表
│       ├── stealing.py          #   空闲核任务转移
│       └── sensitivity.py       #   标定值不确定度 → 结论区间
├── tests/                       # 引擎 / 建图 / 实现层 / IR / 校验 / golden / 场景
├── examples/                    # 场景文件 + 六个实测 run 的复现脚本 + 设计空间扫描
├── tools/                       # 指纹核对 / 参数审计 / 标定域 / trace 比对 / 报告
├── data/                        # 实测 run 的 trace 与配置 (打点 bin 不入库)
├── bench/                       # 访存带宽微基准 (标定用)
└── mega_moe/                    # vendored kernel 源码 (验证点, 不由本项目的 lint 管)
```
