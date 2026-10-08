# MegaMoE Cost Model

在 CPU 上模拟昇腾 NPU 的执行过程，算出一套编排 / tiling 参数下的 **执行时间、各核利用率、瓶颈所在**。

面向 MegaMoE 算子开发人员：改一个参数跑一次，差值就是那个选择的代价。不必上板、不必编译、不占板卡，
一次运行 0.2–4 秒，所以可以一口气扫几十组参数。它不回答"某次已有运行为什么慢"（那是 profiler 的事）。

```
路由计数 ──► 波规划 ──► 事件图 ──► 多资源调度 ──► 执行时间 + 利用率 + 瓶颈
             (实现适配器)  (与算子无关)
```

**目录**：[五分钟上手](#五分钟上手) · [一、建模理论方法](#一建模理论方法) ·
[二、项目如何使用](#二项目如何使用) · [三、项目结构](#三项目结构)

---

## 五分钟上手

```bash
pip install -e .                      # 或直接 pytest (pyproject 已配 pythonpath)
python examples/run_scenario.py       # 跑 examples/scenario_basic.toml
```

```python
import moe_cost_model as m

s = m.Scenario(
    workload=m.Workload(tokens=72, topk=6, world=8, local_experts=3, routing="uniform"),
    h=5120,              # 模型隐藏维
    hidden_dim=9216,     # = 2 × 中间维 (SwiGLU 的 gate + up), 即中间维 4608
    aic_num=28,
    profile="megamoe-a8w8",
)
r = m.simulate(s)
print(r["kernel_total_us"])        # 255.34
```

这一次运行给出：

```
执行时间   255.34 µs   (记到最后一个 combine 结束; 含尾段 260.78 µs)
物理下界   算力 40.45 / 带宽 149.01 / 依赖链 116.41 µs  ->  带宽绑定, 高出 71.4%
利用率     AIC 77%   AIV0 4.4%   AIV1 8.2%
可避免空闲 0 核·µs    (工作守恒成立)
访存量     GM→L1 292.0 MB
```

### 两个最容易弄错的字段

| 字段 | 含义 | 在公式里是什么 |
| --- | --- | --- |
| `h` | **模型隐藏维** | GMM1 的 K、GMM2 的 N |
| `hidden_dim` | **GMM1 的输出宽度 = 2 × 中间维**（SwiGLU 的 gate 与 up 两块投影） | GMM1 的 N；GMM2 的 K = `hidden_dim / 2` |

所以中间维 `I = hidden_dim / 2`（`analysis/bounds.py`、`builders/gmm2.py` 的 `k_gmm2`）。

> 一个 run 标着 `h5120_i4608`，就该写 `h = 5120`、`hidden_dim = 9216`。
> 写成 `hidden_dim = 4608` 建模的是中间维 2304，只有一半。

### 术语

| 词 | 在本项目里指什么 |
| --- | --- |
| **事件** | 调度的最小单位。只有五个字段（见下），一个事件一个时长 |
| **stage** | 一份实现把执行划成的阶段（MegaMoE：dispatch / gmm1 / activation / gmm2 / combine） |
| **执行角色** | 一个核上的三个执行单元：AIC（Cube）、AIV0、AIV1（两个向量核） |
| **池 / 绑定时刻** | 资源名写成 `AIC:*` 时，具体落哪个核在**提交那一刻**才定（派发时刻绑定）；否则是编译期静态钉核 |
| **就绪** | 依赖已满足 **且** 计数信号量可准入 **且** 非核独占资源空出。刻意不含"我自己的核空着" |
| **粒度** `granularity` | 一个事件覆盖多少份该 stage 的工作单元（打包，省同步点） |
| **就绪粒度** `readiness` | 消费者沿共享轴分几段独立就绪（时序，开工更早） |
| **标定域** | 一个标定值只在它量过的（实现 id、编译指纹、形状域、拓扑）四元组里有效 |

---

# 一、建模理论方法

## 1. 它把执行抽象成什么

一次执行被展开成有限个事件。**一个事件只有五样东西**：

| 字段 | 表达什么 |
| --- | --- |
| `resources` | 本事件独占的资源（某个核的某个执行角色），占用区间就是事件自身那一段 |
| `deps` | 前置事件 + 每条边上的同步延迟 λ |
| `acquires` / `releases` | 计数信号量，**可跨事件持有** —— 用来表达片上缓冲槽（取槽与放槽在不同事件里） |
| `colocate_with` | 必须与哪个具名事件落同一个核（L0C→UB 的 Fixpipe 只在绑定对内存在） |
| `duration_us` | 时长，由闭式公式给出 |

**这五样就是模型表达力的全部。** 表达不出来的约束，模型不会声称自己算了它。

## 2. 怎么排出时间

```
t_base(e) = max( max_p( end_p + λ(e,p) ),        ← 前置都结束了
                 max_r free(r) )                  ← 所需资源都空了
再按计数信号量未来的归还时刻向前搜索一个可准入时刻
从就绪集合里选"最早能开始"的那个提交
资源名是池占位符时, 具体落哪个核在提交那一刻才定
```

即 **P\|prec\|Cmax 的表调度**。核心简化也是最大的假设：**事件时长由公式给，与并发无关**——
并发、排队、空闲都是调度的**结果**，不回写进时长。

> **贪心是模型的仿真机制，不是硬件行为。**
> 这份 kernel 里 tile 到核的分配是一个随程序推进的滚动游标（`mega_moe_arch35.h` 的
> `startBlockIdx_`，跨 stage 与专家连续滚动；kernel 源码树不在本仓）；Ascend 上哪个 block 跑哪个
> tile 也由程序的块索引算术决定，硬件不重分配工作。所以不存在"会做贪心决策的硬件调度器"。

## 3. 一般性：换规则不改引擎

调度引擎只做一件与规则无关的事——在多资源、计数信号量、共位约束下把事件摆到时间轴上。
它不认识任何 stage 名。具体派发规则由三条**正交**的轴表达：

| 轴 | 在哪 | 可选 |
| --- | --- | --- |
| tile→核 的**绑定时刻** | `core_assignment` / `options.late_bind_pools` | 编译期静态（`static_round_robin` 复现那个滚动游标 / `greedy_least_busy` / `contiguous_block`）；派发时刻绑定 |
| **就绪集选择规则** | `scheduling_policy` | `earliest_start`（缺省）/ `critical_path_first` / `work_conserving_critical_path` / `priority_by_stage` |
| **核内程序序** | 依赖边 | URMA 路径的 AIV1 程序序链、`dispatch_pacing` |

策略可以声明 `wants_view = True`，引擎每步给它一个**只读状态视图**（`scheduler/view.py`）：能问谁就绪、
各自最早何时能开始、某个资源何时空出，也能做一步试探 `start_if(事件, {资源: 被占到何时})`。
视图没有提交、没有回退，策略改不了状态。

`LookaheadOneStep` 是这个契约上的第一个实现，**实测不如缺省策略**（三个形状：持平 / +0.40% /
+1.26%，慢 100–200 倍；把那个下界放排序首项更差，+47%）。原因是最小化一个松的下界 ≠ 最小化墙钟，
而贪心的"最早开始"虽然短视却保证了工作守恒。它留在仓里是为了把这条契约跑通，并作为后续多步搜索
（beam）的对照基线——用它做结论得先有新证据。多步搜索还缺"能复制整个调度状态"这一步。

## 4. 时长从哪来：机制性闭式，不是回归拟合

每个 stage 一条公式，**项的结构**来自算法必做的事，**系数**来自标定或规格：

```
GMM tile       = max(载入, 计算)
dispatch 一段  = max(λ, 槽内重叠的行) + 多出来的行 × 每行节拍      ← 行级软流水
combine 一个窗 = 读回(tile + 路由元数据) + 本卡行写 + 跨卡行写
```

每个常数带**出处标签**（`spec:` / `algo:` / `impl:` / `measured:` / `assumed:`），并绑**标定域四元组**；
域外的结果带 `out_of_domain` 判定，不假装通用。源码依据写成 `文件:行号` 的形式记在注释与
`source_refs` 里；**kernel 源码树不在本仓**（2026-10-08 删除），所以这些引用要人拿着 kernel
工程核对，仓内只校验引用的形式。

## 5. 模型怎么证伪自己：三条物理下界

```
算力下界   = 总乘加 / (核数 × 每核 Cube 速率)
             GMM1 每专家 m·h·hidden_dim,  GMM2 每专家 m·I·h
带宽下界   = 算法必搬字节 / min(单核带宽 × 核数, 聚合 HBM 规格)
依赖链下界 = 一个 token 必经链上每个 stage 最小一份的和
```

下界**只由形状决定，与编排无关**——换任何编排它都不变，所以它是尺子，不是预测。
时长低于三者之最大即抛 `BoundViolation`：穿透下界说明漏算了某项代价，那个数不该拿去做决策。

> 实例：上面 bs=72 那次运行三条是 40.45 / 149.01 / 116.41 µs，带宽绑定，模型值高出 71.4%。
> 这 71.4% 就是编排带来的那部分，它的构成由空闲分解给出。

## 6. 模型自己的不变量：工作守恒

核空闲分两类（`analysis/idle.py`）：

- **forced**：此刻全局没有就绪的活，是 DAG 逼出来的（t=0 时 GMM1 还在等 dispatch，所有 AIC 必须空着），
  换绑定方式消不掉；
- **avoidable**：**有**就绪的活却有核空着。

护栏挂在模型层（`simulate_multi`，与下界同一位置——护栏不能被入口绕过），缺省开：
**派发时刻绑定的角色池里出现 avoidable 就抛 `WorkConservationViolation`**，不返回这个时长。
三处不检查，每处都有理由：

| 不检查 | 为什么 |
| --- | --- |
| 静态钉核的池 | 工作钉死在某个核上，那个核忙而别处空着时搬不过去。那是**那种分核方式的代价**（量出来就是结论，换绑定能回收），不是调度器没做到位 |
| 跑 ACT 的那个角色 | ACT 与它的 GMM1 必须同核，成对漂移不能独立落核；现在的测量分不开"配对逼出来的"与"真可回收的" |
| 声明 `work_conserving = False` 的策略 | 按优先级排序的策略**主动**让核空着去等高优先级的 stage，那是它的语义 |

实测：40 个 golden 形状强制改成派发时刻绑定后扫一遍，**0 违反**。

> 一条要记住的精度：**核·µs 的可回收空闲 ≠ 墙钟收益**。`scenario_basic` 上静态钉核留下 1023 核·µs，
> 改成派发时刻绑定后只换来 23.5 µs（−1.34%）——那个形状带宽绑定，空闲多半不在关键路径上。

## 7. 表达力的上界（显式声明，不含糊）

- 一个事件内部**只有一个时长**：没有指令、没有周期、不区分发射与完成。能分辨的最小差异是
  某个 tile 的某个相位提前或推迟多少。**只改 tile 内部指令行为的参数**（指令调度、循环展开、
  双缓冲的具体写法）在此评估不了。
- **不建模带宽争用**：`channel_bytes` 只累加成访存量，不参与准入、不影响任何时长。
- **写得出但表达不出后果的取值直接报错**，不静默忽略（stage 边的 `readiness` 校验就是这条）。
  否则工程师在那儿扫一圈得到"0 收益"，会读成"硬件上也没收益"。

## 8. 已知最弱的四处（按影响排）

| 缺什么 | 后果：哪类结论不可用 |
| --- | --- |
| **Cube 速率缺省 0** | 计算项整个不参与计时。改 tile 几何、改量化位宽这类影响计算量的参数不可用。（注意：载入绑定的形状上给不给这个值**时长相同**，看输出分辨不出来） |
| **时长与并发无关 + 不建带宽争用** | 靠"多搬字节换少等待"的方案显得免费 |
| **表调度对输入不单调**（Graham 1969 的时序异常） | 实测一条依赖边加 0.01 µs 让总时长变化 **−2.81%**，就绪均分 3 段比 2 段差 1.0%。所以**几个百分点以下的差值不能当有效差异读**，且必须同一绑定方式、同一策略下比较 |
| **跨卡写无聚合上限** | `BW_REMOTE_WRITE` 是从最快 tile 反扣的**单核**值，28 核同写给出 28 倍聚合；而 combine 的主导项正是跨卡写 |

---

# 二、项目如何使用

## 2.1 场景文件

表名与字段名**就是对象属性名**，写错字段名直接报错并给出提示。

```toml
# examples/scenario_basic.toml
profile = "megamoe-a8w8"      # 以某份实现的取值为底; 不写 = 最少假设, 不复现任何实现
h = 6144                      # 模型隐藏维
hidden_dim = 4096             # = 2 × 中间维 (这里中间维 = 2048)
aic_num = 28                  # 核数
p1_override = 2               # 每波 m 组数; 0 = 按 kernel 分档自动取
p2_override = 1

[workload]
tokens = 64                   # 每 rank token 数
topk = 8
world = 4                     # rank 数
local_experts = 64            # 每 rank 专家数
routing = "uniform"           # uniform | cyclic | random | explicit | file

[kernel]                      # 编译期参数: tile 几何、量化模式、通信拓扑 ...
tile_m = 256
tile_n = 256
topo_urma = false             # true = URMA Layered 路径

[policy]                      # 波推进: 前瞻 / 滞后
dispatch_lookahead = 2

[calibration]
cube_mac_per_us = 2.7e7       # 缺省 0 = 计算项不参与计时, 见"已知最弱的四处"
```

```python
import moe_cost_model as m
r = m.simulate(m.load_scenario("examples/scenario_basic.toml"))
```

## 2.2 比较若干方案

方案描述是 **Scenario 的点分路径**，覆盖面与场景文件完全一致——编排参数、tile 几何、
分核 / 打包 / 调度策略**都能换**：

```python
rows = m.compare_variants(scenario, {
    "基线":            {},
    "GMM1 缓冲双槽":    {"options.links": [{"producer": "gmm1", "consumer": "activation",
                                           "location": "onchip", "depth": 2,
                                           "colocated_by_hardware": True}]},
    "GMM2 逐 K 块就绪": {"options.links": [{"producer": "activation", "consumer": "gmm2",
                                           "readiness": "per_chunk"}]},
    "tile_n 128":      {"kernel.tile_n": 128},
    "combine 整片":     {"options.combine_granularity": "per_expert"},
    "派发时刻绑定":     {"options.late_bind_pools": ["AIC", "AIV1"]},
})
print(m.format_design_space(rows))
```

bs=72 / h=5120 / 中间维 4608 / topk=6 / 8 卡那个形状上的实际输出：

```
方案                 时长        Δ        Δ%     利用率              瓶颈           关键路径变化
基线 (静态钉核)     255.34    +0.00    +0.0%   AIC94% AIV012%     bandwidth +71%   -
派发时刻绑定        255.34    +0.00    +0.0%   AIC94% AIV012%     bandwidth +71%   -
GMM1 缓冲双槽       226.14   -29.20   -11.4%   AIC99% AIV017%     bandwidth +52%   gmm1-64.7, gmm2+35.5
GMM2 逐 K 块就绪    245.51    -9.83    -3.9%   AIC92% AIV011%     bandwidth +65%   gmm2-15.8, act+6.0
tile_n 128          302.86   +47.51   +18.6%   AIC91% AIV08%      bandwidth +103%  gmm1+28.4, gmm2+14.2
combine 整片        407.91  +152.56   +59.7%   AIC94% AIV152%     dependency +52%  combine+152.6
```

怎么读这张表：

| 列 | 含义 |
| --- | --- |
| **时长 / Δ / Δ%** | 执行时间与相对基线的差 |
| **利用率** | 每个执行角色的忙碌之和 ÷ 跨度之和 |
| **瓶颈** | 三条物理下界里哪条绑定、时长比它高多少 |
| **关键路径变化** | 关键路径上各 stage 的时长差（总忙碌不变而时长变了，说明换了走的路） |
| **总忙碌变化** | 各 stage 忙碌的核·µs 差（工作量真的变了才会动） |
| **最大等待** | 关键路径上最大的那种等待（依赖 / 容量 / 资源排队） |
| **不变量** | 工作守恒守住没有；静态钉核留下多少可回收空闲 |

上例读出来的三条结论：GMM1 片上缓冲单槽改双槽是这个形状上最大的一笔（−11.4%，GMM1 自己省 64.7 µs、
GMM2 多付 35.5 µs）；`tile_n` 减半反而 +18.6%（B 流复用变差，GM→L1 多搬 79.6 MB）；combine 整片
+59.7% 且瓶颈从带宽换成依赖链（要等整个专家切片的 GMM2 做完）。

## 2.3 一次运行能读到什么

```python
r = m.simulate(scenario)
rr = r["rank_results"][r["slowest_rank"]]
```

| 字段 | 含义 |
| --- | --- |
| `r["kernel_total_us"]` / `kernel_dag_end_us` | 执行时间（到最后一个 combine 结束）/ 含尾段 |
| `rr["events"]` | 每个事件的开始、结束、等了谁、等了多久、落在哪个核 |
| `rr["resource_busy_us"]` / `resource_idle_us` / `resource_utilization` | 逐资源的忙碌、空闲、利用率 |
| `rr["idle_decomposition"]` | 核空闲分解（forced / avoidable，含违规区间样本） |
| `rr["bounds"]` | 三条下界、哪条绑定、是否穿透 |
| `rr["traffic_bytes"]` | 逐通路搬了多少字节 |
| `rr["stage_busy_us"]` | 各 stage 的忙碌核·µs |
| `rr["critical_path"]` | 关键路径逐事件，含"因什么而关键" |
| `rr["implementation"]` | 实现 id、源码依据、编译指纹、运行拓扑 |
| `r["provenance"]` | 本次用到的常数及其出处标签 |

## 2.4 两道护栏（缺省都开）

| 护栏 | 触发 | 关掉它 |
| --- | --- | --- |
| 物理下界 | 时长低于算力 / 带宽 / 依赖链三条之最大 → `BoundViolation` | `simulate(..., check_bounds=False)`（只记录，用于排查） |
| 工作守恒 | 派发时刻绑定的池里出现 avoidable 空闲 → `WorkConservationViolation` | `A8W8WaveCostModel(..., check_work_conservation=False)` |

两道都挂在模型层，任何入口（`api` / `Scenario` / 直接构造）都绕不过去。

## 2.5 命令行工具

```bash
python tools/gen_golden.py --check                    # 40 个配置的调度指纹逐位核对 (约 40 秒)
python tools/knob_audit.py                            # 每个参数: 生效 / 换对形状才动 / 被拒 / 未建模
python tools/check_work_conservation.py <场景> --assert-conserving
python tools/calibration_domain.py --scenario <场景>  # 这次结果在不在标定域内
python tools/compare_trace_structure.py               # 预测事件图 vs 实测 trace 的结构比对
python tools/make_report.py <场景>                    # 单页 HTML 甘特图 + 等待归因
```

## 2.6 已知的使用限制

- `data/*/raw/` 的打点 bin 不入库，所以依赖实测 tiling 真值的场景（`examples/446172_bs128_*.toml` 等）
  在新克隆上跑不起来（5 个测试因此 skip）。在有 bin 的机器上跑一次 `python tools/export_tiling.py --all`
  把 json 导出入库即可。`examples/scenario_basic.toml` 与 Python 里直接构造的 `Scenario` 不受影响。
- 改了 C++ 控制流、同步协议、buffer 复用方式或流水阶段结构之后，必须重新生成实现描述或 trace schema——
  模型只对"改参数"自动给结果。

---

# 三、项目结构

```
moe-cost-model/
├── src/moe_cost_model/
│   ├── scenario.py              # 统一入口: Scenario / load_scenario / simulate
│   ├── api.py                   # simulate_routing_counts 底层入口 (直接给路由计数)
│   ├── model.py                 # 驱动: 选适配器 → 建图 → 调度 → 后处理 + 两道护栏
│   ├── shape.py                 # MegaMoeShape (工作量) / ModelOptions (编排参数)
│   ├── costs.py                 # 各 stage 的时长公式
│   ├── guardrails.py            # 路由守恒 / tiling 真值核对
│   ├── registry.py              # 策略名注册表 ("greedy_least_busy" → 类)
│   ├── profiles.py              # 复现某份实现用的成套取值 (MEGAMOE_A8W8 等)
│   │
│   ├── config/                  # 第 0 层: 纯参数, 不含逻辑
│   │   ├── hardware.py          #   硬件常数 + KernelConfig (编译点) + select_kl1
│   │   ├── platform.py          #   白皮书规格 (峰值算力/带宽), 与实测分开记
│   │   ├── stages.py            #   StageVocabulary: 一份实现有哪些 stage 与边
│   │   ├── links.py             #   StageLink: 一条 stage 边的就绪 / 落点 / 槽数
│   │   ├── readiness.py         #   Readiness: 消费者沿共享轴分几段就绪
│   │   ├── roles.py             #   RoleAssignment: 哪个 stage 跑在哪个执行角色
│   │   ├── granularity.py       #   StageGranularity: 一个事件覆盖多少个单元
│   │   ├── policy.py            #   InstancePolicy: 波前瞻 / 滞后
│   │   ├── pipeline.py          #   相位流水的队列深度 + tiling 文件解析
│   │   └── provenance.py        #   常数出处标签系统
│   │
│   ├── implementations/         # 第 1.5 层: 这是哪份 kernel 的哪个编译点
│   │   ├── identity.py          #   ImplementationId = hardware.implementation.variant
│   │   ├── compile.py           #   CompileConfig: 编译轴 + 指纹
│   │   ├── runtime.py           #   RuntimeTopology: 几卡几核
│   │   ├── adapter.py           #   适配器接口 (stages / accepts / plan / lower)
│   │   ├── megamoe.py           #   三份实现: a8w8_wave / layered / a8w4 (已声明未建图)
│   │   ├── megamoe_stages.py    #   MegaMoE 的 stage 词汇表 (五个 stage + 四条边)
│   │   └── calibration.py       #   标定域: 一个数只在它量过的四元组里有效
│   │
│   ├── scheduler/               # 第 2 层: 通用调度引擎 (不认识任何 stage 名)
│   │   ├── events.py            #   Event / ScheduledEvent / 池占位符
│   │   ├── engine.py            #   MultiResourceScheduler
│   │   ├── view.py              #   SchedulerView: 策略用的只读状态视图 + 一步试探
│   │   ├── policies.py          #   就绪集选择规则
│   │   └── normalize.py         #   删掉起不了约束的计数信号量 (判据是一条定理)
│   │
│   ├── planning/                # 第 3 层: 波规划 + tile 网格
│   │   ├── waves.py             #   plan_waves / swizzle / Layered 波规划
│   │   ├── core_assignment.py   #   static_round_robin / greedy_least_busy / contiguous_block
│   │   ├── wave_packing.py      #   sequential_greedy / longest_expert_first / balanced_waves
│   │   └── tile_grid.py         #   TileGrid: 行范围 × 列范围, 可自定义切分
│   │
│   ├── builders/                # 第 4 层: 把一份实现的编排展开成事件
│   │   ├── base.py              #   公共基类 (发事件 / 共享专家 / 尾段)
│   │   ├── context.py           #   BuildContext: 各 stage 之间的共享状态
│   │   ├── gmm1.py / activation.py / gmm2.py
│   │   ├── tiling.py            #   tile 合并 (事件粒度) 与标签
│   │   ├── barriers.py          #   全核栅栏 (融合 vs 分段)
│   │   ├── comm/                #   通信后端
│   │   │   ├── base.py          #     DispatchTransport / CombineTransport 接口
│   │   │   ├── peerwrite.py     #     直写对端对称窗口 + 配对 tile combine
│   │   │   └── urma.py          #     URMA: 批量 GET/PUT + AIV1 程序序链
│   │   ├── mte.py / layered.py  #   两种波编排
│   │   └── pipeline_expand.py   #   相位拆分 + 信道/容量
│   │
│   ├── ir/                      # 事件图的类型化只读视图
│   │   ├── vocabulary.py        #   Engine / Pipe / MemorySpace / TokenKind
│   │   └── graph.py             #   资源与令牌的分型 + 图视图
│   │
│   ├── validation/              # 校验与实测对账
│   │   ├── invariants.py        #   建图代码必须满足的结构规则
│   │   ├── trace.py             #   Chrome Trace 读取 (容忍截断)
│   │   └── compare.py           #   预测事件图 vs 实测 trace 的结构比对
│   │
│   └── analysis/                # 第 6 层: 解读
│       ├── bounds.py            #   三条物理下界 (模型怎么证伪自己)
│       ├── idle.py              #   核空闲分解 + 工作守恒护栏
│       ├── critical_path.py     #   关键路径与等待归因
│       ├── design_space.py      #   compare_variants: 多方案一张表
│       ├── stealing.py          #   空闲核任务转移
│       └── sensitivity.py       #   标定值不确定度 → 结论区间
│
├── tests/                       # 引擎 / 建图 / 实现层 / IR / 校验 / golden / 场景
├── examples/                    # 场景文件 + 六个实测 run 的复现脚本 + 设计空间扫描
├── tools/                       # 指纹核对 / 参数审计 / 标定域 / trace 比对 / 报告
├── data/                        # 实测 run 的 trace 与配置 (打点 bin 不入库)
└── bench/                       # 访存带宽微基准 (标定用)
```
