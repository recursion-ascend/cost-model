# 核空闲分解

`forced` 与 `avoidable` 的定义、怎么把 `avoidable` 清零、以及这个度量的已声明上界。
`docs/why_aic_idles.md` 讲的是 AIC 为什么空闲, 本文讲怎么度量。


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

| 参数 | 作用 |
| --- | --- |
| `ModelOptions.combine_layout` | 写出落点布局: `"token_scatter"` (缺省, 落点由 token 全局编号定, UNPERMUTE 顺序读) / `"expert_contiguous"` (按专家连续写, 读侧改 gather)。布局决定统计的**跨度**; 跨度的代价系数 (`AnalyticalCombineCosts.scatter_us_per_row`) 缺省 0, 要由扫 token 数的 run 定。读侧代价尚未建模 —— 见 `docs` 缺口 10 |
| `ModelOptions.combine_granularity` | 一个 combine 事件覆盖多少工作: `"per_tile"` (缺省, 与 GMM2 tile 1:1 配对、与计算交错) / `"per_expert"` (一个专家切片一个事件、等自己那片 GMM2 做完)。与角色正交。实测合并省 0.5% 元数据字节但总时长 +30.5% —— 不过模型还算不出它的主要好处 (写侧跨度, 见 `docs` 缺口 10) |
| `ModelOptions.roles` | `stage -> 执行角色` 映射 (`RoleAssignment`): 哪个 stage 跑在 AIC / AIV0 / AIV1 上。矩阵乘只能在 AIC (物理), ACT 与它的 GMM1 必须同核 (Fixpipe), 其余可换。**实测在 A8W8 主路径上换角色不改总时长** —— 两个向量核利用率都不到 6%, 关键路径在 AIC |
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

**零空闲不等于最快**: 纯贪心在两个形状上把总时长拖长了 (有就绪的活就立刻上核, 可能把更关键
的 tile 挤后), 关键路径打破平手才把这部分补回来。反过来, 只换策略不开池也不够 —— 静态绑定
下 6 个形状里 4 个仍有 avoidable (AIC 最多 1711.9 核·us)。两件事互相独立。

⚠️ 一处残留保守: `"per_core"` 配速边按建图时的核号连, 晚绑定后可能指向别的核 (占总边数
1.4%~3.0%, 计入 `forced`, 总时长略高估)。详见 `docs/design_space_gaps.md` 缺口 3。

晚绑定与相位流水**可以同用**: 同一个 tile 的几个相位编成**核组**,
核号由组里最先派发的那个事件选定, 同组其余事件跟随 —— 相位事件不持核资源, 所以不能靠
`colocate_with` (它要求锚点先绑定, 而先跑的恰恰是不持核的那一相)。实测 9216/3: 静态+拆相位
220.83 us, 晚绑定+不拆 196.92 us, 两者叠起来 **183.62 us**, 不变量仍然全 0。
