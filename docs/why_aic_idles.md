# 核在什么情况下空闲

模型**不是"允许"空闲, 而是"算出"空闲**。调度器是逐资源 work-conserving 的 —— 只要某个核
上有事件满足下面全部准入条件, 它立刻启动, 模型从不主动让核闲着。

空闲全部来自两个外部输入:

- **DAG 的形状** (编排选择决定) -> 条件 ①⑥
- **绑定方式与资源配置** (编排选择决定) -> 条件 ③④⑤⑦

所以"不让核空闲"改的不是调度器, 是**喂给它的 DAG、绑定方式与资源声明**。

> **2026-10 更新**: 条件 ⑤ 已可通过 `ModelOptions.late_bind_pools` 消除 (派发时晚绑定),
> 配合 `WorkConservingCriticalPath` 策略, AIC/AIV0/AIV1 的 `avoidable_idle_us` 在 6 个
> 形状 x 3 种 `dispatch_pacing` 下全为 0。做法见本文末尾"如何消除"。

---

## 七个准入条件 (来自 scheduler/engine.py)

一个核在时刻 t 空闲, 等价于"没有任何能落到它上面的事件能在 t 启动"。事件不能启动只有
七个原因。

### ① 前置事件未完成

```python
dep_ready = max((end_by_name[d] + edge_latency(ev, d) for d in ev.deps), default=0.0)
```

`dep_ready > t` -> 不能启动。最主要的原因, 且是**物理必然** (数据还没产出)。

### ② 该核被占用 — **这一条不是空闲的原因**

```python
res_ready = max((resource_free.get(r, 0.0) for r in ev.resources), default=0.0)
```

`resource_free[AIC:c] > t`。**一个核被占用意味着它正在干活, 按定义就不空闲** —— 这一条
解释的是"某个事件为什么晚开始", 不是"某个核为什么空着"。

本文早期版本把 ② 列为空闲原因并与 ⑤ 并称, 那是错的: 让核空着的是 ⑤ (活在别的核上, 这个
核没分到), ② 只是那个活为什么还没做完。释放是**即时**的 (`resource_free[r] = end`, 无延迟)。

### ③ 计数信号量不足

```python
ok, t_cap = capacity_feasible(ev, t)
```

`Q:aic:c{core}` 的容量 (`EngineQueueDepths.aic`) 被占满 -> 推迟到下一个归还时刻。

与 ② 的区别在**持有时长**: `resources` 严格只占事件自身这一段; `acquires/releases`
可以**跨事件持有** (例如 `QUEUE:mte_aic` 由 load 相位取、fix 相位还, 表达的是一块 L1 缓冲
槽)。所以 ③ 能造成真正的空闲, 而 ② 不能。

**缺省容量 1, 且不开相位拆分时事件本就独占核资源, 所以缺省配置下这条与 ② 等价**,
不额外产生空闲。(晚绑定会把这类自取自还的按核 token 直接去掉, 见下。)

### ④ 信道带宽不足

```python
st, d = chan_state[cname].probe(ch_t, nbytes, entitled)
```

速率服务器判定窗口内剩余带宽不够 -> 降速或推迟。**缺省关闭** (只在 `default_channels=True`
或显式给 `PipelineConstraints.channels` 时生效), 不额外产生空闲。

### ⑤ 这个核根本没被分到活

建图时静态发牌 (`shape.py` 的 `BlockCursor.owners`) 没给它事件。n-tile 数 18 发给 28 核时,
10 个核对这个专家就是这种情况。

**这是 `avoidable_idle_us` 的唯一来源**, 也是唯一纯粹由编排选择造成的 —— 其余六条都有物理
或资源依据。已可通过晚绑定消除。

### ⑥ 边延迟未走完

```python
end_by_name[d] + edge_latency(ev, d) > t
```

前置事件**已经结束**, 但这条边上的同步延迟 (flag 握手 RTT, `dep_latency_us` /
`dep_latency_overrides`) 还没走完。缺省 0, 开启时是物理量 (实测 `WAIT_GMM1_BUFFER` 中位数)。

### ⑦ 非核的共享资源被占

`resources` 里除了核还可以写别的独占资源, 最典型是 `DISPATCH_COMM`
(`ModelOptions.serialize_dispatch_comm`, 表达"跨卡搬运通道一次只许一个核用")。这类资源被
占时, 事件等的不是核。

---

## 从空闲核自己的视角看, 只有两种情况

准入条件是"事件为什么不能启动"的分类。换成"**这个核为什么空着**"来问, 答案只有两种:

| 情况 | 含义 | 占比 (静态绑定实测) |
| --- | --- | ---: |
| **A** | 分给它的事件全做完了, 后面也没有它的活 | 26%~45% |
| **B** | 还有它的活, 但前置未完成 (①⑥) | 55%~74% |

情况 A 是你定的约束允许的: "该核被分到的活都做完了, 且后续没有其他 tile 任务才可以空闲"。
情况 B 是物理必然。

**真正的违规**是第三种: 有就绪的活、这个核空着, 但那个活被绑在**别的忙核**上 —— 即条件 ⑤。
`analysis/idle.py` 把它单独量出来叫 `avoidable_idle_us`。

---

## 一条常见的误解

"核不能有空闲"作为**绝对**约束在逻辑上不可能: 条件 ① 直接推出 t=0 时所有 AIC 必须空着
(GMM1 等 `dispatch_ready`, 后者等 AIV1 的 dispatch)。

能成立的不变量只有 **work-conserving**: 核不得在"存在已就绪的活"时空闲, 即
`idle_decomposition(...).avoidable_idle_us == 0`。用
`tools/check_work_conservation.py --assert-conserving` 判定。

---

## 跨波填充的三层限制

"一个专家只占 18 个 tile, 空闲的部分能不能让第二个专家进来" —— 能, 但要分清三层:

1. **同波内跨专家**: wave0 装了 E0+E1 共 36 个 tile, 本来就一起发给 28 核。这一层不需要
   任何改动, 现在就在做。
2. **跨波**: 下一波的 GMM1 要等它的 `dispatch_ready`, 也就是**下一个专家的 token 已经被
   dispatch 完**。这是条件 ①, 物理必然, 不能绕过 —— 但可以通过加深
   `WaveOffsets.dispatch` 让 dispatch 提前跑, 把这段等待缩短。
3. **绑定**: 即使活已就绪, 静态绑定也会把它钉在某个忙核上。这是条件 ⑤, 晚绑定解决。

---

## GMM2 的两层 K 结构

GMM2 的 K 轴就是 GMM1 切分的那个 N 轴 (`k_gmm2 = hidden_dim / activation_n_half`), 所以
"GMM2 要等整个 wave 的全部 activation"这个说法只对**尾段**成立:

- 模型已有两级 K 分段 (`ModelOptions.gmm2_k_segments`, 缺省 2): 首段只等覆盖首个 kL1 块的
  **1 个** ACT, 尾段等其余 17 个。
- 所以首段早就能在 wave 的 ACT 还没跑完时启动; 真正等整个 wave 的只有尾段。
- 分更多段能把这条等待链继续打散, 收益与代价见 README 的 `gmm2_k_segments` 一节。

---

## 如何消除 (2026-10 现状)

| 条件 | 消除手段 | 状态 |
| :---: | --- | --- |
| ⑤ | `ModelOptions.late_bind_pools=("AIC","AIV1")` — 事件只声明要一个 AIC/AIV1, 调度器在**派发时刻**绑最早空闲的成员 | 已实现, `avoidable` 归 0 |
| ⑤ 的顺序代价 | `scheduling_policy=WorkConservingCriticalPath()` — 零空闲之上按剩余关键链打破平手, 消掉纯贪心把关键 tile 挤后的回退 | 已实现 |
| ① 波间 | `dispatch_pacing="wave"` / `"none"` — 放开"下一波 dispatch 等本核上一波 combine"这条配速边 | 已实现 |
| ① 中段 | `gmm2_k_segments` 调细 | 已实现 |
| ① 尾段 | unpermute 分给 AIC (缺口 4) | 未实现 |
| ⑤ 的根因之一 | 跨核 split-K, 让 18 个 tile 变成 36/54 个更细的任务 (缺口 1) | 未实现 |
| ① 头部 | 消不掉 — GMM1 必须等 dispatch | 物理 |

实测 (ep=5, 每专家 256 行全远端, aic=28, hidden=9216 专家=3, rank0), 单位核·us:

| 配置 | AIC busy | AIC 利用率 | AIC forced | AIC avoidable | dag_end |
| --- | ---: | ---: | ---: | ---: | ---: |
| 静态绑定 + 贪心 | 5455.0 | 59.5% | 1693.2 | **2023.8** | 327.6 us |
| 双池晚绑定 + 贪心 | 5455.0 | 61.1% | 3479.5 | **0** | 319.1 us |
| 双池晚绑定 + 关键路径 | 5455.0 | 64.0% | 3071.7 | **0** | 304.5 us |

`busy` 三行完全相同 —— 晚绑定只换"哪个核做", 不改工作量。`avoidable` 清零后它转成了
`forced` (那段时间确实没有已就绪的活), 同时墙钟下降。

缺口编号见 `design_space_gaps.md`。
