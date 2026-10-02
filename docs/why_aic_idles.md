# AIC 核在什么情况下空闲

模型**不是"允许"空闲, 而是"算出"空闲**。调度器本身是逐资源 work-conserving 的 ——
只要某个核上有事件满足下面全部准入条件, 它立刻启动 (`EarliestStart` 取最早可启动的),
模型从不主动让核闲着。

空闲全部来自两个外部输入:

- **DAG 的形状** (编排选择决定) -> 条件 ①
- **tile -> 核的静态绑定** (编排选择决定) -> 条件 ⑤

所以"不让 AIC 空闲"改的不是调度器, 是**喂给它的 DAG 与绑定方式**。

---

## 五个准入条件 (来自 scheduler/engine.py)

一个 AIC 在时刻 t 空闲, 等价于"没有任何分给它的事件能在 t 启动"。事件不能启动只有
五个原因:

### ① 前置事件未完成

```python
# engine.py:291-293
dep_ready = max((end_by_name[d] + edge_latency(ev, d) for d in ev.deps), default=0.0)
```

`dep_ready > t` -> 不能启动。最主要的原因, 且是**物理必然** (数据还没产出)。
`edge_latency` 是逐边延迟 (L0 同步延迟, 缺省 0)。

### ② 该核被占用

```python
# engine.py:295
res_ready = max((resource_free.get(r, 0.0) for r in ev.resources), default=0.0)
```

`resource_free[AIC:c] > t`。释放是**即时**的 (`resource_free[r] = end`, 无延迟)。

### ③ 计数信号量不足

```python
# engine.py:446-448
ok, t_cap = capacity_feasible(ev, t)
if not ok:
    return float("inf"), ...
```

`Q:aic:c{core}` 的容量 (`EngineQueueDepths.aic`) 被占满 -> 推迟到下一个归还时刻。
**缺省容量 1, 所以这条在缺省配置下与 ② 等价**, 不额外产生空闲。

### ④ 信道带宽不足

```python
# engine.py:433-442
st, d = chan_state[cname].probe(ch_t, nbytes, entitled)
if ch_t > t: ch_wait += ch_t - t; t = ch_t
```

速率服务器判定窗口内剩余带宽不够 -> 降速或推迟。**缺省关闭** (只在
`default_channels=True` 或显式给 `PipelineConstraints.channels` 时生效), 不额外产生空闲。

### ⑤ 这个核根本没被分到活

静态发牌 (`shape.py` 的 `BlockCursor.owners`) 没给它事件。n-tile 数 18 发给 28 核时,
10 个核对这个专家就是这种情况。**这是唯一纯粹由编排选择造成的**, 其余四条都有物理依据。

---

## 映射到实测的空闲分类

实测 (ep=5, 每专家 256 行全远端, aic=28, hidden=9216 专家=3, rank0):

| 核·us | 占比 | 条件 | 说明 |
| ---: | ---: | :---: | --- |
| 2023.8 | 54.0% | ②+⑤ | 有就绪活却空着 — 活在别的忙核上, 这个核没分到 |
| 691.9 | 18.5% | ① | 等 activation (GMM2 的 K 段还没齐) |
| 614.2 | 16.4% | ① | 尾段: epilogue 全在 AIV, AIC 无后继 |
| 227.6 | 6.1% | ① | 头部: GMM1 等 dispatch_ready |
| 171.7 | 4.6% | ② | 收尾: 剩下的活都在跑, 先做完的核无事 |
| 3.4 | 0.1% | ① | 等 dispatch_ready (波间) |

`analysis/idle.py` 的 `forced` / `avoidable` 两分法与此对应:
`avoidable` = 第一行 (②+⑤), `forced` = 其余。

---

## 按"能不能消除"分三类

| 类 | 条件 | 占比 | 消除手段 |
| --- | :---: | ---: | --- |
| **编排可消** | ⑤ (+②) | 22%~54% | tile->核晚绑定 (缺口 3)、跨核 split-K (缺口 1) |
| **改 DAG 可消** | ① 中段/尾段 | 40%~70% | `gmm2_k_segments` 调细、unpermute 分给 AIC (缺口 4) |
| **物理必然** | ① 头部 | 6%~8% | 消不掉 — GMM1 必须等 dispatch |

缺口编号见 `design_space_gaps.md`。

---

## 一条常见的误解

"AIC 不能有空闲"作为**绝对**约束在逻辑上不可能: 条件 ① 直接推出 t=0 时所有 AIC 必须
空着 (GMM1 等 `dispatch_ready`, 后者等 AIV1 的 dispatch)。

能成立的不变量只有 **work-conserving**: 核不得在"存在已就绪的活"时空闲, 即
`idle_decomposition(...).avoidable_idle_us == 0`。用
`tools/check_work_conservation.py --assert-conserving` 判定。
