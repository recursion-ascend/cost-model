# 使用方法

本文件是**操作手册**: 装上、跑起来、看懂输出、改旋钮。
为什么这样建模、每个常数是谁定的, 看 `README.md`;
哪些编排表达得出、哪些还表达不出, 看 `docs/design_space_gaps.md`;
哪些系数还要上板测才能定, 看 `docs/calibration_runs.md`。

---

## 0. 装上

```bash
cd moe-cost-model
pip install -e .          # 或什么都不装: pyproject 已配 pythonpath
pytest tests/ -q          # 327 项 (5 项需 tiling 真值), 约 15 分钟
```

不想 `pip install` 就用 `PYTHONPATH=src python3 ...`。

---

## 1. 四个入口, 按"你想干什么"选

| 你想干什么 | 用哪个 | 一句话 |
| --- | --- | --- |
| 看一个形状跑多久 | `examples/run_basic.py` | 最底层: 直接给形状 + 代价公式 |
| 把场景写进文件, 改旋钮对比 | `examples/run_scenario.py` + `.toml` | **日常用这个** |
| 扫一片编排选择, 看每个值多少钱 | `examples/run_design_space.py` | 出一张对比表 |
| 和一次实测 run 逐 stage 对账 | `tools/compare_measured.py` | 需要 trace + tiling 工件 |

```bash
python examples/run_scenario.py        # 场景 + 变体对比
python examples/run_design_space.py    # 编排设计空间扫描 (950PR / 950DT 各一张表)
python tools/compare_measured.py data/<run_dir> examples/112575_bs36_noshared.toml
```

---

## 2. 日常路径: 写一个场景文件

`examples/scenario_basic.toml` 是模板。表名与字段名**就是对象属性名**,
写错会直接报错并提示正确拼写。

```toml
profile = "megamoe-a8w8"   # 以那份实现的取值为底; 不写 = 模型缺省 (最少假设)
h = 6144
hidden_dim = 4096
aic_num = 28               # 950PR 单卡真实可用 AIC 数

[workload]
tokens = 64                # 每 rank token 数
topk = 8
world = 4                  # rank 数
local_experts = 64
routing = "uniform"        # uniform | cyclic | random | explicit | file

[calibration]
cube_mac_per_us = 2.7e7    # **必填, 无缺省**。规格值见下方第 6 节

[kernel]
tile_m = 256
tile_n = 256

[policy]
dispatch_lookahead = 2

[options.granularity]        # 事件粒度, 每 stage 一个; 不写 = 全最细
# gmm2 = 2                   # 一个 GMM2 事件覆盖 2 个相邻 n-tile
# combine = 0                # 一个 combine 事件覆盖整个专家切片
```

跑它并改旋钮:

```python
from pathlib import Path
from moe_cost_model import load_scenario, simulate

base = load_scenario(Path("examples/scenario_basic.toml"))
for label, ov in {
        "基线": {},
        "tile_n=128": {"kernel.tile_n": 128},
        "GMM2 滞后 1 波": {"policy.gmm2_lag_waves": 1},
        "GMM2 攒 2 个 tile": {"options.granularity": {"gmm2": 2}},
}.items():
    r = simulate(base.with_overrides(ov))
    print(f"{label:24s} {r['kernel_total_us']:9.3f} us")
```

`profile = "megamoe-a8w8"` 是**一个坐标点**, 不是标准。
不写 `profile` 时用模型缺省 = 最少假设, 它不复现任何实现。

---

## 3. 输出怎么读

`simulate()` / `simulate_routing_counts()` 返回:

| 键 | 含义 |
| --- | --- |
| `kernel_total_us` | **执行时间**: 五个 stage, 记到最后一个 COMBINE 结束 |
| `kernel_dag_end_us` | 整个 DAG 末端 (含尾段 UNPERMUTE/FINALIZE 等) |
| `rank_results[i]["events"]` | 逐事件: 名字、资源、起止、等待分解、`meta` |
| `rank_results[i]["stage_busy_us"]` | 每 stage 的总忙碌核·µs |
| `avoidable_idle_us` | **护栏**: 必须是 0 —— 有就绪的活却有核空闲就是调度没做到位 |

每个事件的 `meta` 里有 `stage / wave / expert / mgroup / ntile / core /
row_begin..row_end / col_begin..col_end`, 粗粒度事件还有 `tiles_in_event`
(或 ACT 的 `gmm1_events_in_event`) —— 用来确认粒度旋钮**到底有没有生效**。

护栏工具:

```bash
python tools/check_work_conservation.py   # 核空闲分解: 哪些消不掉, 哪些是真浪费
python tools/audit_edges.py               # 依赖边自检 (ACT 孤儿 / head 挂错 K 块)
python tools/make_report.py               # 渲染可交互 HTML 报告
python tools/diagnose.py data/<run_dir>   # 六问诊断: 瓶颈/归因/改什么/收益/下一瓶颈
```

---

## 4. 旋钮速查: 想换什么就改哪一个

完整表在 `docs/design_space_gaps.md` 的"速查: 哪些编排已经能表达"。最常用的:

| 想换什么 | 改哪个 |
| --- | --- |
| tile 几何 (受 L1/L0C 容量约束, **物理**) | `KernelConfig.tile_m` / `tile_n` |
| **事件粒度** (同步点密度 ↔ 并行度, **编排**) | `ModelOptions.granularity`, 每 stage 一个 |
| stage 跑在哪个核 (AIC/AIV0/AIV1) | `ModelOptions.roles` |
| stage 之间那条边 (等多少/放哪/存几块) | `ModelOptions.links` 里的 `StageLink` |
| tile→核 怎么分 | `core_assignment` |
| tile→核 什么时候定 | `ModelOptions.late_bind_pools` |
| 波怎么打包 / 波宽 | `wave_packing` / `p1_override` `p2_override` |
| ready 集怎么选序 | `scheduling_policy` |
| 通信路径 (MTE / URMA Layered) | `KernelConfig.topo_urma` |
| 相位流水 (load/cube 跨 tile 重叠) | `ModelOptions.pipeline` |

### 事件粒度 (2026-10-04 起五个 stage 统一)

```python
OPT(granularity={"gmm1": 2, "gmm2": 2, "combine": 0})
#   1 = 最细 (缺省)   N = 攒 N 个单元   0 = 整片
#   dispatch 的 0 特殊: = 沿用 tiling 的 routeItemsPerBatch
```

三条必须知道的耦合:

1. **ACT 粒度 > 1 可能静默无效。** ACT 必须与产它的 GMM1 同核 (L0C→UB 的 Fixpipe),
   所以只在"喂它的 tile 既同核又 n 相邻"时才合并。轮转分核下相邻 n-tile 散在不同核,
   这时它是空操作 —— 查 `meta["gmm1_events_in_event"]` 确认。
2. **ACT 粒度 g 要求 UB 槽数 ≥ g** (`StageLink("gmm1","activation").depth`), 否则死锁,
   构造时直接报错。
3. **粗粒度不是收益开关, 是一笔交换。** 一个事件只能落一个核, 项数少于核数就有核闲着。
   两个方向都真实存在:
   * tile 数**填不满**核数时变慢 —— 28 核确定性夹具 (40 个 GMM1 tile) 上四种粗粒度
     全部变慢, `combine=0` 从 245.97 慢到 596.47;
   * tile 数**远多于**核数时变快 —— `examples/scenario_basic.toml` (4 卡 x 64 专家,
     28 核) 上 `gmm2=2` 从 1751.48 快到 **1741.46**。
   所以这个旋钮要**扫**, 不能照搬别人的取值。

---

## 5. 扫设计空间 (给算子工程师的主用法)

```python
import moe_cost_model as m
rows = m.design_space(run, points, platform=m.ASCEND_950PR)
print(m.format_design_space(rows))
```

`points` 是 `{方案名: ModelOptions}`。输出每行给出:
时长、Δ、Δ%、**关键路径落在哪个 stage**、总忙碌变化、最大等待类别
(capacity / dep / res)、访存量差、以及两条护栏 (`avoidable_idle` 是否为 0、
所需聚合带宽占平台规格的百分比)。

照着 `examples/run_design_space.py` 改 `points` 就行。

---

## 6. 两件必须自己填的事

### Cube 速率没有缺省

`cube_mac_per_us` 必填。规格推导值 (`config.platform.cube_mac_per_us`):

| dtype | 每 Cube 核 MAC/µs |
| --- | --- |
| fp8 | 1.35e7 |
| fp16 | 6.75e6 |
| mxfp4 | 2.7e7 |

规格峰值不等于可达速率, 真实效率要上板测 (`docs/calibration_runs.md` 的 R1)。

### 平台选哪个

```python
m.ASCEND_950PR   # 聚合 HBM 1.6 TB/s
m.ASCEND_950DT   # 聚合 HBM 4.0 TB/s
```

`aic_num=28` 是单卡真实可用核数 (不是规格的 32)。
模型**不建模带宽争用**, 所以聚合带宽是事后核对的护栏, 不是约束。

---

## 7. 精度边界: 用之前该知道的三条

1. **不建模带宽争用。** 调度只按资源独占排。实测中位数会比模型慢
   (bs128 的 COMBINE 中位是最快值的约 3 倍), 那部分是争用, 不是公式错。
   COMBINE 的公式对标的是"没被挤住"的那条。
2. **UNPERMUTE 读侧没建模** —— 只有一个 `字节/BW_UNPERMUTE_AGG` 的除法, 与输出落点
   布局无关。所以一旦给 `token_scatter` 填了写侧系数, `expert_contiguous` 会单方面
   显得变好, 那是模型的偏置, 不是结论。
3. **工作守恒 ≠ 最优** (Graham 异常)。放宽 UB 深度或加同步延迟在晚绑定下**可能让
   墙钟变差**。`avoidable_idle == 0` 只保证没有可避免的空闲, 不保证单调。

四个系数仍未标定 (`BW_L1_GM` 的并发分档、COMBINE 跨度、`BW_UB`、URMA PUT),
`BW_REMOTE_WRITE` 只定到量级 (4.5–9.5 GB/s 之间取了 8.6)。域外结论不作数。

---

## 8. 改代码时的约定

* 加常数必须带出处标签: `spec:` (硬件/格式标准) / `algo:` (算法定义) /
  `impl:` (某实现的选择) / `measured:` / `derived:` / `assumed:`。
  有测试强制, 且 `impl:` 的值必须能被参数覆盖。
* **不许用"等于现有 kernel"来定义缺省值。** 缺省是最少假设; 那份实现的取值放
  `profiles.MEGAMOE_A8W8`。
* 改了默认行为就要重新生成指纹: `python tools/gen_golden.py` (40 个用例)。
