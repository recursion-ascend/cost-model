# 四层架构: 实现 / 编译点 / 运行期 / 事件图

回答"这是哪份 kernel 的哪个编译点", 以及换实现时哪些东西必须重新生成。
README 的「工作原理」是入口, 本文是细节。


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

`tests/test_topk_prefetch.py` 的 15 个测试约束这条路。

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
`profiles.MEGAMOE_A8W8` 的指纹一致, 有测试约束。

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

**差的是一个量, 不是一个参数**: `WeightAntiQuantComputeNzNk` 的向量吞吐。它的地位与 ACT 的
`ACT_BYTES_PER_VEC` / `BW_UB` 相同 —— 要实测。仓内没有 A8W4 的打点 (`data/` 下六个 run 的
`dtype` 都是 `fp8_e5m2`), 所以现在给不出。

这正是本项目的边界: 改同一个 variant 的参数可以自动出结果; 改了 C++ 控制流 / 同步协议 /
缓冲复用 / 流水阶段结构, 就必须重新生成实现描述并重新标定。
