# 可调的参数与覆盖审计

全部可调参数的清单与判定 (生效 / 需对的形状 / 被拒 / 未建模)。表由 `tools/knob_audit.py --markdown` 生成, 有测试核对粘贴的那份没有过期。


## 可调的参数与覆盖审计

### 可调的全部参数

(数据源: `knob_audit.EXPECTED` + `knob_audit.WHAT`), `tests/test_knob_coverage.py`
核对 README 里这份与生成结果逐字一致, 所以它不会和代码分叉。

判定的含义: **生效** = 五个形状上都动了模型; **生效\*** = 只在有作用对象的形状上动
(只有一波就谈不上超前几波); **被拒** = 模型显式拒绝该取值 (缺标定常数); **动不了** =
有参数但当前建模下没有可表达的后果。

<!-- BEGIN knob-table (generated: python tools/knob_audit.py --markdown) -->
| 参数 | 判定 | 作用 |
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
| `kernel.topo_urma` | 生效 | 通信路径: MTE 波循环 / URMA Layered 宏波循环 (换建图代码) |
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
python tools/knob_audit.py          # 五个形状逐参数扫一遍, 打印判定与差值
python tools/knob_audit.py --emit   # 重新生成 EXPECTED
```

**`topo_urma` 不是平级参数，是结构分叉。** 切换后波粒度从 256 行 m-group 变为专家范围，dispatch 从源推变为目的拉，combine 从配对 tile 变为批量 PUT。

| 参数                                                     | MTE 路径           | Layered 路径                              |
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

`KernelConfig` 的编译期参数（`l1_buf_num`、`l1_tile_k`、`combine_quant_mode`）以 `KernelConfig` 为唯一事实源。手工拼 `PrimitiveCosts` 时入口自动按 kernel 重绑公式，任何拼法都生效。

权重 (B 流) 是载入项里**更大**的那一股 (`b_load = wb·K·cols / bw_b`), 所以
`weight_nz` 与 `gmm1_b_reuse_frac` 都显著改时长。同一个 tile (m=256, K=6144, cols=256)
实测: 基线 90.917 µs; `weight_nz` 配 NZ 带宽 80000 得 **69.627 µs (−23%)**;
`gmm1_b_reuse_frac=0.53` 得 **62.430 µs (−31%)**。`gmm1_b_reuse_frac` 是比例, 不是布尔。

策略参数用名字引用：

| 参数                  | 可选名字                                                          | 作用                   |
| --------------------- | ----------------------------------------------------------------- | ------------------------ |
| `tile_grid`         | `row_major` (缺省) / `swizzled` / `split_rows`                   | GMM1/GMM2 的 tile 怎么切 |
| `wave_packing`      | `sequential_greedy` / `longest_expert_first` / `balanced_waves` | 专家怎么组成波           |
| `core_assignment`   | `static_round_robin` / `greedy_least_busy` / `contiguous_block` | tile 分给哪个核          |
| `scheduling_policy` | `earliest_start` / `critical_path_first` / `priority_by_stage`  | 就绪集里谁先跑           |
| `restructure`       | `idle_core_stealing`                                              | 运行时图重构             |
| `orchestration`     | `mte` / `layered` / `"包.模块:类"`                              | 用哪个建图代码             |

带参数时写成表：`{name = "split_rows", parts = 2}`。自定义策略用 `moe_cost_model.register(类别, 名字, 构造函数)` 注册。

### 覆盖审计: 每个参数都必须能动模型

这个项目是给算子工程师改**编排 / 编译期 / 运行期**参数用的, 所以一个参数扫出
**0 收益**必须能分清是哪一种 0。四种意思, 指示完全相反:

| 判定 | 意思 | 下一步 |
| --- | --- | --- |
| 生效 | 每个形状上都动 (与形状无关) | 这个取舍可以照着做 |
| 生效* | 至少一个形状上动, 本形状没有作用对象 | 换形状再扫: 只有一波谈不上超前几波, 只有一个 K 块谈不上逐块就绪 |
| 被拒 | 模型显式拒绝该取值 (缺标定 / 这条路径没实现) | 拒绝是诚实的 |
| 动不了 | 模型里**没有可表达的后果** | **最需要分辨的一类**: 这个 0 是模型的空白, 不是硬件的事实 |

```bash
python tools/knob_audit.py --quiet    # 五个互补形状 x 全部参数
```

参数树自动走 (`dataclasses.fields` + `scenario._NESTED`), 所以**新加的字段自动进审计**;
全量判定固定在 `knob_audit.EXPECTED`, `tests/test_knob_coverage.py` 守着它:
新参数忘了接线、老参数被改没了、"动不了"的声明过期了, 三种都会红。

扫法三条 (为什么结论能信):

1. 从**本场景的生效值**出发扰动, 不是从 dataclass 缺省值出发 —— 场景带 `profile` 时
   两者不同, 拿缺省值当基线会把"值根本没变"误判成"没有读者"。
2. 一个参数给**一串**候选取值 —— 翻倍常落在无语义的档上 (`l1_buf_num` 2→4 与 2 同构,
   2→1 才是关 ping-pong; `swizzle_direction` 只有 0/1 两档)。
3. 比对五项: 总时长 / 事件数 / 事件名集合 / 每个事件时长 / 逐信道字节。只看总时长会把
   "结构变了而两边恰好等长"当成没动。

审计当前给出的几项判定:

* **引擎队列深度不是参数。** 持核事件独占 AIC/AIV0/AIV1, 同核在途数恒 ≤ 1, 所以每核
  引擎队列的容量固定为 1; 相位拆分后的载入相位刻意不继承 `Q:*`, 否则容量 1 的引擎信号量
  会限制 L1 缓冲深度。要表达"更深的队列"必须先有发射开销这类可观测的物理后果, 模型里
  没有。图里起不了约束的计数信号量由 `scheduler/normalize.py` 按一条定理删掉 —— 判据、
  八类 token 的分类表与实测见 `docs/design_space_gaps.md` 的「空约束」一节。
* **`topk_weights_prefetch` 的唯一出处是 `KernelConfig`。** 它是编译期宏
  `MEGAMOE_TOPK_PREFETCH`, 属于编译点而不是编排选项。它有后果: epilogue 行块 256→128、
  GMM1 的输出改走 GM 往返、每个行块多一次 topk 权重读 (见 `docs/architecture.md`)。
* **`options.roles` 与 `options.epilogue_overheads` 在场景文件里写得出**: 文件里写
  `[options.roles]` 下 `combine = "AIV0"` 即可, 不必在 Python 里构造对象。

目前唯一标为"动不了"的是 `options.combine_layout`: 写侧只经 `scatter_us` 的
`(spread_slots/m) ** scatter_exponent`, 而 `scatter_exponent` 缺省 0 使指数项恒 1,
两种布局算出同一个数 —— 0 不是保守, 是实测把"落点跨度"这个机制否掉了; 读侧 UNPERMUTE
从顺序读变 gather 的代价完全没建模 (缺口 10)。

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
