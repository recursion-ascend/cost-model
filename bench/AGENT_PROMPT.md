# 交给服务器上 agent 的任务说明

把下面 `---` 之间的全部内容复制给那台机器上的 agent。它需要能访问 NPU、CANN 环境、
以及本仓库的 `bench/` 目录。

---

你的任务：在这台装有昇腾 NPU 和 CANN 的机器上，编译并运行 `bench/` 下的访存微基准，
把原始数据和分析结果带回来。这些数字要用来给一个 MoE 算子成本模型定死几个硬件常数，
所以**数据的诚实性比"跑通"重要得多**。

## 背景（决定了你该怎么处理异常）

这个基准要分离两组容易被混淆的量：

1. **每次请求的固定开销** vs **每字节的代价** —— 靠扫传输尺寸得到截距和斜率
2. **单核独占带宽** vs **整卡聚合带宽** —— 靠扫并发核数（`activeCores`）得到

第 2 条是全部重点。被测模型里现有的常数全是在 28 核并发下反解出来的"单核速率"，
里面已经含了平均争用；如果再拿它当无争用速率喂给一个按聚合带宽分配的仲裁器，争用
就被算了两遍。所以**并发=1 的那一档必须是真正只有一个核在动**，这是整个实验的关键。

## 第一步：环境信息（先报告，再动手）

```bash
echo $ASCEND_HOME_PATH
cat $ASCEND_HOME_PATH/../version.info 2>/dev/null || cat $ASCEND_HOME_PATH/version.info
npu-smi info
```

报告：CANN 版本、SoC 型号（`ascend950pr_957c` 之类）、卡数、每卡 AI Core 数（aic）
和 AI Vector 数（aiv）。**如果 aic 不是 28，告诉我**——`bench_mem_main.cpp` 里的
`BLOCKS` 和扫描用的 `CORES[]` 需要按实际核数改，改了要在报告里说明。

## 第二步：编译

```bash
source $ASCEND_HOME_PATH/../set_env.sh   # 或你环境里正确的 set_env.sh
cmake -S bench -B build_bench -DSOC_VERSION=<上一步查到的型号>
cmake --build build_bench -j
```

**这份 kernel 没有在 NPU 上编译验证过**（写它的机器没有 CANN）。编译失败是预期内的。
修的时候遵守下面的规矩：

**可以改**（这些是 API 适配，不影响测量语义）：
- `DataCopyPad` / `DataCopyExtParams` / `DataCopyPadExtParams` 的字段名、顺序、类型
- `LocalTensor` 直址构造的写法（`TPosition::VECCALC` / `TPosition::A1` 那几处）
- `GetTaskRation()` / `GetSubBlockIdx()` / `InitSocState()` / `KERNEL_TASK_TYPE_DEFAULT`
  的有无和替代写法
- `Exp` / `Div` / `Mul` 需要的额外头文件
- 链接库名（`ascendcl` 可能叫别的）、include 路径
- **跨核栅栏 `Barrier()`**——这是我最没把握的一段。只要保证语义：**每个计时区间之前，
  全部核（包括不参与计时的核）都到齐**。用什么手段随你（GM 原子加 + 自旋、
  `AscendC` 自带的 sync-all、`SyncAll()` 之类都行）。

**不要改**（这些是测量本身）：
- 6 个 case 各自测什么、循环结构、`GetSystemCycle()` 取样的位置（紧贴被测操作两侧）
- `activeCores` 的语义：`blockIdx >= activeCores` 的核**不做被测操作**，但**仍要参与栅栏**
- 扫描网格（`CORES[] / BYTES[] / ROWS[] / ROWB[] / VEC[]`）的取值
- 每核使用独立 8MB GM 区域这一点（防止地址冲突掩盖真实带宽）
- 输出的原始性：内核只吐 cycle，不在内核或 host 里做任何平均/拟合

**参考实现**：`mega_moe/op_kernel/arch35/` 下有大量同版本 CANN 的现成用法，
尤其 `stage/mega_moe_token_dispatch.h`（`DataCopyPad`、`LocalTensor` 直址、
`SetFlag`/`WaitFlag`）和 `stage/mega_moe_gmm2_combine.h`。**以那里的写法为准**。

改了什么，逐条记下来，报告里要有。

## 第三步：定标 cycle → 微秒

`GetSystemCycle()` 的频率要先定下来，否则所有绝对值都没意义。两个办法，**都做**：

1. 查这台机器的 system counter 频率（CANN 文档 / `aclrtGetSocName` 相关手册）
2. 实测：写个最小程序，`GetSystemCycle()` 取一次 → host 侧 `sleep 1s` 同步等待 →
   再取一次，或者用一个已知耗时的大传输反推

参考量级：同仓库之前的 trace 标定是 **~310–353 ticks/µs**。如果你量出来差很远，
以你量的为准，但要说明怎么量的。

## 第四步：运行

```bash
./build_bench/bench_mem 0 > bench_mem.csv
wc -l bench_mem.csv
python bench/analyze.py bench_mem.csv --cycles-per-us <你定标的值>
```

## 第五步：三个必做的自检（不做这三项，数据不可信）

1. **并发=1 真的只有一个核在动**：从 `bench_mem.csv` 里取 `active_cores=1` 的行，
   确认只有 `core=0` 有非零 cycle。
2. **栅栏真的起作用**：比较同一个 `bytes` 下 `active_cores=1` 与 `active_cores=28`
   的单核 cycle 中位。如果**完全相等**，栅栏很可能没生效（各核错开跑了，没有真并发），
   报告里要指出来。正常应该是 28 核时单核变慢。
3. **`scatter_store` 的 8B 档**：`rows=72, row_bytes=8` 这一点的耗时应该**远高于**
   按带宽算出的 576 字节所需时间——因为 8 字节远低于任何突发粒度，代价由请求次数决定。
   如果它和 512B 档差不多，说明 `rows` 那个循环被编译器优化掉了或者测错了。

## 第六步：报告格式

请按这个结构回报：

```
## 环境
CANN 版本 / SoC / 卡数 / aic / aiv / cycles_per_us（含定标方法）

## 编译改动
逐条：改了哪个文件哪一处、为什么、参照了哪个现成用法

## 自检结果
三项各自的结论（通过/不通过 + 支撑数字）

## analyze.py 输出
（原样粘贴）

## 原始数据
bench_mem.csv（整个文件，或者如果太大，按 case 分别给前后各 200 行 + 行数）

## 你注意到的异常
任何看起来不对的地方：非单调、方差异常大、某个 case 全零、数量级可疑等
```

## 最后几条硬要求

- **不要为了让数字"好看"而调任何东西。** 我要的是这台机器的真实行为，包括难看的和
  不符合预期的。非单调、方差大、和预期差一个数量级——都原样报告。
- **不要自己拟合或推断常数。** 你只负责跑出原始 cycle 和跑一遍 `analyze.py`。
  怎么解释、填进模型哪个位置，是我这边的事。
- **跑不通就说跑不通。** 如果某个 case 编译不过或跑崩，把那个 case 关掉、其余照跑，
  然后明确说明哪个没跑成、报什么错。**不要编数据，不要用估算值填空。**
- 如果你发现这个实验设计本身有问题（比如某个 case 测不到它声称要测的东西），
  说出来——这比跑完更有价值。

---

## 文件怎么传到那台机器

`bench/` 已经在版本库里，任选其一：

```bash
# 如果服务器能拉这个仓库
git pull && ls bench/

# 否则直接拷
scp -r bench/ 用户@服务器:/path/to/repo/
# 注意 bench/CMakeLists.txt 里的 include 路径不依赖仓库其他部分,
# 但 agent 可能需要 mega_moe/op_kernel/arch35/ 作为 API 写法的参考, 建议一起拷
scp -r mega_moe/ 用户@服务器:/path/to/repo/
```
