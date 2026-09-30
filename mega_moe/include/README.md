# MegaMoE 打点演示

一个 Python 入口完成：读取配置 → CMake 编译 → 多卡执行 → 导出 Chrome trace JSON。
不做数值验证、不生成报告、不采集 msprof，不依赖 skill 或特定任务队列。

## 环境准备

需要 Ascend 950、兼容的驱动与 CANN、CMake、C/C++ 编译器，以及当前 Python 环境中的
`torch`、`torch_npu`、`cann_ops_transformer`。后者需提供
`cann_ops_transformer.ops.mc2.common.CommContextManager` 的 channel 后端。
这是通信环境依赖，不是通过 pip 安装普通绘图依赖可以替代的。

已运行环境参考：CANN 9.2.0-weekly.20260902.01，Ascend950PR_957c。
CANN 需包含 `tensor_api` 和 Ascend C 编译工具。本示例使用已安装 CANN 中的配套头文件，
请使用与仓库内核兼容的版本。其他 CANN 版本没有承诺兼容。
该版 CANN 打包工具不能正确枚举隐藏目录下的对象文件，请将仓库和构建目录放在不含隐藏目录的路径中。

```bash
source /your/cann/set_env.sh
python -m pip install -r megamoe_profile/requirements.txt
```

可通过 `CC`、`CXX` 环境变量选择编译器；通信扩展的即时编译也使用 `CXX`。
当前 PyTorch 配套扩展要求支持 C++20，本次使用 GCC 11；系统默认编译器较旧时，
请在启动前设置 `export CC=gcc-11 CXX=g++-11`（按实际安装路径调整）。
入口与所有 Python 子进程使用同一个解释器。
设备使用权限及排队由运行者所在环境管理，脚本本身直接以前台子进程运行。

## 运行

在仓库根目录：

```bash
python megamoe_profile/run.py \
  --config megamoe_profile/configs/four_card_v4_flash_shared.json5 \
  --name v4_shared
```

全部参数位于配置文件。支持 JSON5，以及本项目的 `#` 行尾注释。
`runtime.cann_env` 为空时沿用启动前加载的环境，也可填写自己的 `set_env.sh`。

| 参数        | 示例                                               |
| ----------- | -------------------------------------------------- |
| 卡数 / 设备 | EP=4，设备0、1、2、3                               |
| 每卡输入    | 8192 tokens，hidden=4096                           |
| 专家        | 全局256个路由专家，每token选6个；每卡1个共享专家   |
| 中间维度    | 2048                                               |
| 类型        | BF16输入输出；MXFP8 E5M2激活/权重；BF16 Combine    |
| 路由        | 固定 seed 的随机无放回 top-k，每rank使用 seed+rank |
| 调用次数    | 3次不打点预热，1次正式打点                         |

尺寸参考 [DeepSeek-V4-Flash-Base 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json)。
这是结构化输入、稀疏合成权重的算子演示，不加载模型权重。官方模型使用的FP8格式与这里的E5M2不同。
配置另支持 `fp8_e4m3fn`、循环路由 `cyclic`、共享专家数0；本入口未接入INT8或FP4。
较小的双卡示例见 `configs/two_card.json5`。参数需满足内核约束，脚本不做配置合法性检查。

## 输出

每次创建 `prof_runs/日期_时间_任务名/`：

```text
config.json5                         本次参数
run.log                              编译、运行与转换日志
build/                               CMake产物
raw/                                 原始打点、tiling及解析CSV
日期_时间_任务名_trace_rank0.json      单卡流水（其余rank同理）
日期_时间_任务名_trace_all_ranks.json  多卡合并流水
```

用 MindStudio Insight 等支持 Chrome trace 的工具打开 JSON。
每卡包含完整和隐藏 WAIT 两个平级组，轨道直接显示 AIC0/AIV0/AIV1。
共享专家显示 `SHARED_GMM1`、`SHARED_ACT_QUANT`、`SHARED_GMM2`。
各rank以各自kernel起点为零，未进行跨卡时钟对齐。

脚本成功仅表示执行及导出完成，不表示精度或性能验收通过。
Combine的打点末端未严格对齐MTE3完成，UNPERMUTE内还包含共享结果等待；
不要把这些区间直接当成纯传输或纯计算时间。失败查看 `run.log`；不自动重跑。

## 单独编译与修改打点

```bash
cmake -S megamoe_profile -B /your/build \
  -DPROFILE_CONFIG="$PWD/megamoe_profile/configs/four_card_v4_flash_shared.json5" \
  -DPROFILE_PYTHON="$(command -v python)"
cmake --build /your/build -j2
```

编译数据库为 `build/compile_commands.json`，可用于编辑器补全。
阶段枚举见 `include/profile_stages.h`，打点宏见 `include/profiler.h`；
内核直接引用源码中的标记，事件编号及配对表由编译过程生成。
原理见 [打点说明](docs/manual_marking.md) 和 [阶段边界](docs/stages.md)。
