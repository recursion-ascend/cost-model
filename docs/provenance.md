# 常数的出处与标定域

规格 (峰值) 与实测 (含争用) 为什么必须分开记, 以及模块常数改了会发生什么。
README 的「常数的出处」是摘要。


## 硬件规格 (spec) 与实测 (measured) 分开记

规格是**峰值**: 用它算出来的是时间下界, 所以要配效率系数。实测值已经含了争用与开销,
不该再乘效率。两者混在一个出处标签里早晚用错, 所以 `spec:` 是独立的一类
(`config/provenance.py` 的分类表)。

出处: 《昇腾 950 NPU 架构白皮书》(华为)。本仓 kernel 是 `__NPU_ARCH__==3510`
(第三代达芬奇 / arch35)。950PR 与 950DT 同源同 die, **每核计算峰值相同**, 差别在存储与
互连档位 —— 所以计算侧一组数, 存储侧两组 (`config/platform.py`)。

| 项 | 值 | 说明 |
| --- | --- | --- |
| 合计算力 (32 Cube / 64 Vector 档) | FP8/MXFP8/HiF8 919, BF16/FP16 486, MXFP4 1784, TF32 243 TFLOPS | 含 Vector 部分 |
| Vector 部分 | FP16/BF16 54, FP32 27 TFLOPS | 从合计里减掉它才是 Cube 部分 |
| 每 Cube 核 FP8 峰值 | 864 / 32 / 2 = **1.35e7 MAC/µs** | 除 2 是因为一次 MAC 两个 FLOP |
| 聚合 HBM | 950PR **1.6 TB/s** / 950DT **4 TB/s** | |
| 片间互连 (灵衢 2.0) | 2 TB/s | |

自洽核对: `2×432 + 54 = 918 ≈ 919`, `4×432 + 54 = 1782 ≈ 1784` —— 白皮书的"FP8 同频给
FP16 的 2 倍、MXFP4 给 4 倍"与合计值对得上。

**未逐字核对**: 这些数取自白皮书规格表的转述, 本容器的网络策略拦了 hiascend.com 与华为
OBS, 没能直接打开官方 PDF。拿到原件请核对 Cube/Vector 的算力拆分与 950PR 的 HBM 容量
(转述有 112GB/128GB 两说; 模型只用带宽, 不用容量)。

### 带宽上界校验: 单核常数不能突破聚合上界

`BW_L1_GM` 是**单核**实测值 (51.9 GB/s), 聚合 HBM 是规格上界。单核值乘活跃核数不得超过
聚合值, 否则那个时长物理上不可能:

| 活跃核数 | 950PR 占聚合 | 950DT 占聚合 |
| ---: | ---: | ---: |
| 28 (单卡真实可用) | **91%** | 36% |
| 32 | 104% (超) | 42% |
| 36 | 117% (超) | 47% |

所以 **950PR 在 28 核上已经抵消 HBM 聚合带宽的 91%** —— 任何增加 GM 流量的编排在这档上
几乎没有余量, 而 950DT 有 2.5 倍。`build_analytical_costs(platform=..., active_cores=...)`
会按 `min(单核上限, 聚合/活跃核数)` 压一次; `design_space(..., platform=...)` 则在每行给出
"这个方案需要的聚合带宽占规格的百分比", 超 100% 直接标 `超!`。

## 模块常数: 哪些有读者, 改了会发生什么

出处标签说"这个数是谁定的", 不说"它现在有没有进公式"。按"改了它会发生什么"分三类:

**一、改了什么都不会发生**

| 常数 | 实际取值走哪里 |
| --- | --- |
| `TOTAL_L1_SIZE` / `TOTAL_L0C_SIZE` / `VEC_REG_WIDTH` | 容量检查走 `KernelConfig.l1_size` 等可覆盖字段 |
| `T_INIT_US` / `T_INPUT_QUANT_FIXED_US` / `T_INPUT_QUANT_PER_TOKEN_US` / `T_CALL_OH` | 这些阶段不在 `kernel_total_us` 口径内 |
| `T_FILL_GMM1` (=0) | `Calibration` 的同名字段 |
| `BW_SCATTER` | 不进任何公式、不进任何申报, 只留复现记录 |

**二、公式读不到, 但 `tools/compile_manifest.py --check` 读得到** —— 它们是对 C++ 源码的
断言, 改了对账失配 (退出码 1):

| 常数 | 对账的 C++ 项 | 公式侧实际走哪里 |
| --- | --- | --- |
| `TOTAL_UB_SIZE` | `LAYERED_USABLE_UB_BYTES` | — |
| `L1_TILE_K` | `L1_TILE_K` | `KernelConfig.l1_tile_k` (模块常数只是 `select_kl1` 的缺省实参, 调用点都显式覆盖) |
| `GMM2_LAG_MIN_TOKEN_NUM` | `GMM2_LAG_MIN_TOKEN_NUM` | `InstancePolicy.gmm2_lag_threshold` |

**三、有真读者**

`SCALE_TRANSFER_BYTES` 进 `select_kl1` 的容量判据 (`units * scale_a <= SCALE_TRANSFER_BYTES`
与同式的 `scale_b`)。改小它, 部分 tile 的 kL1 从 512 掉回 256 (`select_kl1(120, 2048)`),
GMM2 沿 K 的分段数随之翻倍 —— 一个分段就绪的形状上事件数 563 → 947, 墙钟不变
(段多了但依赖都已满足)。后果在事件图里, 不在 `total_us` 上。
