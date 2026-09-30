# 阶段与计量边界

| 阶段 | 含义 |
|---|---|
| INPUT_QUANT / 输入准备相关阶段 | 本卡输入量化及通信元信息准备 |
| DISPATCH_XFER / DISPATCH_LOCAL | 从远端或本卡窗口获取输入，包含卡内暂存 |
| GMM1 | gate/up 两路矩阵乘及阶段内搬运 |
| ACT_QUANT | SwiGLU激活、量化与结果写出 |
| GMM2 | 中间维度到输出维度的矩阵乘及搬运 |
| COMBINE | 将路由专家结果按目标token写回目标卡窗口；也可能包含本卡行 |
| SHARED_GMM1 / SHARED_ACT_QUANT / SHARED_GMM2 | 本卡共享专家的对应计算 |
| UNPERMUTE | 合并路由与共享专家结果；当前也包含等待共享GMM2结果的时间 |
| WAIT_* | 已单独标记的依赖或同步等待 |

原始事件payload包含专家与wave编码，共享专家使用最高位标识。示范导出器保留原始payload，
不额外推导shape、FLOP或带宽。GMM1/GMM2/ACT/COMBINE/WAIT_*的位段为：
`[31]`=共享专家标识，`[30:24]`=本卡专家号，`[23:16]`=全局wave序号（dispatch/gmm1/gmm2
三条流水各自计数，编号对同一物理wave一致；GMM2/Combine滞后GMM1一拍执行），
`[15:0]`=`n_group * expert_m_groups + m_group`（tile线性索引，专家内m-group=mLoc/256）。
DISPATCH_SCHEDULE使用`[23:16]`=dispatch wave序号、`[15:0]`=起始专家号；
DISPATCH_XFER/LOCAL保持独立编码：peer[31:24]、每行字节数/32[23:12]、行数[11:0]，
其所属wave由时间上包围它的DISPATCH_SCHEDULE区间给出。

Scalar读取时间戳。GMM1的FIX_S、ACT的V_S对齐用于观测异步完成；
Combine末端的MTE3_MTE2不保证Scalar读END时MTE3已经完成，不能直接算纯传输带宽。
不同阶段的外层区间可能包含内部子片段，不能相加作为实际总耗时。
不生成UNATTRIBUTED或任何估算的空白填充。

每rank生成完整/隐藏WAIT两个平级视图，隐藏只过滤，不移动时间戳；KERNEL不显示。
AIC0-00、AIV0-00、AIV1-00等轨道直接位于rank组内。
各rank分别从本卡kernel起点对齐到0，未进行跨卡时钟同步。
记录和完成对齐有扰动，导出工具不验证结果精度或衡量打点开销。
