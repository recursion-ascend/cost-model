"""第 4 层: 通信协议接口 — dispatch / combine 的可插拔传输后端.

定位: cost model 是更高阶评估工具, 不复现某个特定 kernel —
默认配置恰好对应一个 kernel 实例, 调整传输/编排可映射到具体 kernel 实现.

  peerwrite: AIV 发 DataCopyPad 直接写入目的卡的对称窗口 (kernel 的 topoType=0,
             其源码称之为 MTE —— 那是核内搬运单元的名字, 见 comm/peerwrite.py)
  urma:      hcomm ReadNbi/WriteNbi 批量 GET/PUT

编排循环 (builders/mte.py 与 builders/layered.py) 只依赖本接口;
任意编排原则上可配任意传输 (对应假设的 kernel 变体, 未验证取值).
"""
from .base import CombineTransport, DispatchTransport
from .peerwrite import PeerWriteCombine, PeerWriteDispatch
from .urma import UrmaCombine, UrmaDispatch, UrmaTransport

__all__ = [
    "CombineTransport", "DispatchTransport",
    "PeerWriteCombine", "PeerWriteDispatch",
    "UrmaCombine", "UrmaDispatch", "UrmaTransport",
]
