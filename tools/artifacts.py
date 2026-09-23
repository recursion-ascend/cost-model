"""prof_rank*.bin 打点流解析 (prof_runs 工件).

tiling 解析已上收到包内 moe_cost_model.pipeline.parse_tiling, 此处只保留
prof bin 读取. 布局对应 megamoe_profile 打点 ring buffer 格式.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Tuple

from moe_cost_model import parse_tiling  # noqa: F401  (统一出口)

N_CORES = 84          # 每卡打点核数上限 (AIC+AIV 阵列)
RING_ALIGN = 64       # 每核 ring header 对齐
REC = 16              # 事件记录 (eid, payload, cycle)

STAGES = {0x6011: "DISPATCH", 0x6021: "GMM1", 0x6031: "ACT", 0x6041: "GMM2", 0x6051: "COMBINE"}


def read_prof_bin(path) -> Dict[int, List[Tuple[int, int, int]]]:
    """解析 prof_rank*.bin -> {core_id: [(cycle, event_id, payload)]} 按时间序."""
    raw = Path(path).read_bytes()
    ring = N_CORES * RING_ALIGN
    slots = (len(raw) - ring) // (N_CORES * REC)
    per = {}
    for cid in range(N_CORES):
        cnt = struct.unpack_from("<I", raw, cid * RING_ALIGN)[0]
        base = ring + cid * slots * REC
        ev = []
        for i in range(cnt):
            o = base + i * REC
            eid, pay = struct.unpack_from("<II", raw, o)
            cyc, = struct.unpack_from("<Q", raw, o + 8)
            ev.append((cyc, eid & 0xFFFF, pay))
        per[cid] = sorted(ev)
    return per
