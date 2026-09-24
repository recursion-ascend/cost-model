"""Mechanistic (source-faithful) Dispatch latency model with FCFS window queueing.

Physical form (per AIV1 DispatchTokenRange call):
    T_call = T_call,oh + sum over expert slices of slice_duration
    slice_duration = sum over segments [
        T_lat(src) + rows * bytes_read / BW(src)        (base service)
        + wait_eff(segment)                              (window contention)
    ]

Window contention model (validated on asym_single/puller/server experiments):
    * All AIV1 cores of a wave issue remote reads near-simultaneously; the
      per-core arrival order at a source window follows the core's dispatch
      begin offset (platform constant, NOT routing-dependent; order stability
      across routings measured r=+0.97).
    * A source window (wave, src) serves request streams FCFS; the data path
      serializes at S0 + rows*r_ser per stream (grants themselves are ~0.09us
      apart and not the bottleneck).
    * The frozen base coefficients (T_lat, BW) were calibrated on random
      routing where each window carries ~n_ref=7 streams, so they already
      absorb an average wait of ref_wait.  The simulator therefore adds only
      the EXCESS wait:  wait_eff = max(0, fcfs_wait - ref_wait).

Parameters are physical constants calibrated by controlled experiments
(differencing / direct slope reads), not regression over workload features.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

BYTES_ALIGN_QUANT = 256
BYTES_ALIGN_SCALE = 32


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass(frozen=True)
class DispatchDataLayout:
    route_items_per_batch: int = 256
    rev_token_elem_cnt: int = 6144   # = H, 1 byte/elem after E5M2 quant
    rev_scale_elem_cnt: int = 192    # = ceil(H/32)

    def bytes_read_per_row(self) -> int:
        return _align(self.rev_token_elem_cnt, BYTES_ALIGN_QUANT) + \
            _align(self.rev_scale_elem_cnt, BYTES_ALIGN_SCALE)

    @staticmethod
    def from_hidden(h: int) -> "DispatchDataLayout":
        return DispatchDataLayout(rev_token_elem_cnt=h, rev_scale_elem_cnt=(h + 31) // 32)


@dataclass(frozen=True)
class DispatchExpertIR:
    expert: int = 0
    dst_rank: int = 0
    row_begin: int = 0
    row_end: int = 0
    segments: Tuple[Tuple[int, int], ...] = ()   # (src_rank, rows) in execution order
    local_segments: int = 0
    remote_segments: int = 0
    rows: int = 0
    local_source_row_fetch_ops: int = 0
    remote_source_row_fetch_ops: int = 0

    def features(self) -> Dict[str, int]:
        return {
            "rows": self.rows,
            "local_segments": self.local_segments,
            "remote_segments": self.remote_segments,
            "local_source_row_fetch_ops": self.local_source_row_fetch_ops,
            "remote_source_row_fetch_ops": self.remote_source_row_fetch_ops,
        }


@dataclass(frozen=True)
class DispatchCallIR:
    dst_rank: int = 0
    wave: int = 0
    aiv1: int = 0
    global_row_begin: int = 0
    global_row_end: int = 0
    experts: Tuple[DispatchExpertIR, ...] = ()

    def features(self) -> Dict[str, int]:
        return {"calls": 1}


def build_dispatch_expert_ir(*, expert, dst_rank, source_counts, row_begin, row_end, layout):
    """Segments of one (core-local) expert row range, src-major order.

    source_counts: per-src row counts C[dst][expert][src] (execution order src0..srcP-1).
    """
    segs: List[Tuple[int, int]] = []
    cur = 0
    for src, cnt in enumerate(source_counts):
        nxt = cur + cnt
        lo = max(row_begin, cur)
        hi = min(row_end, nxt)
        if hi > lo:
            segs.append((src, hi - lo))
        cur = nxt
    local = [(s, r) for s, r in segs if s == dst_rank]
    remote = [(s, r) for s, r in segs if s != dst_rank]
    return DispatchExpertIR(
        expert=expert,
        dst_rank=dst_rank,
        row_begin=row_begin,
        row_end=row_end,
        segments=tuple(segs),
        local_segments=len(local),
        remote_segments=len(remote),
        rows=sum(r for _, r in segs),
        local_source_row_fetch_ops=sum(r for _, r in local),
        remote_source_row_fetch_ops=sum(r for _, r in remote),
    )


@dataclass
class DispatchMechanisticLatency:
    """Frozen-base service + FCFS window queueing.

    Base coefficients are the differencing-calibrated values at B=64 random
    routing (they absorb the n_ref~7 average window wait via ref_wait_us).
    """
    # ---- Directly measured physical constants (no fitting) ----
    # T_call_oh: empty-call mean (n=2133 events, asym data) = 1.006 us
    t_call_oh_us: float = 1.006  # 空调用均值 (µs)
    # T_lat_local: local 1-row single-segment call mean (4.82) - T_call_oh - row
    # cost (0.04) = 3.78 us;  BW_local: 1->2 row local slope 0.04 us/row = 158 GB/s
    t_lat_local_us: float = 3.78
    bw_local_bytes_per_us: float = 6336.0 / 0.04
    # T_lat_remote + window occupancy: single-window first-grant 3-row call
    # (asym_single w0, 5.14 us) = T_call_oh + T_lat_remote + 3*bytes/BW_window,
    # with BW_window from drain-occupancy slope (0.578 us per 2.9-row stream
    # = 0.19 us/row = 33 GB/s)  =>  T_lat_remote = 3.57 us.
    t_lat_remote_us: float = 3.57
    bw_remote_bytes_per_us: float = 6336.0 / 0.19      # window data-path occupancy
    # GMM1-overlap contention: dispatch calls of waves >= 1 execute while the
    # AIC streams GMM1 weights; measured excess of random w1-15 first-grants
    # (single-remote-segment calls) vs the clean prediction = +0.9 us PER CALL
    # (per-segment would double-charge multi-segment calls).  w0 dispatch runs
    # before any GMM1 (no term).  Magnitude scales with co-running GMM1
    # intensity (hot w2 during the 256-row GEMM shows ~+1.9 us).
    gmm1_overlap_us_per_call: float = 0.9
    # FCFS window queue: service occupancy = bytes / bw_remote (33 GB/s).
    window_bw_bytes_per_us: float = 6336.0 / 0.19
    # ref_wait REMOVED: base coefficients are uncontended by construction.
    ref_wait_us: float = 0.0
    # Platform constant: per-AIV1 dispatch begin offset (arrival order at the
    # windows).  Zeros => simultaneous arrival, FCFS tie-break by aiv1 id.
    begin_offset_us: Tuple[float, ...] = ()

    def _offset(self, aiv1: int) -> float:
        n = len(self.begin_offset_us)
        return self.begin_offset_us[aiv1 % n] if n else 0.0

    def call_base_us(self) -> float:
        return self.t_call_oh_us

    def segment_us(self, src: int, dst: int, rows: int,
                   layout: DispatchDataLayout = None) -> float:
        """单段基础服务 (无争用): λ_src + rows·b_row/BW_src.
        争用不在此计费 —— 远端段声明片间信道 (fab_src/fab_dst),
        由调度器 Channel 速率服务器按聚合带宽共享/降速 (带宽共享物理模型)."""
        if layout is None:
            layout = DispatchDataLayout()
        if src == dst:
            return self.t_lat_local_us + rows * self._bytes_row(layout) / self.bw_local_bytes_per_us
        return self.t_lat_remote_us + rows * self._bytes_row(layout) / self.bw_remote_bytes_per_us

    def _bytes_row(self, layout: DispatchDataLayout) -> int:
        return layout.bytes_read_per_row()

    def expert_us(self, ir: DispatchExpertIR, layout: DispatchDataLayout = None) -> float:
        """Uncontended base service of one expert slice (no window wait)."""
        if layout is None:
            layout = DispatchDataLayout()
        t = 0.0
        for src, rows in ir.segments:
            if src == ir.dst_rank:
                t += self._base_local_us(rows, layout)
            else:
                t += self._base_remote_us(rows, layout)
        return t

