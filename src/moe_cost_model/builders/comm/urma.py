"""URMA 传输: hcomm ReadNbi/WriteNbi 批量 GET/PUT (目的拉模型).

URMA Layered 路径. dispatch = flag 轮询 + 批量 GET + 本地 MTE 混合;
combine = 批量 PUT, WQE 跨专家积累, 满批/波尾提交.
GET 与 PUT 共享 AIV1 程序序链 (aiv1_last), 由 UrmaTransport 统一持有.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ...config.hardware import BW_LOCAL_GM, LAYERED_META_BYTES_PER_ROW, ceil_div
from ...costs import LayeredDispatchLayout
from ..context import BuildContext
from .base import CombineTransport, DispatchTransport

_ALIGN_32 = 32
_FLAG_WINDOW_BYTES = 2048  # 256 token × 8B (URMA_FLAG_WINDOW_TOKENS × URMA_FLAG_BYTES)


class UrmaTransport:
    """URMA 传输套件: dispatch(GET) + combine(PUT) 共享 AIV1 程序序链."""

    def __init__(self, urma_mech):
        self.urma = urma_mech
        self.aiv1_last: Dict[int, str] = {}
        self.dispatch = UrmaDispatch(self)
        self.combine = UrmaCombine(self)


class UrmaDispatch(DispatchTransport):

    def __init__(self, parent: UrmaTransport):
        self.parent = parent

    @property
    def urma(self):
        return self.parent.urma

    @property
    def aiv1_last(self):
        return self.parent.aiv1_last

    def add_wave(self, builder, ctx: BuildContext, w, shape, km, p, c, policy, shared_gates):
        if not shape.expert_source_tokens:
            raise ValueError("layered URMA receive requires exact expert_source_tokens")
        world = len(shape.expert_source_tokens[0])
        layout = LayeredDispatchLayout.from_hidden(shape.h)
        per_token = layout.get_bytes_per_token()
        # mask 槽字节 ≈ 路由容量/8 (numMaxTokensPerRank 未建模, 按 bs×topk 近似)
        mask_bytes = (shape.token_num * shape.topk + 7) // 8
        mask_bytes = (mask_bytes + _ALIGN_32 - 1) // _ALIGN_32 * _ALIGN_32
        tile_m = km.tile_m
        e_begin, e_end = w.begin.expert, w.end.expert
        # (expert, group) -> [(event, rows)]
        contrib: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}

        for src in range(world):
            core = src % p        # kernel GetChannelOwnerBlock: rank % blockNum
            is_local = (src == shape.rank_id)
            # 远端批量状态: 满 256 token 即发批 (kernel remoteBatchTokenCount),
            # 通道尾批在扫完本 Wave 全部专家后 flush。
            batch_units: List[Tuple[int, int, int]] = []   # (expert, group, rows)
            batch_cap = 0
            batch_idx = 0

            def _emit_remote_batch(final: bool) -> None:
                nonlocal batch_units, batch_cap, batch_idx
                if not batch_units:
                    return
                tokens = sum(x[2] for x in batch_units)
                data_bytes = tokens * per_token
                # flag 窗口轮询 (ReadNbi+Drain) + 批量 GET (双 WQE/token) + meta 落盘
                duration = (self.urma.flag_poll_us()
                            + self.urma.get_batch_us(data_bytes)
                            + (tokens * int(LAYERED_META_BYTES_PER_ROW) + _FLAG_WINDOW_BYTES)
                            / BW_LOCAL_GM)
                ev = builder._event(
                    f"W{w.index}.recv.s{src}.b{batch_idx}", (f"AIV1:{core}",), duration,
                    deps=(self.aiv1_last[core],) if core in self.aiv1_last else (),
                    meta={"stage": "dispatch_recv", "wave": w.index,
                          "core": core, "src_rank": src, "batch": batch_idx,
                          "tokens": tokens, "bytes": data_bytes,
                          "final": final})
                self.aiv1_last[core] = ev
                for (e_, g_, rows_in_batch) in batch_units:
                    contrib.setdefault((e_, g_), []).append((ev, rows_in_batch))
                batch_units, batch_cap, batch_idx = [], 0, batch_idx + 1

            for e in range(e_begin, e_end):
                rows = int(shape.expert_source_tokens[e][src])
                if rows <= 0:
                    continue
                # mask/count 扫描 (本卡 win 内该 (expert, src) 槽位)
                scan_deps = [d for d in (self.aiv1_last.get(core), shared_gates) if d]
                scan_name = builder._event(
                    f"W{w.index}.maskscan.s{src}.e{e}", (f"AIV1:{core}",),
                    (mask_bytes + _ALIGN_32) / BW_LOCAL_GM, deps=scan_deps,
                    meta={"stage": "mask_scan", "wave": w.index, "core": core,
                          "src_rank": src, "expert": e, "mask_bytes": mask_bytes})
                self.aiv1_last[core] = scan_name
                # 该 src 在专家 e 行空间中的区间 [prefix, prefix+rows)
                prefix = sum(int(x) for x in shape.expert_source_tokens[e][:src])
                g0 = prefix // tile_m
                g1 = (prefix + rows - 1) // tile_m
                for g in range(g0, g1 + 1):
                    gb = g * tile_m
                    m = max(0, min(prefix + rows, gb + tile_m) - max(prefix, gb))
                    if m <= 0:
                        continue
                    if is_local:
                        name = builder._event(
                            f"W{w.index}.localcopy.s{src}.e{e}.g{g}", (f"AIV1:{core}",),
                            m * layout.local_copy_bytes_per_token() / BW_LOCAL_GM,
                            deps=(self.aiv1_last[core],),
                            meta={"stage": "dispatch_local", "wave": w.index,
                                  "core": core, "src_rank": src, "expert": e,
                                  "mgroup": g, "m_rows": m})
                        self.aiv1_last[core] = name
                        contrib.setdefault((e, g), []).append((name, m))
                    else:
                        off = 0
                        while off < m:
                            take = min(m - off, 256 - batch_cap)
                            batch_units.append((e, g, take))
                            batch_cap += take
                            off += take
                            if batch_cap == 256:
                                _emit_remote_batch(final=False)

            if not is_local:
                _emit_remote_batch(final=True)   # 通道尾批 flush

        # dispatch_ready: 每 (expert, 256 行组) 汇合全部贡献通道
        for e in range(e_begin, e_end):
            rows_e = int(shape.expert_tokens[e])
            if rows_e <= 0:
                continue
            for g in range(ceil_div(rows_e, tile_m)):
                key = (e, g)
                if key in ctx.dispatch_ready_event:
                    raise ValueError(f"duplicate DispatchReady producer for {key}")
                got = contrib.get(key, [])
                got_rows = sum(x[1] for x in got)
                required = max(0, min(tile_m, rows_e - g * tile_m))
                if got_rows != required:
                    raise ValueError(
                        f"DispatchReady row mismatch for expert={e}, group={g}: "
                        f"got {got_rows}, expected {required}")
                rn = f"W{w.index}.dispatch_ready.e{e}.g{g}"
                builder._event(rn, (), 0.0, deps=tuple(x[0] for x in got), meta={
                    "stage": "dispatch_ready", "dst_rank": shape.rank_id,
                    "wave": w.index, "expert": e, "mgroup": g,
                    "required_rows": required, "contributed_rows": got_rows,
                    "contributor_count": len(got),
                    "contributor_events": tuple(x[0] for x in got),
                    "contributor_rows": tuple(x[1] for x in got)})
                ctx.dispatch_ready_event[key] = rn


class UrmaCombine(CombineTransport):

    def __init__(self, parent: UrmaTransport):
        self.parent = parent

    def on_gmm2_tile(self, builder, ctx, w, shape, si, sl, t, label, ntile, core,
                     gname, global_group, call_iteration):
        pass   # URMA 聚合按波批量; tail 名已由 gmm2 stage 登记进 gmm2_tail_by_group

    def flush_wave(self, builder, ctx: BuildContext, w, shape, km, p):
        world = len(shape.expert_source_tokens[0])
        h = shape.h
        tile_m = km.tile_m
        put_row_bytes = 2 * h          # COMBINE_NO_QUANT: BF16 行直接 PUT
        e_begin, e_end = w.begin.expert, w.end.expert

        for dst in range(world):
            core = dst % p
            # 该 (wave, dst) 的段序列 (expert, group, m) — expert 顺序
            segs: List[Tuple[int, int, int]] = []
            for e in range(e_begin, e_end):
                rows = int(shape.expert_source_tokens[e][dst])
                if rows <= 0:
                    continue
                prefix = sum(int(x) for x in shape.expert_source_tokens[e][:dst])
                g0 = prefix // tile_m
                g1 = (prefix + rows - 1) // tile_m
                for g in range(g0, g1 + 1):
                    gb = g * tile_m
                    m = max(0, min(prefix + rows, gb + tile_m) - max(prefix, gb))
                    if m > 0:
                        segs.append((e, g, m))
            if not segs:
                continue

            # 批 λ 归因: 每 (wave,dst) 256-token 满批 commit + 波尾 flush。
            # 段 [lo,hi) 内触发的满批数 = #{k≥1 : 256k-1 ∈ [lo,hi)};
            # flush (=1 当 total%256≠0) 归最后一个非空段。总 λ 数 = ceil(total/256)。
            n_lambda = [0] * len(segs)
            total = sum(s[2] for s in segs)
            offset = 0
            for si, (_, _, m) in enumerate(segs):
                lo, hi = offset, offset + m
                first_k = max(1, -(-(lo + 1) // 256))     # ceil((lo+1)/256)
                last_k = hi // 256                        # floor(hi/256)
                n_lambda[si] = max(0, last_k - first_k + 1)
                offset += m
            if total % 256 != 0:
                n_lambda[-1] += 1

            for si, (e, g, m) in enumerate(segs):
                deps = list(builder.gmm2_tail_by_group.get((e, g), ()))
                if core in self.parent.aiv1_last:
                    deps.append(self.parent.aiv1_last[core])
                if dst == shape.rank_id:
                    # 本地: 读 GMM2 输出 + MTE 写 combineSend (复用 MTE 标定公式)。
                    # 这一批按 dst 分好了, 全是写回本卡的行 → remote_rows=0;
                    # 跨卡的批走下面的 URMA PUT 公式, 不进 combine_tile。
                    duration = builder.costs.combine_tile(m, h, 0)
                else:
                    # 远端: meta 读 + 批量 PUT (WQE 引擎直读 GMM2 输出, 无本地写)
                    duration = (
                        m * int(LAYERED_META_BYTES_PER_ROW) / BW_LOCAL_GM
                        + n_lambda[si] * self.parent.urma.t_put_lat_us
                        + m * put_row_bytes / self.parent.urma.bw_put_single_bytes_per_us)
                name = builder._event(
                    f"W{w.index}.combine.e{e}.g{g}.d{dst}", (f"AIV1:{core}",),
                    duration, deps=deps,
                    meta={"stage": "combine", "wave": w.index, "expert": e,
                          "mgroup": g, "dst_rank": dst, "core": core,
                          "m_rows": m, "part": "layered",
                          "put_batches": n_lambda[si]})
                self.parent.aiv1_last[core] = name
