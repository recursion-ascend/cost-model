"""Generate synthetic inputs and record MegaMoE invocations; no accuracy validation.

Timing contract (dataset mega_moe_fixed_case_p1_p2_validation):
- repeats (config key, default 1): number of formal measured launches.
- Per repeat: host perf_counter pair around launch+synchronize; device-side
  per-core KERNEL markers land in the prof ring buffer; both are saved.
- Routing truth: topkIds are generated on host with a fixed seed; the per-rank
  (dst,local_expert)->count slice and a sha256 of ids are logged per run.
- y checksum + device counts tensor are captured per repeat for cross-config
  identity checks (wave policy must not change math).
"""
from config_io import load_config
import hashlib
import json
import os
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path


def make_routing(case, rank):
    import torch

    tokens, experts, topk = case['tokens'], case['experts'], case['topk']
    world = case['ep']
    local = experts // world
    mode = case.get('routing', 'cyclic')
    if mode == 'random':
        generator = torch.Generator(device='cpu').manual_seed(case.get('seed', 0) + rank)
        # Bound temporary memory independently of the full batch size.
        chunks = []
        for start in range(0, tokens, 1024):
            scores = torch.rand((min(1024, tokens-start), experts), generator=generator)
            chunks.append(scores.topk(topk, dim=1).indices)
        return torch.cat(chunks, dim=0)
    offsets = torch.arange(tokens)[:, None]
    if mode == 'cyclic':
        return (offsets + torch.arange(topk)[None, :] + ((rank+1)%world)*local) % experts
    if mode == 'hot':
        # 极端不均衡路由: 每 token 的前 topk/2 个 slot 固定发给每 rank 的第一个
        # 本地 expert (global [0, local, 2*local, ...]), 其余 slot 均匀随机到
        # 非 hot expert。用于 cost model 极端路由泛化验证。
        hot_count = topk // 2
        hot_ids = torch.arange(world) * local  # [0, local, 2*local, 3*local]
        generator = torch.Generator(device='cpu').manual_seed(case.get('seed', 0) + rank)
        chunks = []
        for start in range(0, tokens, 1024):
            scores = torch.rand((min(1024, tokens - start), experts), generator=generator)
            scores[:, hot_ids] = -1.0
            cold = scores.topk(topk - hot_count, dim=1).indices
            chunks.append(cold)
        cold = torch.cat(chunks, dim=0)
        hot_part = hot_ids.unsqueeze(0).expand(tokens, hot_count)
        return torch.cat([hot_part, cold], dim=1)
    if mode in ('asym_puller', 'asym_server', 'asym_single'):
        # 互连瓶颈定位实验（B=64，与标定域同工况）：
        #   asym_puller: 拉取端隔离。所有 rank 的每 token 第 1 个 slot 发给 rank0 的
        #     expert0（每源 64 行），其余 slot 全部留在本 rank 本地专家。rank0 拉 3 条
        #     远端流共 192 行（同 hot w0 的拉取负载），但每个服务卡只被 1 家拉。
        #     若每行成本仍退化到 ~1.8us -> 瓶颈在拉取端。
        #   asym_server: 服务端隔离。rank1 的每 token 前 3 个 slot 分别发给 rank0/2/3
        #     的 expert0，其余本地；其它 rank 全本地（排除自身 e0）。rank1 共服务
        #     3 x 64 = 192 行（同 hot w0 的服务负载），但每个拉取卡只有 1 条流。
        #     若每行成本退化 -> 瓶颈在服务端。
        #   asym_single: 单流基线。仅 rank0 的 expert0 收 rank1 的 64 行，全网唯一
        #     远端流。给出无争用每行成本。
        generator = torch.Generator(device='cpu').manual_seed(case.get('seed', 0) + rank)
        hot_ids = []
        if mode == 'asym_puller':
            hot_ids = [0]
        elif mode == 'asym_server':
            if rank == 1:
                hot_ids = [0, 2 * local, 3 * local]
        elif mode == 'asym_single':
            if rank == 1:
                hot_ids = [0]
        chunks = []
        for start in range(0, tokens, 1024):
            scores = torch.rand((min(1024, tokens - start), experts), generator=generator)
            keep = torch.full_like(scores, -1.0)
            keep[:, rank * local:(rank + 1) * local] = 0.0
            # hot 目标由固定 slot 提供，必须从 cold 候选排除（避免 token 内重复专家
            # 与超额流量），与 hot 模式的 scores[:, hot_ids] = -1.0 同一约束
            for hid in hot_ids:
                keep[:, hid] = -1.0
            # 排除会被 hot slot 覆盖的自身 e0，避免 token 内重复专家
            if mode in ('asym_server', 'asym_single') and rank != 1:
                keep[:, rank * local] = -1.0
            if mode == 'asym_puller' and rank == 0:
                keep[:, 0] = -1.0
            cold_n = topk - len(hot_ids)
            chunks.append((scores + keep).topk(cold_n, dim=1).indices)
        cold = torch.cat(chunks, dim=0)
        if not hot_ids:
            return cold
        hot_part = torch.tensor(hot_ids, dtype=cold.dtype).unsqueeze(0).expand(tokens, len(hot_ids))
        return torch.cat([hot_part, cold], dim=1)
    if mode.startswith('l2_n'):
        # L2 B 矩阵命中率实验: 每 dst rank 的前 K 个专家均匀收满行, 其余为空
        # K = local/N (N=复用深度)。
        # 每 dst rank 总行数 = tokens*topk (4 个 src rank 的发送均摊到 4 个 dst)
        # B=2048, topk=8: 每 dst 16384 行 -> n=1:64专家×256行, n=8:8专家×2048行
        n_depth = int(mode.split('_n')[1])
        active_experts = local // n_depth
        if tokens * topk % active_experts != 0:
            raise ValueError(f'l2_n{n_depth}: tokens*topk={tokens*topk} not divisible by {active_experts}')
        ids = torch.zeros((tokens, topk), dtype=torch.int64)
        global_send = 0
        for t in range(tokens):
            for k in range(topk):
                slot = global_send % (world * active_experts)
                dst = slot // active_experts
                e = slot % active_experts
                ids[t, k] = dst * local + e
                global_send += 1
        return ids
    if mode in ('combine_seq', 'combine_stride', 'combine_random'):
        # COMBINE 散射写带宽实验: 控制同一专家内 token 的原始位置分布
        #   combine_seq:    token 块连续 -> 块顺序写
        #   combine_stride: token 交错   -> 步长写
        #   combine_random: 随机分布     -> 全散射写
        if mode == 'combine_random':
            generator = torch.Generator(device='cpu').manual_seed(case.get('seed', 0) + rank)
            chunks = []
            for start in range(0, tokens, 1024):
                scores = torch.rand((min(1024, tokens - start), experts), generator=generator)
                chunks.append(scores.topk(topk, dim=1).indices)
            return torch.cat(chunks, dim=0)
        # 块顺序/步长: 均在本 rank 内路由 (src=dst), 控制 token 在专家内的位置分布
        ids = torch.zeros((tokens, topk), dtype=torch.int64)
        for k in range(topk):
            for t in range(tokens):
                if mode == 'combine_seq':
                    # 连续块: token [j*blk,(j+1)*blk) -> expert j (blk = tokens/64)
                    e = (t * topk + k) % local  # 在本 rank 64 个专家内轮转
                else:  # combine_stride
                    # 步长: token i -> expert (i//stride) % 64
                    e = (t * topk + k) % local
                ids[t, k] = rank * local + e
        return ids
    raise NotImplementedError(f'Routing mode {mode!r} is not implemented')


def arguments():
    from types import SimpleNamespace
    root=Path(os.environ['MEGAMOE_RUN_DIR'])
    config=load_config(root/'config.json5')
    a=SimpleNamespace(**{k:v for k,v in config['case'].items() if k!='ep'})
    a.output_dir=root/'raw'
    return a,config['case']['ep']


def main():
    a, world = arguments()
    import ctypes
    import torch
    import torch_npu
    import torch.distributed as dist
    import cann_ops_transformer
    from cann_ops_transformer.ops.mc2.common import CommContextManager
    from types import SimpleNamespace

    root = Path(os.environ["MEGAMOE_RUN_DIR"])
    config=load_config(root/"config.json5")
    dev=config["device"]
    core_count=dev["aic_cores"]+dev["aiv_cores"]
    rank = int(os.environ['RANK'])
    torch_npu.npu.set_device(int(os.environ['LOCAL_RANK']))
    print(json.dumps({'rank': rank, 'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                      'package': cann_ops_transformer.__file__, 'case': vars(a)}, default=str), flush=True)
    dist.init_process_group('hccl', timeout=timedelta(seconds=config["runtime"]["distributed_timeout"]))
    group = dist.new_group(ranks=list(range(world)), backend='hccl')
    group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
    h, n, local = a.hidden, a.intermediate, a.experts // world
    dtype = getattr(torch, a.dtype.replace('fp8', 'float8'))

    def routing(r):
        return make_routing(config['case'], r)

    ids_cpu = routing(rank)
    t_input_prep_begin = time.perf_counter()
    gates_cpu = torch.full((a.tokens, a.topk), 1 / a.topk, dtype=torch.bfloat16)
    x = torch.ones((a.tokens, h), dtype=torch.bfloat16).npu()
    ids, gates = ids_cpu.to(torch.int32).npu(), gates_cpu.npu()
    # Generate weights on-device: a CPU float32 staging copy of w1/w2 costs
    # ~10 GB/rank at large H and OOMs the host when 4 ranks run concurrently.
    w1 = torch.zeros((local, 2 * n, h), device=x.device)
    w2 = torch.zeros((local, h, n), device=x.device)
    idx = torch.arange(n, device=x.device)
    col = idx % h
    for expert in range(local):
        w1[expert, idx, col] = 1
        w1[expert, idx + n, col] = 1
        # Distinguish expert ownership while keeping weights exactly representable.
        w2[expert, col, idx] = 1 + (rank * local + expert) % 2
    w1, w2 = w1.to(dtype), w2.to(dtype)
    s1 = torch.full((local, 2 * n, (h + 63) // 64, 2), 127,
                    dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    s2 = torch.full((local, h, (n + 63) // 64, 2), 127,
                    dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    shared = getattr(a, 'shared_experts', 0)
    shared_tensors = {}
    if shared:
        sw1 = torch.zeros((shared, 2*n, h))
        sw2 = torch.zeros((shared, h, n))
        for expert in range(shared):
            sw1[expert, idx, idx % h] = 1
            sw1[expert, idx+n, idx % h] = 1
            sw2[expert, idx % h, idx] = 1 + expert % 2
        shared_tensors = dict(
            sw1=sw1.to(dtype).npu(), sw2=sw2.to(dtype).npu(),
            ss1=torch.full((shared, 2*n, (h+63)//64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu(),
            ss2=torch.full((shared, h, (n+63)//64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu())
    # Use the exact layout compiled with this kernel, not the installed package's
    # potentially older bitmask sizing formula (the kernel uses compact indices).
    host = ctypes.CDLL(str(root/'build/libmegamoe_host.so'))
    host.megamoe_peermem_size.restype = ctypes.c_uint64
    required_bytes = host.megamoe_peermem_size()
    group_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank, init_comm=False)
    manager = CommContextManager(group_name, world, backend='channel', customCclBufferSize=required_bytes)
    buffer = SimpleNamespace(context=manager.create_context(), group=group, _ctx_manager=manager,
                             ccl_buffer_size=manager.ccl_buffer_size, comm_alg='ub-mem',
                             topo_type=manager.topo_type, rank_num_per_server=manager.rank_num_per_server)
    # Standalone repository harness: profiler address embedded in tiling, then GM->local word copy.
    lib = ctypes.CDLL(str(root/'build/lib/libmegamoe_kernel.so'))
    host.megamoe_tiling_size.restype = ctypes.c_uint64
    host.megamoe_make_tiling.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    host.megamoe_make_tiling.restype = ctypes.c_uint64
    prof = torch.zeros(core_count * (64 + dev["slots_per_core"] * 16), dtype=torch.uint8, device=x.device)
    td_cpu = ctypes.create_string_buffer(host.megamoe_tiling_size())
    workspace_bytes = host.megamoe_make_tiling(td_cpu, prof.data_ptr())
    assert workspace_bytes > 0, 'platform/tiling generation failed'
    (a.output_dir / f'tiling_rank{rank}.bin').write_bytes(td_cpu.raw)
    td = torch.tensor(list(td_cpu.raw), dtype=torch.uint8, device=x.device)
    workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device=x.device)
    # ListTensor descriptor format: offset=24, dim=0, count=1, sentinel, data ptr.
    descriptors = [torch.tensor([24,1 << 32,0xffffffff,t.data_ptr()],dtype=torch.int64,device=x.device)
                   for t in (w1,w2,s1,s2)]
    y = torch.empty_like(x)
    counts = torch.empty(local,dtype=torch.int32,device=x.device)
    lib.megamoe_profile_launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p]*16
    lib.megamoe_profile_launch.restype = None
    torch_npu.npu.synchronize()
    dist.barrier()
    stream = torch_npu.npu.current_stream().npu_stream
    # Core helper already supports nullptr: no GM records are written during warmup.
    warm_td_cpu = ctypes.create_string_buffer(host.megamoe_tiling_size())
    assert host.megamoe_make_tiling(warm_td_cpu, 0) == workspace_bytes
    warm_td = torch.tensor(list(warm_td_cpu.raw),dtype=torch.uint8,device=x.device)

    shared_descriptors = [torch.tensor([24,1 << 32,0xffffffff,t.data_ptr()], dtype=torch.int64, device=x.device)
                          for t in shared_tensors.values()]
    shared_ptrs = [t.data_ptr() for t in shared_descriptors] if shared else [0]*4
    tensors = [buffer.context,x,ids,gates,*descriptors[:2],*descriptors[2:],y,counts,workspace,td]
    for warm in range(a.warmup):
        lib.megamoe_profile_launch(dev["aic_cores"],stream,*[t.data_ptr() for t in tensors[:-1]],warm_td.data_ptr(),*shared_ptrs)
        torch_npu.npu.synchronize()
        print(f'WARMUP_DONE rank={rank} iteration={warm}',flush=True)

    # ---- routing truth (host-side, from the seeded topkIds actually fed to the kernel) ----
    # Placement (confirmed in op_kernel send_mask.h:83-84): dst = gid // moeExpertPerRank,
    # local = gid % moeExpertPerRank, i.e. contiguous uniform sharding.
    moe_per_rank = a.experts // world
    ids_np = ids_cpu.to(torch.int64).numpy()
    slice_counts = Counter()
    for gid in ids_np.reshape(-1).tolist():
        slice_counts[(int(gid) // moe_per_rank, int(gid) % moe_per_rank)] += 1
    routing_rows = [(d, e, d * moe_per_rank + e, c) for (d, e), c in sorted(slice_counts.items())]
    with open(a.output_dir / f'routing_rank{rank}.csv', 'w') as f:
        f.write('dst_rank,local_expert_id,global_expert_id,token_count\n')
        for d, e, g, c in routing_rows:
            f.write(f'{d},{e},{g},{c}\n')
    ids_hash = hashlib.sha256(ids_cpu.to(torch.int32).numpy().tobytes()).hexdigest()
    # global routing hash = sha256 over the 4 per-rank hashes in rank order (merged offline too)
    print(f'ROUTING_SLICE rank={rank} n_rows={len(routing_rows)} '
          f'sent_total={sum(c for _, _, _, c in routing_rows)} ids_sha256={ids_hash}',flush=True)

    repeats = int(getattr(a, 'repeats', 1))
    print(f'PROFILE_MEASURE_BEGIN rank={rank} launches={repeats} warmup={a.warmup} library={lib._name}',flush=True)
    input_prep_us = (time.perf_counter() - t_input_prep_begin) * 1e6
    for rep in range(repeats):
        prof.zero_()
        torch_npu.npu.synchronize()
        dist.barrier()
        # host 侧计时：4 进程同机共用同一单调时钟，跨 rank 可比。
        # launch_t/done_t 由后处理取 max(done)-min(launch) 得到"四卡都结束"的真实耗时。
        t_launch = time.perf_counter()
        lib.megamoe_profile_launch(dev["aic_cores"],stream,*[t.data_ptr() for t in tensors],*shared_ptrs)
        torch_npu.npu.synchronize()
        t_done = time.perf_counter()
        own_e2e_us = (t_done - t_launch) * 1e6
        (a.output_dir / f'prof_rank{rank}_rep{rep}.bin').write_bytes(prof.cpu().numpy().tobytes())
        counts_host = counts.cpu().numpy()
        y_sum = float(y.float().sum().item())
        # bf16 无法直接转 numpy：按原始字节（uint16 视图）哈希，保证跨 config 逐位可比。
        y_hash = hashlib.sha256(y.cpu().view(torch.uint16).numpy().tobytes()).hexdigest()
        y_finite = bool(torch.isfinite(y.float()).all().item())
        print(f'STEP_HOST_TIMING rank={rank} rep={rep} launch_t={t_launch:.9f} done_t={t_done:.9f} '
              f'own_e2e_us={own_e2e_us:.1f}',flush=True)
        print(f'RESULT_CHECK rank={rank} rep={rep} y_sha256={y_hash} y_sum={y_sum:.6f} '
              f'y_finite={y_finite} counts={json.dumps(counts_host.tolist())}',flush=True)
    print(f'PROFILE_MEASURE_END rank={rank} repeats={repeats}',flush=True)
    print(f'INPUT_PREP rank={rank} input_prepare_us={input_prep_us:.1f}',flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
