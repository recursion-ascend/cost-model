"""路由生成器: vendor 自 megamoe_profile/scripts/run_case.py 的 make_routing.

单一来源在 prof 仓 run_case.py; 路由模式变更必须双向同步.
case 字段: tokens/experts/topk/ep/seed/routing.
"""
from __future__ import annotations


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
