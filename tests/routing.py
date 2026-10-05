"""测试夹具的路由助手: 从路由矩阵算出**守恒**的 token 数.

守恒是算法事实: 每个 token 恰好选 topk 个路由专家, 所以
  Σ_{dst,e} routing_counts[dst][e][src] == tokens × topk   (对每个 src)
model.simulate_multi 会拒绝不守恒的输入 (2026-10-05 起; 之前只警告, 于是多个夹具一直在
给物理上不可能的输入建图, 例如 64 token × top-8 最多 512 行却发出 1176 行)。

夹具通常先写好路由矩阵 (它表达的是"这个测试要的形状"), 再需要一个与之自洽的 token 数 ——
那正是本函数。手写 token 数再凑路由很容易写出不守恒的组合, 所以优先用这个。
"""
from typing import Sequence


def conserving_tokens(routing_counts: Sequence, topk: int) -> int:
    """守恒要求的每 rank token 数 = 每源 rank 发出的行数 / topk.

    逐源核对一致 (各源发出的行数必须相同, 否则一个 token 数无法同时满足), 且必须整除。
    不满足就报错 —— 夹具该改路由矩阵, 不该让测试带着不可能的输入跑。
    """
    world = len(routing_counts)
    per_src = []
    for src in range(world):
        per_src.append(sum(int(routing_counts[dst][e][src])
                           for dst in range(world)
                           for e in range(len(routing_counts[dst]))))
    if len(set(per_src)) != 1:
        raise ValueError(
            f"各源 rank 发出的行数不同 {per_src}: 一个 token 数无法同时满足, 改路由矩阵")
    rows = per_src[0]
    if rows % int(topk):
        raise ValueError(
            f"每源 {rows} 行不是 topk={topk} 的整数倍: 凑不出整数 token 数, 改路由矩阵")
    return rows // int(topk)
