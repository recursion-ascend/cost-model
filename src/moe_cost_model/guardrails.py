"""护栏: 在仿真前把"会静默给出错数"的输入形态挡下来.

三类护栏, 都来自实际踩过的坑:

1. tiling 真值核对 (check_against_tiling)
   场景文件里手抄的 kernel 参数会抄错, 抄对了也会随 kernel 改版失效。
   examples/bs36_4rank.toml 原先写着
       # p1/p2 由实测 tiling_rank0.bin 反解: mGroupsPerWave = 2
       p1_override = 2
   —— 注释说了出处, 但没人核对。给 [tiling] path 之后逐字段核对, 不一致直接报错。
   顺带把 kernel 真值 (行级软流水槽数、路由批大小) 直接采用, 不再靠缺省常数碰巧相等。

2. 信道尺度核对 (check_channels)
   速率服务器的两个常数必须同尺度: 逐事件速率是**无争用单核**带宽, 聚合是**整卡**
   带宽。片间信道占位就踩了这个坑 —— 聚合 BW_WINDOW=33000 是整卡值, 而逐事件速率
   BW_REMOTE_GM=31000 是从 28 核并发的真实运行反解的单核值 (已含平均争用), 于是
   单事件吃掉整卡 94%, 并发再叠 26 倍降速, 争用被计了两遍。
   本护栏给出两条可机检的判据 (见 check_channels 的实现注释)。

3. 路由守恒 (check_routing_conservation)
   C[dst][expert][src] 每个源 rank 发出的行数必须等于 tokens x topk。
   生成类路由不会错, explicit/file 会。

护栏只报告, 由调用方决定 raise 还是 warn (strict)。
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from .planning.waves import calc_m_groups_per_wave


# ---------------------------------------------------------------------------
# 1. tiling 真值
# ---------------------------------------------------------------------------

#: 场景字段 -> tiling 键. 只列两边都表达同一个量的字段。
TILING_SHAPE_KEYS = (
    ("h", "h", "h"),
    ("hidden_dim", "hidden", "hiddenDim"),
    ("aic_num", "aic", "aic"),
)
TILING_WORKLOAD_KEYS = (
    ("tokens", "bs", "bs"),
    ("topk", "topk", "topk"),
    ("world", "ep", "ep"),
    ("local_experts", "moeEpr", "每卡专家"),
    ("shared_expert_num", "shared", "共享专家"),
)


def check_against_tiling(scenario, tiling: Mapping[str, int]) -> List[str]:
    """核对场景与 tiling 真值, 返回不一致项的说明 (空 = 全对).

    包含 p1/p2 的间接核对: tiling 没有 p1/p2 (它们是主机侧策略), 但有派生量
    mGroupsPerWave —— 用场景的 p1/p2 重算一遍必须对上, 这正好抓住手抄错。
    """
    bad: List[str] = []
    for attr, key, label in TILING_SHAPE_KEYS:
        got, want = getattr(scenario, attr, None), tiling.get(key)
        if want is not None and got is not None and int(got) != int(want):
            bad.append(f"{label}: 场景 {got} != tiling {want}")
    wl = scenario.workload
    for attr, key, label in TILING_WORKLOAD_KEYS:
        got, want = getattr(wl, attr, None), tiling.get(key)
        if want is None or got is None:
            continue
        if int(got) != int(want):
            bad.append(f"{label}: 场景 {got} != tiling {want}")

    want_mgw = tiling.get("mGroupsPerWave")
    if want_mgw:
        p1 = scenario.p1_override or 1
        p2 = scenario.p2_override or 1
        try:
            got_mgw = calc_m_groups_per_wave(
                hidden_dim=int(scenario.hidden_dim), h=int(scenario.h),
                aic_num=int(scenario.aic_num), p1=p1, p2=p2,
                tile_n=int(scenario.kernel.tile_n))
        except ValueError as exc:
            bad.append(f"mGroupsPerWave: p1/p2 非法 ({exc})")
        else:
            if int(got_mgw) != int(want_mgw):
                bad.append(
                    f"mGroupsPerWave: 场景 p1={p1} p2={p2} 推出 {got_mgw} "
                    f"!= tiling {want_mgw} (p1/p2 是手填的, 很可能抄错或已随 kernel 改版失效)")
    return bad


#: tiling 键 -> 该直接采用的 kernel 真值 (不是标定值, 没有"域"的问题)
TILING_ADOPT = {
    "dispatchBufferCount": "dispatch 行级软流水槽数 (buffer_count)",
    "dispatchRouteItemsPerBatch": "dispatch 路由批大小 (route_items_per_batch)",
}


# ---------------------------------------------------------------------------
# 2. 信道尺度
# ---------------------------------------------------------------------------

#: 单事件占聚合的比例超过它就可疑: 说明两个常数大概不是同一尺度
SINGLE_EVENT_SHARE_WARN = 0.5


def check_channels(channels: Sequence, aic_num: int) -> List[str]:
    """核对速率服务器两个常数的尺度, 返回可疑项 (空 = 没发现问题).

    判据一 —— 单事件吃掉聚合的比例过大:
      逐事件速率应当是**单核独占**带宽, 聚合是**整卡**带宽, 所以正常情形下
      单事件顶多占聚合的 1/核数 量级。若一个事件就能占到一半以上, 通常意味着
      逐事件速率其实是个已含并发争用的整卡级数字 —— 拿它当单核速率, 再让速率
      服务器按聚合分一次, 争用就被计了两遍 (片间信道占位正是如此: 31000/33000
      = 94%)。

    判据二 —— 聚合恰好等于 每事件上限 x 核数 (刀刃配置):
      这种配置下"核数个满速率事件"把聚合吃到最后一位, 是个中性基线 (无争用,
      与闭式一致), 本身没错; 但只要再多一个消费者上这条信道就立刻超订, 而且
      浮点残渣会让第 核数+1 个事件分到接近 0 的速率。default_channels 就是这么
      配的, 所以往上加消费者前必须先有该层级真实的聚合带宽。
    """
    out: List[str] = []
    for ch in channels:
        total = float(getattr(ch, "bw_total", 0.0) or 0.0)
        cap = float(getattr(ch, "max_rate_per_event", 0.0) or 0.0)
        name = getattr(ch, "name", "?")
        if total <= 0 or cap <= 0:
            continue
        share = cap / total
        if share > SINGLE_EVENT_SHARE_WARN:
            out.append(
                f"信道 {name}: 单事件上限 {cap:.0f} 占聚合 {total:.0f} 的 {share:.0%} "
                f"—— 逐事件速率像是已含争用的整卡级数字, 与聚合不同尺度, 会双重计费")
        elif aic_num > 0 and abs(total - cap * aic_num) <= 1e-6 * max(1.0, total):
            out.append(
                f"信道 {name}: 聚合 {total:.0f} 恰好 = 每事件上限 {cap:.0f} x {aic_num} "
                f"—— 中性基线 (无争用), 再加一个消费者就会超订且尾事件分到近 0 速率; "
                f"共享前需要该层级真实的聚合带宽")
    return out


# ---------------------------------------------------------------------------
# 3. 路由守恒
# ---------------------------------------------------------------------------

def check_routing_conservation(routing_counts, tokens: int, topk: int) -> List[str]:
    """每个源 rank 发出的行数必须 = tokens x topk. 返回不守恒项."""
    bad: List[str] = []
    if not routing_counts:
        return bad
    world = len(routing_counts)
    want = int(tokens) * int(topk)
    sent: Dict[int, int] = {s: 0 for s in range(world)}
    for dst, experts in enumerate(routing_counts):
        for expert, srcs in enumerate(experts):
            for src, n in enumerate(srcs):
                if n < 0:
                    bad.append(f"routing_counts[{dst}][{expert}][{src}] = {n} < 0")
                sent[src] = sent.get(src, 0) + int(n)
    for src in sorted(sent):
        if sent[src] != want:
            bad.append(
                f"源 rank {src} 发出 {sent[src]} 行 != tokens x topk = {tokens} x {topk} "
                f"= {want}")
    return bad
