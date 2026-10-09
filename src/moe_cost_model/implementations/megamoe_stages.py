"""仓内两份 MegaMoE 实现的 stage 词汇表.

这是 ``config.stages.StageVocabulary`` 的一个**实例**, 不是框架的一部分:
MegaMoE 把自己划成 dispatch/GMM1/ACT/GMM2/combine 五个 stage 并把它们钉在
AIC/AIV0/AIV1 上, 这是这份 kernel 的编排, 换一个算子就换一份声明。

每一项的出处:
  pipeline / unit_of / default_items   config/granularity.py
  edges                                config/links.py EDGE_AXES
  roles / cube_only                    config/roles.py
  chain                                analysis/bounds.py
  steal_groups                         analysis/stealing.py
  compared                             validation/compare.py
  drain                                builders/base.py DRAIN_STAGES
"""
from __future__ import annotations

from ..config.stages import AIC, AIV0, AIV1, SharedAxis, StageVocabulary

#: 仓内两份 MegaMoE 实现 (a8w8_wave / layered) 共用的词汇表。
MEGAMOE = StageVocabulary(
    operator="megamoe",
    pipeline=("dispatch", "gmm1", "activation", "gmm2", "combine"),
    unit_of={
        "dispatch": "row",
        "gmm1": "tile",
        "activation": "tile",
        "gmm2": "tile",
        "combine": "tile",
    },
    #: dispatch 的单元是行, 一行一个事件既不是任何实现的做法也不是合理缺省,
    #: 所以这里的 0 不表示"整片", 而表示"没有覆盖, 用 tiling 算出的
    #: routeItemsPerBatch" (见 config/granularity)。
    default_items={"dispatch": 0},
    roles={
        "gmm1": AIC,
        "gmm2": AIC,
        "activation": AIV0,
        "shared_gmm1": AIC,
        "shared_gmm2": AIC,
        "shared_act": AIV0,
        # 通信与归约: 缺省放另一个向量角色, 与 ACT 分开
        "dispatch": AIV1,
        "dispatch_call": AIV1,
        "dispatch_recv": AIV1,
        "mask_scan": AIV1,
        "dispatch_local": AIV1,
        "combine": AIV1,
    },
    cube_only=("gmm1", "gmm2", "shared_gmm1", "shared_gmm2"),
    edges={
        ("dispatch", "gmm1"): SharedAxis(
            axis="token 行", chunk="m-group", segmentable=False,
            note=("一个 GMM1 事件的行落在一个 m-group 内, 它等的就是那一个组的就绪标记 "
                  "(builders/gmm1 的 dispatch_ready_event) —— 一次只吃一个块, "
                  "没有可分的段; 要更细得先改 tile_m 或 m-group 的划分")),
        ("gmm1", "activation"): SharedAxis(
            axis="N (GMM1 的输出列)", chunk="GMM1 tile", segmentable=False,
            note=("一个 ACT 对一个 GMM1 tile (1:1), 一次只吃一个块。ACT 内部按 "
                  "epilogue 行块再分是**粒度**而不是就绪 (builders/activation)")),
        ("activation", "gmm2"): SharedAxis(
            axis="K (GMM2 的归约轴 = GMM1 的输出列)", chunk="kL1 块", segmentable=True,
            consumed_by="builders/gmm2.add_gmm2_wave",
            note=("L0C 本来就沿 K 分块累加, 所以第 j 段只等覆盖自己那段 K 的 ACT。"
                  "块大小由 select_kl1 定 (options.gmm2_kl1 可覆盖)")),
        ("gmm2", "combine"): SharedAxis(
            axis="切片内的 tile", chunk="GMM2 tile", segmentable=False,
            packed_by_granularity=True,
            note=("这条边的共享轴就是 combine 的打包单元: 一个 combine 事件要写的那些行, "
                  "由它覆盖的 GMM2 tile 给齐。分段 = 少打包")),
    },
    #: dispatch_call 不列: 模型把它的开销折进了各段 dispatch 的 once_per_core,
    #: 模型侧根本没有这个事件, 条数比没有意义 (见 validation/compare)。
    compared=("gmm1", "activation", "gmm2", "combine", "dispatch"),
    drain=(
        ("aic", ("gmm1", "gmm2", "shared_gmm1", "shared_gmm2")),
        ("aiv0", ("activation",)),
        ("aiv1", ("dispatch_call", "dispatch", "combine",
                  "dispatch_recv", "dispatch_local", "mask_scan")),
    ),
    #: gmm1 -> act 是 AIC/AIV0 同核 (L0C->UB 硬件通路);
    #: gmm2 -> combine 是 AIC/AIV1 同核 (gmmToEpilogueFlag 按核索引)。
    steal_groups=(
        ("gmm1", ("activation",)),
        ("gmm2", ("combine",)),
    ),
    #: 缺省只搬 gmm1 组。gmm2 组是 opt-in: 实测两组同开会让贪心反复搬同一个 tile
    #: (hidden=18432/6 专家直接打爆引擎的注入上限), 且收益不单调
    #: (hidden=14336/6 专家 679.8 -> 687.3 反而变差)。两组同开要先解决 thrashing。
    #: 只搬 gmm1 不搬 gmm2 会把瓶颈推到 gmm2 的静态落位上, 实测反而制造新的
    #: work-conservation 违规 (hidden=9216/3 专家: 违规 0 -> 10) —— 两头都有代价,
    #: 缺省取的是实测更好的那头。
    default_steal_drivers=("gmm1",),
)
