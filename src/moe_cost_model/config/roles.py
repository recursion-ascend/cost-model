"""哪个 stage 跑在哪个执行角色上 (缺口 2).

一个核 (block) 上有三个执行角色:

    AIC     Cube 单元   矩阵乘
    AIV0    向量核 0
    AIV1    向量核 1

"哪个 stage 用哪个角色"原先是建图代码里写死的 f-string, 于是这些编排问不出来:
  * combine 交给 AIV0 (A8W8 下 AIV0 做完 ACT 就闲着 —— 实测两个向量核利用率都不到 6%)
  * dispatch 分给两个 AIV
  * A8W4 的角色互换 (激活搬到 AIV1, AIV0 做权重 W4->W8 解压)

注意这与晚绑定 (ModelOptions.late_bind_pools) 是两件事: 晚绑定管"同一角色池里哪个**核号**",
本模块管"哪个**角色**"。

### 哪些不许改

同核关系分两种, 只有一种是硬件强制的:

    GMM1 -> ACT      **物理**: 结果经 L0C->UB 的 Fixpipe 直给**配对**的 AIV
                     (CopyCL0c2GmOrUb(..., copyUbToV1)), 这条通路只在绑定对内存在。
                     所以 ACT 必须和 GMM1 在同一个核的向量角色上 —— 换哪个向量角色可以,
                     换到别的核不行。这条由 StageLink.colocated_by_hardware 表达。
    GMM2 -> combine  **不是物理**: GMM2 写 GM, combine 用 copyGM2UB 从 GM 读回,
                     配对只靠 gmmToEpilogueFlag[jobIndex] 这个索引约定。所以 combine
                     换角色、甚至换核都成立。

本模块只管"角色", 不碰上面那条物理共位 —— 它在 StageLink 里, 两边不重叠。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

#: 三个执行角色。AIC 是 Cube, 另两个是同一个核上的两个向量核。
AIC = "AIC"
AIV0 = "AIV0"
AIV1 = "AIV1"
ROLES = (AIC, AIV0, AIV1)
#: 向量角色 (可以互换的那两个)
VECTOR_ROLES = (AIV0, AIV1)

#: 只能跑在 Cube 上的 stage —— 矩阵乘没有别的去处, 这是物理
CUBE_ONLY_STAGES = ("gmm1", "gmm2", "shared_gmm1", "shared_gmm2")

#: 缺省分工 = 最少假设: 矩阵乘上 Cube, 其余各占一个向量角色。
#: ACT 与 GMM1 同核由 StageLink 的物理共位保证, 这里只说它用哪个向量角色。
DEFAULT_STAGE_ROLES: Dict[str, str] = {
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
}


@dataclass(frozen=True)
class RoleAssignment:
    """stage -> 执行角色 的映射. 不给的 stage 用缺省表.

    用法 (把 combine 挪到 AIV0, 让它和 ACT 抢同一个向量核, 换取 AIV1 专做 dispatch):

        RoleAssignment(overrides={"combine": "AIV0"})
    """

    overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for stage, role in self.overrides.items():
            if role not in ROLES:
                raise ValueError(
                    f"stage {stage!r} 的角色 {role!r} 不在 {ROLES} 里")
            if stage in CUBE_ONLY_STAGES and role != AIC:
                raise ValueError(
                    f"stage {stage!r} 是矩阵乘, 只能跑在 {AIC} 上 (物理), 收到 {role!r}")
            if stage not in CUBE_ONLY_STAGES and role == AIC:
                raise ValueError(
                    f"stage {stage!r} 不是矩阵乘, 放到 {AIC} 上没有物理依据; "
                    f"可选 {VECTOR_ROLES}")

    def role_of(self, stage: str) -> str:
        """这个 stage 用哪个角色; 未知 stage 报错而不是猜一个."""
        if stage in self.overrides:
            return self.overrides[stage]
        if stage in DEFAULT_STAGE_ROLES:
            return DEFAULT_STAGE_ROLES[stage]
        raise KeyError(
            f"未知 stage {stage!r}: 要么加进 config/roles.py 的缺省表, "
            f"要么在 RoleAssignment.overrides 里显式给角色")

    def resource(self, stage: str, core: int) -> str:
        """该 stage 在 core 号核上的资源名 (建图器就调这个, 不再拼 f-string)."""
        return f"{self.role_of(stage)}:{core}"

    def roles_in_use(self) -> Tuple[str, ...]:
        """这份分工实际用到的角色 (按 ROLES 的顺序), 供资源池声明用."""
        used = {self.role_of(s) for s in
                set(DEFAULT_STAGE_ROLES) | set(self.overrides)}
        return tuple(r for r in ROLES if r in used)

    def stages_on(self, role: str) -> Tuple[str, ...]:
        """哪些 stage 落在这个角色上 (排查"谁和谁抢核"时用)."""
        allst = set(DEFAULT_STAGE_ROLES) | set(self.overrides)
        return tuple(sorted(s for s in allst if self.role_of(s) == role))


#: 缺省分工 (单例, 省掉每次构造)
DEFAULT_ROLES = RoleAssignment()
