"""词汇表驱动的缺省建图器: 一条非 MegaMoE 的流水端到端跑通.

这里刻意**不用** MegaMoE 的词汇表: 生成器的价值就在于它不认识任何具体算子, 所以验证
也该用一条结构不同的流水 (四个 stage, 两条轴, 一条片上边)。
"""
import pytest

from moe_cost_model import MultiResourceScheduler
from moe_cost_model.config.links import StageLink
from moe_cost_model.config.stages import SharedAxis, StageVocabulary
from moe_cost_model.builders.generic import (PipelineSpec, WorkItem,
                                             lower_pipeline, onchip_capacities)

TOY = StageVocabulary(
    operator="toy_moe",
    pipeline=("recv", "gemm", "act", "send"),
    unit_of={"recv": "row", "gemm": "tile", "act": "tile", "send": "tile"},
    roles={"recv": "AIV1", "gemm": "AIC", "act": "AIV0", "send": "AIV1"},
    cube_only=("gemm",),
    edges={
        ("recv", "gemm"): SharedAxis("m", "row-block", False),
        ("gemm", "act"): SharedAxis("n", "n-tile", False),
        ("act", "send"): SharedAxis("n", "n-tile", False),
    },
    compared=("gemm", "act"),
    drain=(("aic", ("gemm",)), ("aiv0", ("act",)), ("aiv1", ("recv", "send"))),
    steal_groups=(("gemm", ("act",)),),
    default_steal_drivers=("gemm",),
    default_items={},
    chain=("recv", "gemm", "act", "send"),
)
SLICES, MTILES, NTILES, CORES = 2, 2, 4, 4
COST = {"recv": 1.0, "gemm": 4.0, "act": 1.5, "send": 2.0}


def _items(stage, shape):
    out = []
    for s in range(SLICES):
        for mg in range(MTILES):
            core = (s * MTILES + mg) % CORES
            if stage == "recv":
                out.append(WorkItem(slice_id=s, core=core, label=f"m{mg}",
                                    axes={"m": (mg * 128, mg * 128 + 128),
                                          "n": (0, NTILES * 256)}))
                continue
            for nt in range(NTILES):
                out.append(WorkItem(slice_id=s, core=core, label=f"m{mg}.n{nt}",
                                    axes={"m": (mg * 128, mg * 128 + 128),
                                          "n": (nt * 256, nt * 256 + 256)}))
    return out


class _Opts:
    """生成器只要两样: 粒度表与"角色+核号 -> 资源名"."""

    def __init__(self, granularity=None):
        self.granularity = granularity

    def role_resource(self, stage, core):
        return f"{TOY.roles[stage]}:{core}"


def _spec():
    return PipelineSpec(vocab=TOY, items=_items, cost=lambda st, it: COST[st])


class _Gran:
    def __init__(self, **per_stage):
        self._m = per_stage

    def items(self, stage):
        return self._m.get(stage, 1)


def test_one_event_per_work_item_at_finest_granularity():
    ev = lower_pipeline(_spec(), None, _Opts())
    per = {}
    for e in ev:
        per[e.meta["stage"]] = per.get(e.meta["stage"], 0) + 1
    assert per == {"recv": SLICES * MTILES, "gemm": SLICES * MTILES * NTILES,
                   "act": SLICES * MTILES * NTILES, "send": SLICES * MTILES * NTILES}


def test_edges_come_only_from_the_vocabulary_and_are_one_to_one():
    """gemm->act 与 act->send 在这条流水上是 1:1; recv->gemm 是 1:N (一行块喂 N 个 n-tile).

    只看共享轴会把"同一列、不同行"的生产者也连进来 —— 那条依赖不存在, 所以判据是
    两端共有的每条轴都相交。
    """
    ev = {e.name: e for e in lower_pipeline(_spec(), None, _Opts())}
    for name, e in ev.items():
        if e.meta["stage"] == "act":
            assert len(e.deps) == 1 and e.deps[0].startswith("gemm."), (name, e.deps)
        if e.meta["stage"] == "send":
            assert len(e.deps) == 1 and e.deps[0].startswith("act."), (name, e.deps)
        if e.meta["stage"] == "gemm":
            assert len(e.deps) == 1 and e.deps[0].startswith("recv."), (name, e.deps)
        if e.meta["stage"] == "recv":
            assert e.deps == ()


def test_graph_is_acyclic():
    ev = lower_pipeline(_spec(), None, _Opts())
    by = {e.name: e for e in ev}
    state = {}

    def walk(n):
        if state.get(n) == 2:
            return
        assert state.get(n) != 1, f"环: {n}"
        state[n] = 1
        for d in by[n].deps:
            walk(d)
        state[n] = 2

    for e in ev:
        walk(e.name)


def test_hardware_colocation_is_enforced_by_the_schedule():
    links = (StageLink("gemm", "act", location="onchip", depth=2,
                       colocated_by_hardware=True),)
    ev = lower_pipeline(_spec(), None, _Opts(), links=links)
    caps = onchip_capacities(_spec(), _Opts(), CORES, links=links)
    _, placed = MultiResourceScheduler().schedule(ev, capacities=caps)
    where = {p.name: p.resources[0].split(":")[1] for p in placed if p.resources}
    for e in ev:
        if e.meta["stage"] == "act":
            assert where[e.name] == where[e.deps[0]], e.name


def test_onchip_depth_is_a_live_constraint():
    """片上深度来自 StageLink, 不是生成器自己编的: 深度变, 时间就变."""
    seen = {}
    for depth in (1, 2, 8):
        links = (StageLink("gemm", "act", location="onchip", depth=depth,
                           colocated_by_hardware=True),)
        ev = lower_pipeline(_spec(), None, _Opts(), links=links)
        caps = onchip_capacities(_spec(), _Opts(), CORES, links=links)
        seen[depth], _ = MultiResourceScheduler().schedule(ev, capacities=caps)
    assert seen[1] > seen[2] >= seen[8], seen


def test_granularity_packs_along_the_shared_axis():
    """粒度沿打包轴合并, 区间取并集; 其余轴不同的项不会被并进来."""
    ev = lower_pipeline(_spec(), None, _Opts(_Gran(gemm=2)))
    gemm = [e for e in ev if e.meta["stage"] == "gemm"]
    assert len(gemm) == SLICES * MTILES * NTILES // 2
    assert all(e.meta["members"] == 2 for e in gemm)
    # 合并后一个 act 依赖的那个 gemm 事件覆盖两个 n-tile, 所以两个 act 指向同一个
    acts = [e for e in ev if e.meta["stage"] == "act"]
    assert len({e.deps[0] for e in acts}) == len(gemm)


def test_zero_granularity_is_refused():
    with pytest.raises(ValueError, match="只接受 >= 1"):
        lower_pipeline(_spec(), None, _Opts(_Gran(gemm=0)))


def test_missing_axis_is_refused_rather_than_guessed():
    def bad_items(stage, shape):
        if stage == "act":
            return [WorkItem(slice_id=0, core=0, label="x", axes={"m": (0, 128)})]
        return _items(stage, shape)

    spec = PipelineSpec(vocab=TOY, items=bad_items, cost=lambda st, it: 1.0)
    with pytest.raises(ValueError, match="共享轴"):
        lower_pipeline(spec, None, _Opts())
