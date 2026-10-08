"""计数信号量: 哪些是真约束, 哪些是同一条约束的第二份抄本.

判据与理由见 scheduler/normalize.py。本文件钉三件事:
  1. 定理在调度器层成立 (自取自还 + 共同独占 -> 容量多大都一样);
  2. 判据第 2 条不能省 (不持资源的自取自还 token 是真约束);
  3. 真图上规范化之后一个空约束都不剩, 而真约束一个都没少。
"""
import pytest

import moe_cost_model as m
from moe_cost_model.scheduler.engine import MultiResourceScheduler
from moe_cost_model.scheduler.events import Event
from moe_cost_model.scheduler.normalize import (inert_semaphores,
                                                prune_inert_semaphores)
from golden_cases import run_api, uniform_routing


def _sched(events, caps):
    _, s = MultiResourceScheduler().schedule(events, capacities=caps)
    return [(e.name, e.start_us, e.end_us, e.actionable_us) for e in s]


# ------------------------------------------------------- 定理: 容量多大都一样

def test_self_paired_token_under_an_exclusive_resource_cannot_bind():
    """自取自还 + 两个持有者都独占同一个核资源 -> 容量 1 与 2 排程相同.

    核的独占已经把"同核在途数 <= 1"表达完了, 计数器是第二份抄本。
    """
    def graph():
        return [Event("A", ("AIC:c0",), 10.0, acquires=(("Q", 1),), releases=(("Q", 1),)),
                Event("B", ("AIC:c0",), 10.0, acquires=(("Q", 1),), releases=(("Q", 1),))]
    one, two = _sched(graph(), {"Q": 1}), _sched(graph(), {"Q": 2})
    assert [(n, s, e) for n, s, e, _ in one] == [(n, s, e) for n, s, e, _ in two]
    assert [s for _, s, _, _ in one] == [0.0, 10.0]


def test_but_it_still_moves_actionable_us():
    """同一个空约束却改了 actionable_us —— 这就是它必须删掉的理由.

    actionable_us 的契约是"依赖齐 + 信号量可准入", **不含**"自己要的资源空出来"
    (那一关是 analysis/idle.py 判 work-conservation 的依据)。空约束把后者从后门
    塞了回来: 下面 B 的 actionable 被推到 10.0, 于是本该算 avoidable 的空闲会被
    算成 forced。
    """
    def graph():
        return [Event("A", ("AIC:c0",), 10.0, acquires=(("Q", 1),), releases=(("Q", 1),)),
                Event("B", ("AIC:c0",), 10.0, acquires=(("Q", 1),), releases=(("Q", 1),))]
    (_, _, _, a1), (_, _, _, b1) = _sched(graph(), {"Q": 1})
    (_, _, _, a2), (_, _, _, b2) = _sched(graph(), {"Q": 2})
    assert (a1, b1) == (0.0, 10.0)      # 容量 1: B 看起来 10.0 才"能动"
    assert (a2, b2) == (0.0, 0.0)       # 容量 2: B 从一开始就能动 (事实如此)


def test_a_self_paired_token_without_a_resource_really_binds():
    """判据第 2 条不能省: .ld 相位 (resources=()) 的 MTE2 是真约束.

    它是"一个 AI Core 只有一条 MTE2"在图里的唯一表达; 只按"自取自还"删会把它
    一起删掉, 载入就能无限并发 (2026-10-05 修过的带宽下界穿透)。
    """
    def graph():
        return [Event("A", (), 10.0, acquires=(("MTE2", 1),), releases=(("MTE2", 1),)),
                Event("B", (), 10.0, acquires=(("MTE2", 1),), releases=(("MTE2", 1),))]
    one, two = _sched(graph(), {"MTE2": 1}), _sched(graph(), {"MTE2": 2})
    assert [s for _, s, _, _ in one] == [0.0, 10.0]     # 串行
    assert [s for _, s, _, _ in two] == [0.0, 0.0]      # 并行 —— 真的咬


# ------------------------------------------------------- 判据的分类

def test_criterion_classifies_the_four_shapes():
    aic0 = ("AIC:c0",)
    cases = {
        # 自取自还 + 共同独占 -> 空约束
        "Q": [Event("A", aic0, 1.0, acquires=(("Q", 1),), releases=(("Q", 1),)),
              Event("B", aic0, 1.0, acquires=(("Q", 1),), releases=(("Q", 1),))],
        # 自取自还但不持资源 -> 真约束
        "MTE2": [Event("L", (), 1.0, acquires=(("MTE2", 1),), releases=(("MTE2", 1),))],
        # 跨事件 (取与还不是同一个事件) -> 真约束
        "UB": [Event("G", aic0, 1.0, acquires=(("UB", 1),)),
               Event("T", ("AIV0:c0",), 1.0, releases=(("UB", 1),))],
        # 持有者不共享资源 -> 真约束 (一个持核、一个不持核)
        "MIX": [Event("A", aic0, 1.0, acquires=(("MIX", 1),), releases=(("MIX", 1),)),
                Event("M", (), 1.0, acquires=(("MIX", 1),), releases=(("MIX", 1),))],
    }
    assert inert_semaphores(cases["Q"]) == {"Q"}
    for key in ("MTE2", "UB", "MIX"):
        assert inert_semaphores(cases[key]) == set(), key


def test_pool_placeholders_are_not_exclusive():
    """"AIC:*" 的两个持有者可以落不同成员, 所以它不算"共同独占"."""
    evs = [Event("A", ("AIC:*",), 1.0, acquires=(("Q", 1),), releases=(("Q", 1),)),
           Event("B", ("AIC:*",), 1.0, acquires=(("Q", 1),), releases=(("Q", 1),))]
    assert inert_semaphores(evs) == set()


def test_prune_returns_what_it_removed_and_leaves_the_rest():
    evs = [Event("A", ("AIC:c0",), 1.0, acquires=(("Q", 1), ("UB", 1)),
                 releases=(("Q", 1),)),
           Event("T", ("AIV0:c0",), 1.0, releases=(("UB", 1),))]
    assert prune_inert_semaphores(evs) == {"Q"}
    assert evs[0].acquires == (("UB", 1),) and evs[0].releases == ()
    assert evs[1].releases == (("UB", 1),)


# ------------------------------------------------------- 真图上的不变量

Q = m.QueueDepths
PIPE = m.PipelineConstraints
CONFIGS = {
    "缺省 (晚绑定)": m.ModelOptions(),
    "静态钉核": m.ModelOptions(late_bind_pools=()),
    "拆相位": m.ModelOptions(pipeline=PIPE(queues=Q(mte_aic=2, cube=2))),
    "拆相位 + 静态": m.ModelOptions(late_bind_pools=(),
                                    pipeline=PIPE(queues=Q(mte_aic=2, cube=2))),
    "拆相位 + AIV 读相位": m.ModelOptions(pipeline=PIPE(
        queues=Q(mte_aic=2), phases=m.PhaseRates(act_load_bw_bytes_per_us=5e4))),
    "ACT 不物化": m.ModelOptions(links=(
        m.StageLink("gmm1", "activation", location="onchip", depth=1,
                    colocated_by_hardware=True),
        m.StageLink("activation", "gmm2", location="onchip"))),
}


def _events(options, aic_num=28):
    return run_api(uniform_routing(2, 4, 128), 128, topk=8, aic_num=aic_num,
                   options=options)["rank_results"][0]["events"]


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_no_inert_semaphore_survives_in_any_configuration(name):
    """规范化之后, 排好的图里一个"起不了约束"的 token 都不该剩."""
    left = inert_semaphores(_events(CONFIGS[name]))
    assert left == set(), f"{name}: 这些 token 是第二份抄本, 规范化漏了: {sorted(left)}"


def test_the_real_constraints_are_all_still_there():
    """删空约束不能把真约束一起删掉."""
    def kinds(evs):
        return {t.split(":c")[0].split(".", 1)[-1]
                for e in evs for t, _ in (e.acquires + e.releases)}
    # 缺省: GMM1->ACT 的 UB 槽 (跨事件: GMM1 取、配对 ACT 还)
    assert "UB:gmm1act" in kinds(_events(CONFIGS["缺省 (晚绑定)"]))
    # 拆相位: 每核一条 MTE2 / 一条 FixPipe + L1 缓冲槽 (跨事件)
    got = kinds(_events(CONFIGS["拆相位"]))
    for tok in ("MTE2", "FIXPIPE", "QUEUE:mte_aic", "QUEUE:fix", "UB:gmm1act"):
        assert tok in got, (tok, sorted(got))
    # QUEUE:cube 是空约束 (.cb 相位独占 AIC 核资源), 规范化后不该在图里
    assert "QUEUE:cube" not in got


def test_the_same_name_can_be_inert_on_one_path_and_real_on_another():
    """Q:vec0 在持核事件上是空约束, 在**拆相位的 AIV 路径**上却是真约束.

    那条路径 (pipeline_expand._expand_aiv 的 need_split 分支) 把 token 变成跨事件
    持有: .ld 相位取、主事件还。所以判据必须逐 token 看全部持有者, 不能按名字一刀切
    —— 这是 2026-10-08 改动里差点漏掉的分支: 当时顺手删了 Q:* 的容量声明,
    这条路径立刻报 "acquires unknown capacity resource"。
    """
    options = m.ModelOptions(pipeline=PIPE(
        queues=Q(mte_aic=2), phases=m.PhaseRates(act_load_bw_bytes_per_us=5e4)))
    evs = _events(options)
    kinds = {t.split(":c")[0].split(".", 1)[-1] for e in evs for t, _ in
             (e.acquires + e.releases)}
    assert "Q:vec0" in kinds          # 这条路径上它是真约束, 必须留着
    assert "QUEUE:vec" not in kinds   # 而主事件自取自还的那个仍是空约束
    assert inert_semaphores(evs) == set()


def test_bandwidth_lower_bound_still_holds_after_pruning():
    """最该怕的回归: 删错 token -> 载入无限并发 -> 墙钟低于带宽下界."""
    for name, options in sorted(CONFIGS.items()):
        rr = run_api(uniform_routing(2, 4, 128), 128, topk=8, aic_num=28,
                     options=options)["rank_results"][0]
        b = rr["bounds"]
        assert b["violation"] is None, (name, b)
        assert rr["dag_end_us"] >= b["lower_us"] - 1e-9, (name, b)


# ------------------------------------------------------- 尺子: 它量到的东西变了

def _avoid(rr, role):
    (rep,) = [v for k, v in rr["idle_decomposition"].items()
              if k.rstrip(":").endswith(role)]
    return rep.avoidable_idle_us


def test_pruning_fixed_the_work_conservation_yardstick():
    """空约束删掉之后, 静态钉核不再把可避免空闲报成 forced.

    实测 (uniform 2x4x128, aic=28): 删之前 R0.AIC 的 avoidable 报 0.0,
    删之后 1454.67 核·µs —— **而排程逐位未变** (墙钟 289.065231 两边相同)。
    所以这不是模型变慢了, 是度量不再被一条空约束遮住。
    晚绑定那一侧本来就没有这些 token (旧代码只在晚绑定路径删), 所以仍是 0。
    """
    static = run_api(uniform_routing(2, 4, 128), 128, topk=8, aic_num=28,
                     options=m.ModelOptions(late_bind_pools=()))
    late = run_api(uniform_routing(2, 4, 128), 128, topk=8, aic_num=28,
                   options=m.ModelOptions(late_bind_pools=("AIC", "AIV1")))
    assert static["kernel_total_us"] == pytest.approx(289.065231, rel=1e-6)
    assert late["kernel_total_us"] == pytest.approx(258.087441, rel=1e-6)
    assert _avoid(static["rank_results"][0], "AIC") > 1400.0
    assert _avoid(late["rank_results"][0], "AIC") < 1e-6
