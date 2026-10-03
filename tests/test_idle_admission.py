"""A: 空闲分解要看**准入**, 不只看依赖.

一个事件的前置跑完了, 它仍可能被别的物理约束挡住。那种等待和"依赖未完成"同类,
不该算成 work-conservation 违规。本文件锁住三类:

  ③ 计数信号量  UB:gmm1act 的槽 / QUEUE:mte_aic 的 L1 槽
  ⑦ 非核独占资源 DISPATCH_COMM (跨卡通道一次只许一个核用)
  并且"这个空闲核"自己的槽也要有余量 —— 核空着不等于槽空着
"""
import moe_cost_model as m
from moe_cost_model.analysis import idle_decomposition


def _run(local=3, hidden_dim=9216, **kw):
    W, PER = 5, 64
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(local)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(local)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=hidden_dim, aic_num=28,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency()),
        p1_override=1, p2_override=1, topk=6, **kw)["rank_results"][0]


def _avoid(rr, role):
    (rep,) = [v for k, v in rr["idle_decomposition"].items()
              if k.rstrip(":").endswith(role)]
    return rep.avoidable_idle_us


LB = ("AIC", "AIV1")


def test_dispatch_comm_wait_is_not_a_violation():
    """⑦ 跨卡通道串行化: 核空着、依赖齐了, 但通道被别的核占着 -> 物理上动不了.

    回归锚: 修之前这一项虚报 5814.8 核·us (AIV1)。
    """
    rr = _run(options=m.ModelOptions(serialize_dispatch_comm=True, late_bind_pools=LB),
              scheduling_policy=m.WorkConservingCriticalPath())
    assert _avoid(rr, "AIV1") < 1e-6
    # 这个编排确实把 dispatch 串起来了 (否则上面的断言没有意义)
    d = [e for e in rr["events"] if e.meta.get("stage") == "dispatch"
         and any("DISPATCH_COMM" in r for r in e.resources)]
    d.sort(key=lambda e: e.start_us)
    assert len(d) > 5
    assert all(b.start_us >= a.end_us - 1e-9 for a, b in zip(d, d[1:]))


def test_ub_slot_wait_is_not_a_violation():
    """③ UB 槽: 28 个核全空着却动不了 —— 每个核的槽都被它自己配对的 ACT 占着.

    回归锚: 修之前这一项虚报 262.8 核·us (AIC)。
    """
    rr = _run(options=m.ModelOptions(late_bind_pools=LB),
              scheduling_policy=m.WorkConservingCriticalPath())
    assert _avoid(rr, "AIC") < 1e-6
    # 确实存在"核空着但等槽"的时刻: 有事件的 actionable 被容量推到了 dep_ready 之后
    pushed = [e for e in rr["events"]
              if e.actionable_us > e.dependency_ready_us + 1e-9]
    assert pushed, "本形状应当有被容量推迟的事件, 否则这条测试没在测东西"


def test_without_capacities_it_degrades_to_an_upper_bound():
    """不传容量表就判不了"这个空闲核的槽有没有余量", 只能给上界.

    simulate 的 rank_results["idle_decomposition"] 是带容量表算的; 直接调
    idle_decomposition(evs, role) 不带容量, 会把等槽也算成违规。
    """
    # 要让"等槽"真的发生才能看出两种度量的差: 用 MEGAMOE_A8W8 那组编排 (per_core
    # 配速 + 按核预切 + 两段就绪), 缺省那组在本形状上压根没有等槽的时刻。
    rr = _run(6, 9216,
              options=m.MEGAMOE_A8W8.with_options(late_bind_pools=LB),
              scheduling_policy=m.WorkConservingCriticalPath())
    exact = _avoid(rr, "AIC")
    (loose,) = idle_decomposition(rr["events"], "AIC:").values()
    # 实测 9216/6: 不带容量 132.95, 带容量 0.0
    assert exact < 1e-6 < loose.avoidable_idle_us


def test_rule_holds_across_shapes_and_pacings():
    """硬护栏: 晚绑定 + 关键路径下, 三个角色池的违规必须恒为 0."""
    for pacing in ("per_core", "wave", "none"):
        for hd, local in ((9216, 3), (9216, 6), (18432, 6)):
            rr = _run(local, hd,
                      options=m.ModelOptions(late_bind_pools=LB, dispatch_pacing=pacing),
                      scheduling_policy=m.WorkConservingCriticalPath())
            for role in ("AIC", "AIV0", "AIV1"):
                assert _avoid(rr, role) < 1e-6, (pacing, hd, local, role)


def test_static_binding_still_reports_real_violations():
    """静态发牌的违规是真的 —— 修度量不能把它一起抹掉.

    静态发牌是某实现的分核方式 (缺省已是晚绑定), 所以这里显式给 ()。
    """
    rr = _run(6, 9216, options=m.ModelOptions(late_bind_pools=(), gmm2_k_segments=2))
    assert _avoid(rr, "AIC") > 1.0
