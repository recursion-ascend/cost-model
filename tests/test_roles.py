"""哪个 stage 跑在哪个执行角色上 (原缺口 2).

原先资源名是建图代码里写死的 f-string ("AIV1:7"), 于是"换个角色干这件事"问不出来。
现在是 ModelOptions.roles (config/roles.py 的 RoleAssignment)。

本文件钉三件事: 物理边界不许被改掉 / 换角色真的改事件落点 / 以及**它值多少钱**
(答案是在 A8W8 主路径上几乎不值钱, 原因见 test_moving_work_between_vector_roles_... )。
"""
import pytest

import moe_cost_model as m
from routing import conserving_tokens

R = m.RoleAssignment


def _run(opts, W=5, LOCAL=3, PER=64, hd=9216, h=5120, aic=28, topk=6):
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(LOCAL)] for d in range(W)]
    # token 数由路由矩阵定 (守恒): 原先写死 "// 6" 与 topk=6 配对, 在 LOCAL=8/PER=256 这组
    # 参数下每源 8192 行不是 6 的整数倍 —— 整除丢掉的 2 行让输入不守恒。
    tok = conserving_tokens(rc, topk)
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=h, hidden_dim=hd, aic_num=aic,
        topk=topk, p1_override=1, p2_override=1,
        costs=m.build_analytical_costs(
            h=h, dispatch_mechanistic=m.DispatchMechanisticLatency(),
            cube_mac_per_us=m.cube_mac_per_us("fp8")),
        options=opts)["rank_results"][0]


def _busy(rr, role):
    hit = [v for k, v in rr["idle_decomposition"].items() if k.endswith(role)]
    return hit[0].busy_us if hit else 0.0


def _roles_of(rr, stage):
    return {e.resources[0].split(":")[0].split(".")[-1]
            for e in rr["events"] if e.meta.get("stage") == stage and e.resources}


# ---------------------------------------------------------------------------
# 物理边界: 矩阵乘只能在 Cube 上
# ---------------------------------------------------------------------------

def test_matmul_stages_cannot_leave_the_cube():
    for stage in ("gmm1", "gmm2", "shared_gmm1", "shared_gmm2"):
        with pytest.raises(ValueError, match="只能跑在 AIC"):
            R({stage: "AIV0"})


def test_vector_stages_cannot_be_put_on_the_cube():
    with pytest.raises(ValueError, match="没有物理依据"):
        R({"combine": "AIC"})


def test_unknown_role_and_unknown_stage_are_refused():
    with pytest.raises(ValueError, match="不在"):
        R({"combine": "AIV9"})
    with pytest.raises(KeyError, match="未知 stage"):
        R().role_of("not_a_stage")


# ---------------------------------------------------------------------------
# 换角色确实改事件落点
# ---------------------------------------------------------------------------

def test_default_assignment_is_unchanged():
    rr = _run(m.ModelOptions())
    assert _roles_of(rr, "gmm1") == {"AIC"}
    assert _roles_of(rr, "activation") == {"AIV0"}
    assert _roles_of(rr, "combine") == {"AIV1"}
    assert _roles_of(rr, "dispatch") == {"AIV1"}


def test_combine_can_move_to_the_other_vector_role():
    """GMM2 -> combine 的同核不是物理约束 (过 GM), 所以 combine 可以换角色."""
    rr = _run(m.ModelOptions(roles=R({"combine": "AIV0"})))
    assert _roles_of(rr, "combine") == {"AIV0"}
    assert _roles_of(rr, "activation") == {"AIV0"}      # 现在两者抢同一个向量核
    assert _busy(rr, "AIV1") < _busy(_run(m.ModelOptions()), "AIV1")


def test_act_can_move_too_a8w4_style_swap():
    """A8W4 的角色互换: 激活搬到另一个向量角色, 通信搬过来."""
    swap = R({"activation": "AIV1", "combine": "AIV0",
              "dispatch": "AIV0", "dispatch_call": "AIV0"})
    rr = _run(m.ModelOptions(roles=swap))
    assert _roles_of(rr, "activation") == {"AIV1"}
    assert _roles_of(rr, "combine") == {"AIV0"}


def test_late_binding_follows_the_assignment():
    """GMM1 入池隐含"跑 ACT 的那个角色"入池 —— 不再写死 AIV0.

    物理依据: ACT 必须与它的 GMM1 同核 (L0C->UB 的 Fixpipe 只在绑定对内)。
    """
    swap = R({"activation": "AIV1", "combine": "AIV0",
              "dispatch": "AIV0", "dispatch_call": "AIV0"})
    rr = _run(m.ModelOptions(roles=swap, late_bind_pools=("AIC",)))
    by = {e.name: e for e in rr["events"]}
    acts = [e for e in rr["events"] if e.meta.get("stage") == "activation"]
    assert acts
    for a in acts:
        g = by[a.name.replace(".act.", ".gmm1.")]
        assert a.resources[0].rsplit(":", 1)[1] == g.resources[0].rsplit(":", 1)[1]


# ---------------------------------------------------------------------------
# 它值多少钱
# ---------------------------------------------------------------------------

def test_moving_work_between_vector_roles_buys_nothing_here():
    """A8W8 主路径上换角色不改墙钟 —— 两个向量核都远没吃满 (各约 6%).

    这是**结论**而不是缺数据: 关键路径在 AIC 上, 把向量工作在两个向量角色之间
    挪来挪去不碰它。所以"AIV0 闲着"这件事不能靠重分角色回收 —— 能挪的工作本来就不在
    关键路径上, 而 AIC 上的矩阵乘没有别处可去 (Cube 独有)。要用上空闲的向量核得给它们
    **新的**工作 (例如跨核 K-split 的归约), 那是另一个问题。
    """
    base = _run(m.ModelOptions())
    moved = _run(m.ModelOptions(roles=R({"combine": "AIV0"})))
    assert moved["total_us"] == pytest.approx(base["total_us"])
    assert _busy(base, "AIC") == pytest.approx(_busy(moved, "AIC"))
    # 工作量守恒: 只是换了角色
    assert (_busy(base, "AIV0") + _busy(base, "AIV1")
            == pytest.approx(_busy(moved, "AIV0") + _busy(moved, "AIV1")))


def test_collapsing_two_vector_roles_into_one_does_cost():
    """把全部向量工作挤到一个角色上会变慢 —— 旋钮是活的, 不是装饰.

    实测 2048/8专家/2核: 3861.56 -> 3924.70 (+1.6%)。两个向量角色对称
    (挤到 AIV0 与挤到 AIV1 同值), 这也是个合理性校验。
    """
    # topk=8: 每源 4 卡 x 8 专家 x 256 = 8192 行 = 1024 x top-8 (top-6 不整除)
    kw = dict(LOCAL=8, PER=256, hd=2048, h=2048, aic=2, topk=8)
    base = _run(m.ModelOptions(), **kw)["total_us"]
    on0 = _run(m.ModelOptions(roles=R({"combine": "AIV0", "dispatch": "AIV0",
                                       "dispatch_call": "AIV0"})), **kw)["total_us"]
    on1 = _run(m.ModelOptions(roles=R({"activation": "AIV1"})), **kw)["total_us"]
    assert on0 > base
    assert on0 == pytest.approx(on1)          # 两个向量角色对称


def test_assignment_reports_who_shares_a_role():
    """排查"谁和谁抢核"要能直接问出来."""
    assert "activation" in R().stages_on("AIV0")
    assert "combine" in R().stages_on("AIV1")
    assert set(R({"combine": "AIV0"}).stages_on("AIV0")) >= {"activation", "combine"}
    assert R().roles_in_use() == ("AIC", "AIV0", "AIV1")


def test_queue_token_follows_the_role_not_a_hardcoded_engine():
    """引擎队列令牌必须跟着角色走, 不能写死.

    2026-10-06 之前建图器各自拼 f-string (activation 写 "Q:vec0:c{core}",
    combine/dispatch 写 "Q:aiv1:c{core}"), 而资源名已经走 role_resource()。于是
    roles={"combine": "AIV0"} 会让一个事件**占着 AIV0 的核、扣着 AIV1 的队列** ——
    两个名字说的是两个不同的引擎。

    今天这三个令牌都是空约束 (容量恒 1, 事件同时独占该核), 所以那个不一致不改时长; 但它让
    按令牌名分型的分析看错引擎 (ir 的 TokenKind 就按前缀分), 而且引擎队列一旦有了真约束,
    它就变成时长错误。
    """
    from moe_cost_model.config.roles import QUEUE_TOKENS, RoleAssignment

    default = RoleAssignment()
    assert default.queue_token("activation", 3) == "Q:vec0:c3"
    assert default.queue_token("combine", 3) == "Q:aiv1:c3"
    assert default.queue_token("gmm1", 3) == "Q:aic:c3"
    # 挪角色 -> 令牌跟着挪, 与 resource() 指向同一个引擎
    moved = RoleAssignment(overrides={"combine": "AIV0"})
    assert moved.resource("combine", 3) == "AIV0:3"
    assert moved.queue_token("combine", 3) == "Q:vec0:c3"
    # 映射表覆盖全部三个角色 (model.py 声明容量时用同一组名字)
    assert set(QUEUE_TOKENS.values()) == {"Q:aic", "Q:vec0", "Q:aiv1"}


def test_moving_combine_moves_its_queue_token_in_the_real_graph():
    """真实建图里也要一致: 把 combine 挪到 AIV0, 它扣的队列也必须是 AIV0 的那条.

    必须用**静态发牌** (late_bind_pools=()): 缺省的晚绑定会把这类"自取自还"的按核令牌
    整个去掉 (model.py: 事件同时独占该核, 同核在途数恒 <= 1, 令牌是空约束), 所以晚绑定下
    根本看不到它 —— 当年那个不一致也就只在静态发牌时能观察到。
    """
    from moe_cost_model.ir import TokenKind, classify_resource, classify_token

    res = _run(m.ModelOptions(roles=R({"combine": "AIV0"}), late_bind_pools=()),
               LOCAL=3, PER=64, aic=4)
    combines = [e for e in res["events"] if e.meta.get("stage") == "combine"]
    assert combines
    for ev in combines:
        engines = {classify_resource(r).engine.value for r in ev.resources
                   if classify_resource(r)}
        assert engines == {"AIV0"}, engines
        queues = [classify_token(t) for t, _ in ev.acquires]
        queues = [q for q in queues if q and q.kind is TokenKind.ENGINE_QUEUE]
        assert queues, "combine 应当持有一个引擎队列令牌"
        for q in queues:
            assert q.raw.split(".", 1)[-1].startswith("Q:vec0:"), q.raw
