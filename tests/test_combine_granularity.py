"""combine 的粒度 (缺口 11 的后一半).

两种都是合理编排:
  "per_tile"   与每个 GMM2 tile 1:1 配对、紧跟其后 —— 延迟低、能与计算交错
  "per_expert" 一个专家切片一个事件、等该切片全部 GMM2 段做完 —— 攒批

粒度与**角色** (ModelOptions.roles) 正交: 前者管"一个事件覆盖多少工作", 后者管"跑在哪"。
参考实现把粒度与量化模板参数绑在一起, 那是它的耦合 (见 docs 缺口 11)。
"""
import pytest

import moe_cost_model as m


def _run(gran, late=("AIC", "AIV1"), W=5, LOCAL=3, PER=64, hd=9216, h=5120, aic=28):
    rc = [[[0 if s == d else PER for s in range(W)] for _ in range(LOCAL)] for d in range(W)]
    tok = sum(rc[d][e][1] for d in range(W) for e in range(LOCAL)) // 6
    return m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=h, hidden_dim=hd, aic_num=aic,
        topk=6, p1_override=1, p2_override=1,
        costs=m.build_analytical_costs(
            h=h, dispatch_mechanistic=m.DispatchMechanisticLatency(),
            cube_mac_per_us=m.cube_mac_per_us("fp8")),
        options=m.ModelOptions(combine_granularity=gran, late_bind_pools=late),
    )["rank_results"][0]


def _combines(rr):
    return [e for e in rr["events"] if e.meta.get("stage") == "combine"]


def test_per_tile_is_the_default():
    assert m.ModelOptions().combine_granularity == "per_tile"


def test_per_expert_emits_one_event_per_expert_slice():
    tile, expert = _run("per_tile"), _run("per_expert")
    assert len(_combines(expert)) == 3                 # 3 个本地专家, 每个 1 个切片
    assert len(_combines(tile)) > len(_combines(expert))
    assert all(e.meta.get("granularity") == "per_expert" for e in _combines(expert))


def test_per_expert_covers_the_whole_width_and_all_rows():
    """一个事件覆盖整片: 列到 h, 行是切片全部行; 行数守恒."""
    for e in _combines(_run("per_expert")):
        assert e.meta["logical_n"] == 5120
        assert e.meta["m_rows"] == sum(e.meta["rows_by_dst"])


def test_per_expert_waits_for_its_whole_slice():
    """它等**自己那个切片**全部 GMM2 段做完 —— 这就是攒批的代价.

    注意是按切片而不是按波: 专家 0 的 combine 不等专家 2 的 GMM2。
    """
    expert = _run("per_expert")
    g2_end = {}
    for e in expert["events"]:
        if e.meta.get("stage") == "gmm2":
            key = (e.meta["expert"], e.meta["slice"])
            g2_end[key] = max(g2_end.get(key, 0.0), e.end_us)
    assert g2_end
    for cb in _combines(expert):
        key = (cb.meta["expert"], cb.meta["slice"])
        assert cb.start_us >= g2_end[key] - 1e-9, (cb.name, cb.start_us, g2_end[key])


def test_per_tile_interleaves_with_compute():
    """per_tile 的 combine 与 GMM2 交错: 第一个 combine 远早于最后一个 GMM2 结束."""
    tile = _run("per_tile")
    last_g2 = max(e.end_us for e in tile["events"] if e.meta.get("stage") == "gmm2")
    first_cb = min(e.start_us for e in _combines(tile))
    assert first_cb < last_g2


def test_batching_saves_metadata_bytes_but_loses_the_pipelining():
    """攒批省的是元数据 (每行读一次 vs 每个 n-tile 读一遍), 丢的是流水交错.

    实测 9216/3 专家/28 核: combine 忙碌合计 305.34 -> 303.86 (省 0.5%),
    墙钟 315.63 -> 411.83 (**+30.5%**)。省下的字节远不抵丢掉的交错。
    """
    tile, expert = _run("per_tile"), _run("per_expert")
    busy_t = sum(e.end_us - e.start_us for e in _combines(tile))
    busy_e = sum(e.end_us - e.start_us for e in _combines(expert))
    assert busy_e < busy_t                              # 字节确实省了
    assert (busy_t - busy_e) / busy_t < 0.02            # 但只省一点
    assert expert["total_us"] > tile["total_us"] * 1.2  # 墙钟明显更差


def test_the_model_cannot_show_per_expert_s_main_upside():
    """诚实声明: per_expert 的主要好处 (写侧落点跨度更可控) 模型**算不出来**.

    跨度不在模型里 (docs 缺口 10), 所以这里只看得见它的代价 (丢交错) 与一点字节收益。
    等缺口 10 补上, 这个对比才有意义 —— 现在不要据此下"攒批没用"的结论。
    """
    tile, expert = _run("per_tile"), _run("per_expert")
    # 两种粒度写出的总字节一致 (同样的行 x 同样的列), 差别只在元数据与事件切分
    def wrote(rr):
        return sum(e.meta["m_rows"] * e.meta["logical_n"] for e in _combines(rr))
    assert wrote(tile) == wrote(expert)


@pytest.mark.parametrize("gran", ["per_tile", "per_expert"])
def test_invariant_holds_under_both(gran):
    """两种粒度下"有就绪的活就不空闲"都要成立 (缺省晚绑定)."""
    rr = _run(gran)
    for role in ("AIC", "AIV0", "AIV1"):
        hit = [v for k, v in rr["idle_decomposition"].items() if k.endswith(role)]
        if hit:
            assert hit[0].avoidable_idle_us < 1e-6, role


def test_granularity_is_orthogonal_to_role():
    """粒度与角色正交: per_expert 也能换角色."""
    rc = [[[0 if s == d else 64 for s in range(5)] for _ in range(3)] for d in range(5)]
    tok = sum(rc[d][e][1] for d in range(5) for e in range(3)) // 6
    rr = m.simulate_routing_counts(
        routing_counts=rc, token_num_per_rank=tok, h=5120, hidden_dim=9216, aic_num=28,
        topk=6, p1_override=1, p2_override=1,
        costs=m.build_analytical_costs(
            h=5120, dispatch_mechanistic=m.DispatchMechanisticLatency(),
            cube_mac_per_us=m.cube_mac_per_us("fp8")),
        options=m.ModelOptions(combine_granularity="per_expert",
                               roles=m.RoleAssignment({"combine": "AIV0"})),
    )["rank_results"][0]
    cb = [e for e in rr["events"] if e.meta.get("stage") == "combine"]
    assert cb and all(":" in e.resources[0] for e in cb)
    assert all(e.resources[0].split(":")[0].endswith("AIV0") for e in cb)
