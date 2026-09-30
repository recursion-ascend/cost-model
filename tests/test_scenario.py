"""统一入口: Scenario / 场景文件 / with_overrides.

等价性用 golden 指纹核对: 经 Scenario 进入与经原有入口进入, 事件级调度逐位相同.
"""
import json
from pathlib import Path

import pytest

import moe_cost_model as m
from golden_cases import CUBE_RATE, fingerprint, skewed_routing

STORED = json.loads((Path(__file__).parent / "golden" / "schedule_fingerprints.json")
                    .read_text(encoding="utf-8"))["cases"]
CAL = m.Calibration(cube_mac_per_us=CUBE_RATE)


def _skewed(**kw):
    return m.Scenario(
        workload=m.Workload(tokens=64, routing="explicit", counts=skewed_routing()),
        p1_override=2, p2_override=1, calibration=CAL, **kw)


def _w3(**kw):
    counts = [[[256] * 2 for _ in range(6)] for _ in range(2)]     # 与 golden 3 波用例同
    return m.Scenario(
        workload=m.Workload(tokens=512, routing="explicit", counts=counts),
        p1_override=2, p2_override=1, calibration=CAL, **kw)


# ---------------------------------------------------------------------------
# 与原有入口等价
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case, scenario", [
    ("mte_skewed_default", _skewed()),
    ("mte_skewed_lag1", _skewed(policy=m.InstancePolicy(gmm2_lag_waves=1))),
    ("mte_tile_n128", _skewed(kernel=m.KernelConfig(tile_n=128))),
    ("layered_skewed", _skewed(kernel=m.KernelConfig(topo_urma=True))),
    ("core_contiguous_block", _skewed(core_assignment="contiguous_block")),
    ("packing_balanced", _skewed(wave_packing="balanced_waves")),
    ("packing_longest_first", _skewed(wave_packing=m.LongestExpertFirst())),
    ("policy_priority_by_stage", _w3(scheduling_policy="priority_by_stage")),
    ("mte_3wave_lag2", _w3(policy=m.InstancePolicy(gmm2_lag_waves=2))),
    ("pipeline_split_channels", _skewed(
        default_channels=True,
        options=m.ModelOptions(pipeline=m.PipelineConstraints(
            queues=m.QueueDepths(mte_aic=2, cube=2, fix=2))))),
    ("fabric_channels_on", _skewed(options=m.ModelOptions(fabric_channels=True))),
])
def test_scenario_matches_golden(case, scenario):
    assert fingerprint(m.simulate(scenario)) == STORED[case]


def test_uniform_workload():
    """uniform 生成器: 2 rank × 6 专家, 每源 tokens×topk = 3072 行均分 → 每格 256 行."""
    wl = m.Workload(tokens=384, topk=8, world=2, local_experts=6, routing="uniform")
    assert wl.routing_counts() == tuple(tuple((256, 256) for _ in range(6)) for _ in range(2))


def test_stealing_by_name():
    sc = m.Scenario(
        workload=m.Workload(tokens=256, world=2, local_experts=4, topk=8, routing="explicit",
                            counts=[[[256] * 2 for _ in range(4)] for _ in range(2)]),
        p1_override=2, p2_override=1, calibration=CAL,
        restructure={"name": "idle_core_stealing", "min_pending": 2})
    assert fingerprint(m.simulate(sc)) == STORED["stealing_gmm1"]


# ---------------------------------------------------------------------------
# 路由生成器
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("routing", ["uniform", "cyclic", "random"])
def test_generated_routing_conserves_rows(routing):
    wl = m.Workload(tokens=50, topk=8, world=4, local_experts=16, routing=routing, seed=3)
    counts = wl.routing_counts()
    assert len(counts) == 4 and len(counts[0]) == 16 and len(counts[0][0]) == 4
    for src in range(4):
        sent = sum(counts[dst][e][src] for dst in range(4) for e in range(16))
        assert sent == 50 * 8
    if routing == "random":
        assert counts == wl.routing_counts(), "同种子必须可复现"


def test_uniform_remainder_is_spread():
    """除不尽时余数行不堆在前几个 rank: 72×6=432 行撒到 8×64=512 个专家."""
    wl = m.Workload(tokens=72, topk=6, world=8, local_experts=64, routing="uniform")
    counts = wl.routing_counts()
    per_dst = [sum(sum(row) for row in counts[dst]) for dst in range(8)]
    assert sum(per_dst) == 8 * 432 and max(per_dst) - min(per_dst) <= 8
    per_expert = [sum(row) for dst in counts for row in dst]
    assert max(per_expert) - min(per_expert) <= 2


def test_cyclic_matches_formula():
    wl = m.Workload(tokens=5, topk=2, world=2, local_experts=3, routing="cyclic")
    want = [[[0] * 2 for _ in range(3)] for _ in range(2)]
    for src in range(2):
        for t in range(5):
            for k in range(2):
                gid = (t + k + ((src + 1) % 2) * 3) % 6
                want[gid // 3][gid % 3][src] += 1
    assert wl.routing_counts() == tuple(tuple(tuple(r) for r in d) for d in want)


# ---------------------------------------------------------------------------
# 场景文件
# ---------------------------------------------------------------------------

TOML = """
name = "t"
p1_override = 2
p2_override = 1
core_assignment = "contiguous_block"

[workload]
tokens = 64
routing = "file"
file = "routing.json"

[calibration]
cube_mac_per_us = 2.7e7

[kernel]
tile_n = 128

[policy.wave_offsets]
dispatch = 1
gmm2 = 0

[scheduling_policy]
name = "priority_by_stage"
stage_order = ["dispatch", "gmm1", "act", "gmm2", "combine"]
"""


def test_load_toml(tmp_path):
    (tmp_path / "routing.json").write_text(
        json.dumps({"routing_counts": skewed_routing()}), encoding="utf-8")
    path = tmp_path / "s.toml"
    path.write_text(TOML, encoding="utf-8")
    sc = m.load_scenario(path)
    assert sc.kernel.tile_n == 128 and sc.core_assignment == "contiguous_block"
    assert sc.policy.wave_offsets == m.StageWaveOffsets(dispatch=1, gmm2=0)
    assert sc.workload.routing_counts() == tuple(
        tuple(tuple(r) for r in d) for d in skewed_routing())
    same = _skewed(kernel=m.KernelConfig(tile_n=128), core_assignment="contiguous_block",
                   policy=m.InstancePolicy(wave_offsets=m.StageWaveOffsets(1, 0)),
                   scheduling_policy=m.PriorityByStage())
    assert fingerprint(m.simulate(sc)) == fingerprint(m.simulate(same))


def test_example_scenario_runs():
    path = Path(__file__).resolve().parents[1] / "examples" / "scenario_basic.toml"
    res = m.simulate(m.load_scenario(path))
    assert fingerprint(res) == STORED["mte_uniform_bs64"]


@pytest.mark.parametrize("text, fragment", [
    ('[workload]\ntokens = 64\nworld = 4\nlocal_experts = 8\n[policy]\ngmm2_lag_wave = 1\n',
     "是否想写 'gmm2_lag_waves'"),
    ('[workload]\ntokens = 64\nworld = 4\nlocal_experts = 8\n[kernel]\ntile_n = "128"\n',
     "kernel.tile_n: 应为整数"),
    ('[workload]\ntokens = 64\nworld = 4\nlocal_experts = 8\n[kernel]\ntopo_urma = 1\n',
     "kernel.topo_urma: 应为布尔值"),
    ('wave_packing = "balanced"\n[workload]\ntokens = 64\nworld = 4\nlocal_experts = 8\n',
     "是否想写 'balanced_waves'"),
    ('[workload]\ntokens = 64\nrouting = "unifrom"\n', "是否想写 'uniform'"),
    ('[workload]\ntokens = 64\nrouting = "uniform"\n', "需要 world 与 local_experts"),
    ('h = 6144\n', "缺 [workload] 表"),
    ('[workload]\ntokens = 64\nworld = 4\nlocal_experts = 8\n[policy.wave_offsets]\ngmm2 = 1\n',
     "policy.wave_offsets"),
])
def test_bad_scenario_file_is_rejected(tmp_path, text, fragment):
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as err:
        m.load_scenario(path)
    assert fragment in str(err.value)
    assert "bad.toml" in str(err.value)


# ---------------------------------------------------------------------------
# 改旋钮
# ---------------------------------------------------------------------------

def test_with_overrides():
    base = _skewed()
    sc = base.with_overrides({
        "policy.gmm2_lag_waves": 1,
        "kernel.tile_n": 128,
        "policy.wave_offsets.gmm2": -1,             # 中间对象为 None: 以缺省值起步
        "options.pipeline.queues.mte_aic": 2,
        "calibration.dispatch.t_lat_remote_us": 3.0,
        "scheduling_policy": "critical_path_first",
    })
    assert sc.policy.gmm2_lag_waves == 1 and sc.kernel.tile_n == 128
    assert sc.policy.wave_offsets == m.StageWaveOffsets(dispatch=1, gmm2=-1)
    assert sc.options.pipeline.queues.mte_aic == 2
    assert sc.calibration.dispatch.t_lat_remote_us == 3.0
    assert base.kernel.tile_n == 256 and base.options.pipeline is None, "原场景不变"
    assert sc.to_dict(defaults=False) == {
        "workload": base.to_dict(defaults=False)["workload"],
        "p1_override": 2, "p2_override": 1,
        "kernel": {"tile_n": 128},
        "policy": {"gmm2_lag_waves": 1, "wave_offsets": {"gmm2": -1}},
        "options": {"pipeline": {"queues": {"mte_aic": 2}}},
        "calibration": {"cube_mac_per_us": CUBE_RATE, "dispatch": {"t_lat_remote_us": 3.0}},
        "scheduling_policy": "critical_path_first",
    }


@pytest.mark.parametrize("overrides, fragment", [
    ({"policy.gmm2_lag_wave": 2}, "是否想写 'gmm2_lag_waves'"),
    ({"kernel.tile_n": "128"}, "应为整数"),
    ({"kernel.tile_n.x": 1}, "不是表"),
    ({"policy.wave_offsets.gmm2": 1}, "gmm2 偏移必须"),
    ({"core_assignment": "round_robin"}, "未知策略"),
])
def test_bad_override_is_rejected(overrides, fragment):
    with pytest.raises(ValueError) as err:
        _skewed().with_overrides(overrides)
    assert fragment in str(err.value)


# ---------------------------------------------------------------------------
# Cube 速率必填
# ---------------------------------------------------------------------------

def test_cube_rate_is_optional():
    """cube_mac_per_us 缺省 0 = 不计计算项; 实测域内两个 GMM 都是权重载入绑定."""
    sc = m.Scenario(workload=m.Workload(tokens=64, world=4, local_experts=8))
    assert m.simulate(sc)["kernel_total_us"] > 0
    g = m.AnalyticalGmmCosts()
    assert g.gmm1_phases(256, 6144, 256)[1] == 0.0     # 不给速率 = 计算项为 0
    assert m.build_analytical_costs(
        h=6144, dispatch_mechanistic=m.DispatchMechanisticLatency()) is not None


def test_gmm_tile_formulas():
    """GMM1 = max(A流, B流, 计算); GMM2 = max(B流, 计算); 单缓冲 = 相加 + restart.

    载入取 max 而非相加, 由两点 m 扫定 (20260930 两个 run, 其余参数全同):
      bs36  m=72  实测单 tile 55.041 us
      bs8192 m=256 实测单 tile 53.810 us
    —— **与 m 无关** (斜率 -0.0067, A 流斜率是 +0.0987 us/行)。相加口径在 m=72
    只高 4.7% (A 流才占 12%), 到 m=256 就高 40.8%。
    """
    bw, rate = 50000.0, 1.0e7
    g = m.AnalyticalGmmCosts(bw_bytes_per_us=bw, cube_mac_per_us=rate)
    m_rows, k, cols, k2 = 256, 6144, 256, 2048
    a_flow = m_rows * k / bw
    b_flow1 = 2 * k * cols / bw
    b_flow2 = k2 * cols / bw
    compute1 = 2.0 * m_rows * cols * k / rate
    compute2 = m_rows * cols * k2 / rate
    assert g.gmm1_tile(m_rows, k, cols) == max(a_flow, b_flow1, compute1)
    assert g.gmm2_tile(m_rows, k2, cols) == max(b_flow2, compute2)
    assert g.gmm1_phases(m_rows, k, cols) == (max(a_flow, b_flow1), compute1)
    assert g.gmm2_phases(m_rows, k2, cols) == (b_flow2, compute2)
    # B 复用: 非首组的 tile 不付 B 流, 载入只剩 A 流
    assert g.gmm1_tile(m_rows, k, cols, False) == max(a_flow, compute1)
    # 计算快到让载入绑定
    fast = m.AnalyticalGmmCosts(bw_bytes_per_us=bw, cube_mac_per_us=1.0e12)
    assert fast.gmm1_tile(m_rows, k, cols) == max(a_flow, b_flow1)
    # 载入绑定时 tile 时长与 m 无关 —— 这正是两点 m 扫实测到的 (实测域 cube_rate=0)
    assert fast.gmm1_tile(64, k, cols) == fast.gmm1_tile(256, k, cols)
    # A 流超过 B 流 (m > 2*cols) 才翻转成 A 绑定
    assert fast.gmm1_tile(4 * cols, k, cols) > fast.gmm1_tile(2 * cols, k, cols)
    assert fast.gmm2_tile(m_rows, k2, cols) == b_flow2
    serial = m.AnalyticalGmmCosts(bw_bytes_per_us=bw, cube_mac_per_us=rate,
                                  l1_buf_num=1, tile_restart_us=0.5, l1_tile_k=256)
    assert serial.gmm1_tile(m_rows, k, cols) == max(a_flow, b_flow1) + compute1 + 24 * 0.5
    assert serial.gmm2_tile(m_rows, k2, cols) == b_flow2 + compute2 + 8 * 0.5


def test_combine_tile_splits_local_and_remote_rows():
    """COMBINE = GM→UB 读回 + 本卡行写 + 跨卡行写, 三段各按自己的带宽.

    内核 CombineTokens 对本窗每一行发一次 DataCopyPad, 目标是该行来源卡的窗口
    (Gmm2Aiv1EpilogueA8W4)。所以本窗代价取决于其中多少行要跨卡 —— 这一项由
    routing 精确给出, 不是按比例摊。
    """
    loc, rem = 100000.0, 5000.0
    cb = m.AnalyticalCombineCosts(bw_local_bytes_per_us=loc, bw_remote_bytes_per_us=rem)
    rows, n = 72, 256
    read = rows * (2 * n + cb.META_BYTES_PER_ROW) / loc      # ElementC = BF16
    row_bytes = 2 * n
    assert cb.write_bytes_per_elem == 2.0
    assert cb.read_us(rows, n) == read
    # 全本卡 / 全跨卡 两个端点
    assert cb.tile(rows, n, 0) == read + rows * row_bytes / loc
    assert cb.tile(rows, n, rows) == read + rows * row_bytes / rem
    # 混合: 逐行线性, 54 行跨卡 (4 卡均匀路由下每专家的实际值)
    assert cb.tile(rows, n, 54) == read + 18 * row_bytes / loc + 54 * row_bytes / rem
    # 跨卡贵 → 本地亲和度越高越便宜, 且是严格单调的
    us = [cb.tile(rows, n, r) for r in range(0, rows + 1, 9)]
    assert us == sorted(us) and us[-1] > 2.0 * us[0]
    with pytest.raises(ValueError):
        cb.tile(rows, n, rows + 1)
    with pytest.raises(ValueError):
        cb.tile(rows, n, -1)


def test_combine_quant_mode_only_changes_write_side():
    """QUANT 模式只改写侧宽度 (FP8 + 1/32 scale); 读回仍是 BF16 的 GMM2 输出."""
    kw = dict(bw_local_bytes_per_us=100000.0, bw_remote_bytes_per_us=5000.0)
    no_q = m.AnalyticalCombineCosts(combine_quant_mode=0, **kw)
    q = m.AnalyticalCombineCosts(combine_quant_mode=1, **kw)
    assert no_q.read_us(72, 256) == q.read_us(72, 256)
    assert q.write_bytes_per_elem == 1.0 + 1.0 / 32.0
    assert q.tile(72, 256, 54) < no_q.tile(72, 256, 54)
    with pytest.raises(ValueError, match="combine_quant_mode"):
        m.AnalyticalCombineCosts(combine_quant_mode=9)
