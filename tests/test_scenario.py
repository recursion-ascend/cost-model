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
            queues=m.QueueDepths(mte_aic=2, cube=2, fix=2),
            phases=m.PhaseRates(cube_mac_per_us=2.7e7))))),
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

def test_cube_rate_is_required():
    sc = m.Scenario(workload=m.Workload(tokens=64, world=4, local_experts=8))
    with pytest.raises(ValueError, match="cube_mac_per_us"):
        m.simulate(sc)
    with pytest.raises(TypeError):
        m.AnalyticalGmmCosts()
    with pytest.raises(TypeError):
        m.build_analytical_costs(h=6144, dispatch_mechanistic=m.DispatchMechanisticLatency())
    with pytest.raises(ValueError, match="cube_mac_per_us"):
        m.AnalyticalGmmCosts(cube_mac_per_us=0.0)


def test_gmm_tile_formulas():
    """GMM1 = max(A流, 计算); GMM2 = 纯计算; 串行 = 相加 + restart; B 流不计."""
    bw, rate = 50000.0, 1.0e7
    g = m.AnalyticalGmmCosts(bw_bytes_per_us=bw, cube_mac_per_us=rate)
    m_rows, k, cols = 256, 6144, 256
    a_flow = m_rows * k / bw
    compute1 = 2.0 * m_rows * cols * k / rate
    assert g.gmm1_tile(m_rows, k, cols) == max(a_flow, compute1)
    assert g.gmm2_tile(m_rows, 2048, cols) == m_rows * cols * 2048 / rate
    # 计算快到让 A 流绑定
    fast = m.AnalyticalGmmCosts(bw_bytes_per_us=bw, cube_mac_per_us=1.0e12)
    assert fast.gmm1_tile(m_rows, k, cols) == a_flow
    serial = m.AnalyticalGmmCosts(bw_bytes_per_us=bw, cube_mac_per_us=rate,
                                  l1_buf_num=1, tile_restart_us=0.5, l1_tile_k=256)
    assert serial.gmm1_tile(m_rows, k, cols) == a_flow + compute1 + 24 * 0.5
    assert serial.gmm2_tile(m_rows, 2048, cols) == m_rows * cols * 2048 / rate + 8 * 0.5
