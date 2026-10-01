"""护栏: tiling 真值核对 / 信道尺度 / 路由守恒.

三条都来自实际踩过的坑 —— 见 src/moe_cost_model/guardrails.py 的模块说明。
"""
import dataclasses
import json
import struct
import sys
from pathlib import Path

import pytest

import moe_cost_model as m
from moe_cost_model.guardrails import (
    check_against_tiling, check_channels, check_routing_conservation)
from moe_cost_model.config.pipeline import (
    TILING_FIELDS, parse_tiling, resolve_tiling_path)
from moe_cost_model.scenario import TilingSource

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import export_tiling  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# 场景文件由 tiling 真值生成, 见 examples/*.toml
SCENARIO = ROOT / "examples" / "112575_bs36_noshared.toml"
RUN = ROOT / "data" / "20260930_154158_112575_bs36_h5120_i4608_k6_cyclic_noshared"
TILING = RUN / "raw" / "tiling_rank0.bin"


def _tiling_available() -> bool:
    """打点 bin (raw/, 已 gitignore) 或入库的 tiling_rank0.json 旁置文件任一就位即可.

    这几条护栏是项目自己防 "手抄参数没人核对" 的唯一一层; 只认 .bin 会让它们在
    CI 与任何干净克隆里全部 skip, 等于不存在。旁置文件由 tools/export_tiling.py
    导出并入库, 见 resolve_tiling_path。
    """
    try:
        resolve_tiling_path(TILING)
    except FileNotFoundError:
        return False
    return True


has_run = pytest.mark.skipif(
    not _tiling_available(),
    reason=f"tiling 真值未就位: {TILING} 与旁置 .json 都不存在 "
           f"(在采集机上跑 python tools/export_tiling.py --all 并入库)")


# ---------------------------------------------------------------- tiling 真值

@has_run
def test_example_scenario_matches_its_tiling():
    """examples/bs36_4rank.toml 声称的形状与它指向的 tiling 逐字段一致."""
    sc = m.load_scenario(str(SCENARIO))
    errors, _ = sc.check()
    assert errors == []


@has_run
def test_tiling_catches_hand_copied_p1():
    """p1/p2 是手填的: 用它们重算 mGroupsPerWave 必须对上 tiling, 否则报错.

    tiling 里没有 p1/p2 (主机侧策略), 但有派生量 mGroupsPerWave —— 这正好抓住
    "注释写了出处但没人核对" 这类错 (原 TOML 就是 `# 由 tiling 反解: 2` + 手填 2)。
    """
    sc = dataclasses.replace(m.load_scenario(str(SCENARIO)), p1_override=3,
                             tiling=TilingSource(path=str(TILING)))
    errors, _ = sc.check()
    assert any("mGroupsPerWave" in e for e in errors), errors
    with pytest.raises(ValueError, match="tiling 真值矛盾"):
        m.simulate(sc)


@has_run
def test_tiling_catches_wrong_shape():
    """hiddenDim 抄成 intermediate (4608 而非 2x4608) 立刻被抓."""
    sc = dataclasses.replace(m.load_scenario(str(SCENARIO)), hidden_dim=4608,
                             tiling=TilingSource(path=str(TILING)))
    errors, _ = sc.check()
    assert any("hiddenDim" in e for e in errors), errors


@has_run
def test_tiling_strict_false_downgrades_to_warning():
    sc = dataclasses.replace(m.load_scenario(str(SCENARIO)), p1_override=3,
                             tiling=TilingSource(path=str(TILING), strict=False))
    res = m.simulate(sc)            # 不报错
    assert any("mGroupsPerWave" in w for w in res["warnings"])


@has_run
def test_tiling_adopt_supplies_kernel_truths():
    """adopt: 行级软流水槽数与路由批大小取 tiling 真值, 不靠缺省常数碰巧相等."""
    sc = m.load_scenario(str(SCENARIO))
    assert int(sc.build_costs().dispatch_mechanistic.buffer_count) == 6
    assert sc.build_dispatch_layout().route_items_per_batch == 256
    # adopt=False 时退回缺省常数
    off = dataclasses.replace(sc, tiling=TilingSource(path=str(TILING), adopt=False))
    assert off.tiling_truth()["dispatchBufferCount"] == 6      # 真值仍读得到


def test_tiling_absent_is_fine():
    sc = m.Scenario(workload=m.Workload(tokens=64, world=4, local_experts=8))
    assert sc.tiling_truth() == {}
    assert sc.check()[0] == []          # 没有硬错


# ---------------------------------------------------------------- 信道尺度

def test_channel_guard_catches_contended_per_event_rate():
    """判据一: 单事件吃掉聚合一半以上 → 两个常数不同尺度, 会双重计费.

    片间信道占位就是这样: 聚合 BW_WINDOW=33000 是整卡值, 逐事件速率
    BW_REMOTE_GM=31000 是 28 核并发下反解的单核值 (已含平均争用) → 94%。
    """
    ch = m.Channel("fab_src:1", bw_total=33000.0, max_rate_per_event=31000.0)
    msgs = check_channels([ch], 28)
    assert len(msgs) == 1 and "94%" in msgs[0] and "双重计费" in msgs[0]


def test_channel_guard_flags_knife_edge_default():
    """判据二: 聚合恰好 = 每事件上限 x 核数 → 中性基线, 加消费者前必须先看见."""
    msgs = check_channels(
        m.default_channels(28, bw_l1_gm=m.BW_L1_GM, bw_scatter=m.BW_SCATTER), 28)
    assert {"gm_to_l1", "hbm_write"} == {msg.split()[1].rstrip(":") for msg in msgs}
    assert all("中性基线" in msg for msg in msgs)


def test_channel_guard_silent_on_healthy_config():
    """聚合是真实整卡值、单事件只占 1/核数 量级 → 无告警."""
    ch = m.Channel("hbm", bw_total=1_000_000.0, max_rate_per_event=50_000.0)
    assert check_channels([ch], 28) == []


def test_channel_guard_reaches_simulate_as_warning():
    """信道告警只警告不拦: default_channels 是有意的中性基线, 不是错."""
    sc = m.Scenario(workload=m.Workload(tokens=64, world=2, local_experts=4, topk=8),
                    p1_override=2, p2_override=1, default_channels=True)
    res = m.simulate(sc)
    assert any("中性基线" in w for w in res["warnings"])


# ---------------------------------------------------------------- 路由守恒

def test_routing_conservation():
    ok = [[[18] * 4 for _ in range(3)] for _ in range(4)]
    assert check_routing_conservation(ok, tokens=36, topk=6) == []
    bad = [[[18] * 4 for _ in range(3)] for _ in range(4)]
    bad[0][0][0] = 17
    msgs = check_routing_conservation(bad, tokens=36, topk=6)
    assert len(msgs) == 1 and "源 rank 0 发出 215 行" in msgs[0]
    neg = [[[-1] * 4 for _ in range(3)] for _ in range(4)]
    assert any("< 0" in msg for msg in check_routing_conservation(neg, 36, 6))


def test_routing_conservation_is_warning_only():
    """tokens 与 counts 在本模型里是两个独立输入 (tokens 只喂 p1 档位与 UNPERMUTE
    字节), 测试夹具故意让它们不一致 —— 所以只警告, 不拦。"""
    sc = m.Scenario(
        workload=m.Workload(tokens=64, topk=8, routing="explicit",
                            counts=[[[256] * 2 for _ in range(4)] for _ in range(2)]),
        p1_override=2, p2_override=1)
    errors, warnings = sc.check()
    assert errors == []
    assert any("!= tokens x topk" in w for w in warnings)
    assert m.simulate(sc)["kernel_total_us"] > 0     # 照跑


# ------------------------------------------------- tiling 旁置文件 (不需实测数据)

def _synth_tiling_bin() -> bytes:
    """最小合法 MegaMoeTilingData; 值无意义, 只为测 bin/json 两条解析路等价."""
    raw = bytearray(256)
    struct.pack_into("<10I", raw, 0, 3, 36, 5120, 9216, 4, 0, 0, 6, 28, 56)
    struct.pack_into("<I", raw, 64, 1)
    struct.pack_into("<Q", raw, 72, 0)
    struct.pack_into("<4i", raw, 80, 256, 1, 6, 0)
    struct.pack_into("<4i", raw, 96, 2, 1, 3, 0)
    struct.pack_into("<4i", raw, 112, 2, 1, 4, 0)
    struct.pack_into("<2i", raw, 132, 8, 2)
    struct.pack_into("<I", raw, 204, 2)
    return bytes(raw)


def test_tiling_sidecar_roundtrip_is_lossless(tmp_path):
    """export_tiling 导出的 JSON 与原 bin 解析出的真值逐字段相等."""
    b = tmp_path / "raw" / "tiling_rank0.bin"
    b.parent.mkdir()
    b.write_bytes(_synth_tiling_bin())
    from_bin = parse_tiling(b)
    out = export_tiling.default_out(b)
    export_tiling.export_one(b, out)
    assert out == tmp_path / "tiling_rank0.json"      # raw/ 被 gitignore, 落上一级
    assert parse_tiling(out) == from_bin
    assert set(from_bin) == set(TILING_FIELDS)


def test_bin_missing_falls_back_to_sidecar_one_level_up(tmp_path):
    """场景文件照旧指向 raw/*.bin; bin 没了也要命中入库的上一级旁置文件."""
    b = tmp_path / "raw" / "tiling_rank0.bin"
    b.parent.mkdir()
    b.write_bytes(_synth_tiling_bin())
    truth = parse_tiling(b)
    export_tiling.export_one(b, export_tiling.default_out(b))
    b.unlink()                                        # 模拟干净克隆: 只有旁置文件
    assert resolve_tiling_path(b) == tmp_path / "tiling_rank0.json"
    assert parse_tiling(b) == truth


def test_both_missing_error_names_the_exporter(tmp_path):
    """两者都没有时的报错必须说清怎么生成, 否则 CI 上只看到一句 FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="export_tiling"):
        parse_tiling(tmp_path / "raw" / "tiling_rank0.bin")


def test_sidecar_missing_field_is_rejected(tmp_path):
    """旁置文件缺字段必须报错, 不能静默返回半张真值表."""
    good = tmp_path / "tiling_rank0.json"
    b = tmp_path / "raw" / "tiling_rank0.bin"
    b.parent.mkdir()
    b.write_bytes(_synth_tiling_bin())
    export_tiling.export_one(b, good)
    data = json.loads(good.read_text())
    del data["mGroupsPerWave"]
    good.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mGroupsPerWave"):
        parse_tiling(good)
