"""golden 调度指纹: 重构与提速后每个事件的起止/等待/关键父事件必须逐位不变.

用例定义见 golden_cases.py; 快照由 tools/gen_golden.py 生成.
有意改变模型行为时重新生成快照, 并在提交说明里写明哪些用例变了、为什么.
"""
import json
from pathlib import Path

import pytest

from golden_cases import CASES, fingerprint

SNAPSHOT = Path(__file__).parent / "golden" / "schedule_fingerprints.json"
STORED = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["cases"]


def test_snapshot_covers_all_cases():
    assert sorted(STORED) == sorted(CASES), "用例表与快照不同步: 运行 tools/gen_golden.py"


@pytest.mark.parametrize("name", sorted(CASES))
def test_schedule_fingerprint(name):
    got = fingerprint(CASES[name]())
    want = STORED[name]
    # 先比可读字段, 失败信息能直接指出哪个 rank / stage 变了
    assert got["kernel_total_us"] == want["kernel_total_us"]
    for g, w in zip(got["ranks"], want["ranks"]):
        assert g["events"] == w["events"], f"rank {g['rank']} 事件数变化"
        assert g["stage_busy_us"] == w["stage_busy_us"], f"rank {g['rank']} stage busy 变化"
    assert got == want, "总时长与 stage busy 未变, 但事件级调度 (起止/归因) 有差异"
