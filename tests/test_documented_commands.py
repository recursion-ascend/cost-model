"""docs 与 README 里写的命令必须真能跑: 脚本存在, 参数被 argparse 接受.

2026-10-06 核对时 docs/calibration_runs.md 写的是
`compare_measured.py --scenario <toml>`, 而那个脚本收两个**位置**参数, 没有 --scenario ——
照着文档敲会直接报错。这类错一行正则就能挡住。
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md"))

CMD = re.compile(r"(python3? (?:tools|examples)/[\w./]+\.py[^\n#`)]*)")

#: 只有长这样的 token 才算参数 —— 否则 README 里的 HTML 注释结尾 `-->` 会被当成参数
FLAG = re.compile(r"^--[a-z][a-z0-9-]*$")


def _commands():
    for doc in DOCS:
        text = (ROOT / doc).read_text(encoding="utf-8")
        for raw in CMD.findall(text):
            yield doc, raw.strip()


def test_every_documented_script_exists():
    missing = [(d, c) for d, c in _commands()
               if not (ROOT / c.split()[1]).is_file()]
    assert not missing, f"文档里的脚本不存在: {missing}"


def test_every_documented_flag_is_accepted():
    bad = []
    for doc, cmd in _commands():
        parts = cmd.split()
        script = (ROOT / parts[1])
        if not script.is_file():
            continue
        source = script.read_text(encoding="utf-8")
        for token in parts[2:]:
            if not token.startswith("--"):
                continue
            flag = token.split("=")[0]
            if not FLAG.match(flag):
                continue
            if flag not in source:
                bad.append(f"{doc}: {cmd} -> {flag}")
    assert not bad, "文档里的参数脚本不接受: " + "; ".join(bad)


def test_there_are_commands_to_check():
    assert len(list(_commands())) > 15
