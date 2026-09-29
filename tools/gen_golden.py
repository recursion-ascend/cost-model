#!/usr/bin/env python3
"""生成/核对 golden 调度指纹 (tests/golden/schedule_fingerprints.json).

用法:
  python tools/gen_golden.py                 # 重新生成全部用例
  python tools/gen_golden.py --check         # 只核对, 不写文件 (退出码 1 = 有差异)
  python tools/gen_golden.py --only mte_     # 名称前缀过滤 (写入时保留其余用例)

仅在有意改变模型行为时重新生成; 重构与提速必须通过 --check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "tests"))

from golden_cases import CASES, fingerprint  # noqa: E402

SNAPSHOT = PROJ / "tests" / "golden" / "schedule_fingerprints.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="只核对, 不写文件")
    ap.add_argument("--only", default="", help="用例名前缀过滤")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    stored = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}
    cases = dict(stored.get("cases", {}))
    names = [n for n in CASES if n.startswith(args.only)]
    diffs = []
    t_all = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        fp = fingerprint(CASES[name]())
        dt = time.perf_counter() - t0
        status = "new"
        if name in cases:
            status = "same" if cases[name] == fp else "DIFF"
        if status == "DIFF":
            diffs.append(name)
        n_ev = sum(r["events"] for r in fp["ranks"])
        print(f"{name:32s} {fp['kernel_total_us']:12.3f} us  {n_ev:6d} events  "
              f"{dt:7.2f} s  {status}", flush=True)
        if not args.check:
            cases[name] = fp
    print(f"total {time.perf_counter() - t_all:.1f} s, {len(names)} cases, "
          f"{len(diffs)} diff")
    if args.check:
        missing = [n for n in names if n not in stored.get("cases", {})]
        if missing:
            print(f"snapshot 缺用例: {missing}")
        return 1 if diffs or missing else 0
    SNAPSHOT.write_text(json.dumps({
        "comment": "调度指纹快照: 每事件起止/等待/关键父事件的 sha256. "
                   "重构与提速必须逐位一致; 由 tools/gen_golden.py 生成.",
        "cases": {n: cases[n] for n in sorted(cases)},
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
