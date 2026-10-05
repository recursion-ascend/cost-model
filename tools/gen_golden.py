#!/usr/bin/env python3
"""生成/核对 golden 调度指纹 (tests/golden/schedule_fingerprints.json).

用法:
  python tools/gen_golden.py                 # 重新生成全部用例
  python tools/gen_golden.py --check         # 只核对, 不写文件 (退出码 1 = 有差异)
  python tools/gen_golden.py --check --explain  # 逐用例说**哪一类**指纹变了
  python tools/gen_golden.py --only mte_     # 名称前缀过滤 (写入时保留其余用例)

仅在有意改变模型行为时重新生成; 重构与提速必须通过 --check.

--explain 存在的理由: 指纹里混着两类东西, 它们对"行为有没有变"的含义完全不同。
  **行为类** schedule_sha256 / total_us / dag_end_us / events / wave_count /
            stage_busy_us / traffic_bytes / bounds / critical_path_len
  **申报类** provenance_summary / provenance_sha256 —— 它们哈希的是**全部常数的名字与
            取值** (config/provenance.py 走 vars(hardware)、vars(policy)、costs 与
            kernel 的数值字段)。所以**仅仅新增一个常数或一个数值字段**, 在事件一个比特
            都没动的情况下, 也会让全部用例的 provenance_sha256 变化。
重构 (分层、改 IR) 时要能一句话证明"只动了申报、没动行为", 靠的就是这个区分;
没有它, 一次正当的重构与一次真实的回归在 --check 的输出里长得一模一样。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "tests"))

from golden_cases import CASES, fingerprint  # noqa: E402

SNAPSHOT = PROJ / "tests" / "golden" / "schedule_fingerprints.json"


#: 指纹字段分两类。行为类变了 = 模型算出来的东西变了; 申报类变了 = 常数的声明变了。
BEHAVIOUR_FIELDS = frozenset({
    "kernel_total_us", "slowest_rank", "total_us", "dag_end_us", "events",
    "wave_count", "stage_busy_us", "critical_path_len", "traffic_bytes", "bounds",
    "schedule_sha256",
})
DECLARATION_FIELDS = frozenset({"provenance_summary", "provenance_sha256"})


def _flatten(fp, prefix=""):
    """指纹摊平成 路径 -> 值, 便于逐字段比对 (ranks 按 rank 号展开)."""
    out = {}
    for key, val in fp.items():
        if key == "ranks":
            for rank in val:
                for k2, v2 in rank.items():
                    if k2 == "rank":
                        continue
                    out[f"r{rank['rank']}.{k2}"] = v2
        else:
            out[prefix + key] = val
    return out


def _diff_fields(old_fp, new_fp):
    """两个指纹里不同的字段路径 (已摊平)."""
    a, b = _flatten(old_fp), _flatten(new_fp)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _classify(field: str) -> str:
    base = field.split(".", 1)[-1]
    if base in DECLARATION_FIELDS:
        return "申报"
    if base in BEHAVIOUR_FIELDS:
        return "行为"
    return "未分类"


def _print_explanation(detail) -> None:
    print("\n差异字段 (按用例):")
    kinds = {}
    for name in sorted(detail):
        fields = detail[name]
        tags = sorted({_classify(f) for f in fields})
        kinds.setdefault(tuple(tags), []).append(name)
        print(f"  {name:32s} {', '.join(fields)}")
    print("\n按类别:")
    for tags, names in sorted(kinds.items()):
        print(f"  {'+'.join(tags):12s} {len(names):3d} 用例"
              + (f"  例: {names[0]}" if names else ""))
    only_decl = kinds.get(("申报",), [])
    if only_decl and len(only_decl) == sum(len(v) for v in kinds.values()):
        print("\n全部差异都只在**申报类** —— 常数的声明变了, 模型算出来的东西没变。")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="只核对, 不写文件")
    ap.add_argument("--explain", action="store_true",
                    help="逐用例列出差异字段, 并按行为类/申报类分组统计")
    ap.add_argument("--only", default="", help="用例名前缀过滤")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    stored = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}
    cases = dict(stored.get("cases", {}))
    detail: Dict[str, List[str]] = {}
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
            if args.explain:
                detail[name] = _diff_fields(cases[name], fp)
        n_ev = sum(r["events"] for r in fp["ranks"])
        print(f"{name:32s} {fp['kernel_total_us']:12.3f} us  {n_ev:6d} events  "
              f"{dt:7.2f} s  {status}", flush=True)
        if not args.check:
            cases[name] = fp
    # 全量跑时清掉已删用例的残留指纹 —— 否则删掉一个用例后它的指纹永远留在
    # 快照里, --check 也发现不了 (它只核对 CASES 里有的名字)。
    stale = [] if args.only else sorted(set(cases) - set(CASES))
    if stale:
        print(f"已删用例的残留指纹: {stale}")
        if not args.check:
            for n in stale:
                del cases[n]
    print(f"total {time.perf_counter() - t_all:.1f} s, {len(names)} cases, "
          f"{len(diffs)} diff")
    if args.explain and detail:
        _print_explanation(detail)
    if args.check:
        missing = [n for n in names if n not in stored.get("cases", {})]
        if missing:
            print(f"snapshot 缺用例: {missing}")
        return 1 if diffs or missing or stale else 0
    SNAPSHOT.write_text(json.dumps({
        "comment": "调度指纹快照: 每事件起止/等待/关键父事件的 sha256. "
                   "重构与提速必须逐位一致; 由 tools/gen_golden.py 生成.",
        "cases": {n: cases[n] for n in sorted(cases)},
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
