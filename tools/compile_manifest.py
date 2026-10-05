#!/usr/bin/env python3
"""编译清单: 从 C++/CMake 抽编译期参数, 并与 Python 侧对账.

用法:
  python tools/compile_manifest.py                 # 打印清单 + 对账结果
  python tools/compile_manifest.py --json          # 机器可读 (入库/比对用)
  python tools/compile_manifest.py --check         # 只对账, 有失配则退出码 1

为什么需要它 —— 编译期参数在 kernel 源码与 Python 里各写一份, 没有机制让它们对上, 而
这个 bug 类已经真实发生过: KernelConfig.swizzle_direction 缺省 1、注释声称 kernel 用
<3, 1>, 而 common/mega_moe_gmm_common.h:33 写的是 BlockSchedulerSwizzle<3, 0> ——
m 组 > 1 时模型的 GMM tile 遍历顺序相对 kernel 是 M/N 转置的, 实测墙钟差 +5.0%。
一个字符串注释不会报错, 一次对账会。

本工具**不改任何常数**: 它只报告。有意不同的项登记在
implementations/manifest.INTENTIONAL 里并写明理由 (例如 combine 元数据缺省取算法下界
16B 而 kernel 搬满 32B), 其余一律算失配。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from moe_cost_model.implementations.manifest import (compare,  # noqa: E402
                                                     extract)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="输出机器可读清单")
    ap.add_argument("--check", action="store_true", help="只对账; 有失配退出码 1")
    ap.add_argument("--root", default=str(PROJ), help="仓库根 (默认本仓库)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    man = extract(args.root)
    if args.json:
        print(man.to_json())
        return 1 if man.errors else 0

    if not args.check:
        print(f"# 编译清单: {len(man.entries)} 项 (源: {args.root})\n")
        print(f"{'名字':40s}{'取值':14s}{'整数':10s}{'种类':14s}出处")
        for name in sorted(man.entries):
            e = man.entries[name]
            got = man.value(name)
            print(f"{name:40s}{str(e.value)[:13]:14s}"
                  f"{('' if got is None else str(got)):10s}{e.kind:14s}{e.source}")
        if man.harness:
            print("\n# 标定语料那份实例化 (模板实参决定标定值的适用编译点)")
            for key, val in man.harness.items():
                print(f"  {key:22s} {val}")

    report = compare(man)
    print(f"\n# 对账: 核了 {report['checked']} 项, 失配 {len(report['mismatch'])} 项")
    for item in report["mismatch"]:
        print(f"  失配 {item['python']} = {item['python_value']} "
              f"但 {item['manifest']} = {item['manifest_value']} ({item['source']})")
    for note in report["skipped"]:
        print(f"  跳过 {note}")
    for err in report["errors"]:
        print(f"  抽取错误 {err}")
    if not report["mismatch"] and not report["errors"]:
        print("  Python 侧与仓内源码一致。")
    return 1 if (report["mismatch"] or report["errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
