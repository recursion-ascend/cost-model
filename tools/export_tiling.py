#!/usr/bin/env python3
"""把 tiling_rank*.bin 的 kernel 真值导出成可入库的 JSON 旁置文件.

为什么需要这个
--------------
打点工件 (data/<run>/raw/) 体积大, .gitignore 把它整个排除了。但 tiling 真值
只是十来个整数 —— 它是 tests/test_guardrails.py 与 tools/eval_suite.py 赖以
核对 "场景文件声称的形状 == 跑出数据的 kernel 配置" 的唯一依据。bin 不入库,
这层护栏在 CI 与任何干净克隆里就全部 skip, 等于不存在。

本脚本在有 bin 的采集机上跑一次, 把同一批整数写成 data/<run>/raw/../
tiling_rank0.json (默认落在 bin 的同目录, 但 raw/ 被 gitignore, 所以用
--out 落到 run 根目录, 见下) 并入库; 之后 parse_tiling 在 bin 缺失时自动
回落到旁置文件, 护栏就恢复了。

用法
----
  # 单个 run: 导出到 run 根目录 (不在 gitignore 的 raw/ 里)
  python tools/export_tiling.py data/<run>/raw/tiling_rank0.bin

  # 全部 run 一次过
  python tools/export_tiling.py --all

  # 自定义输出
  python tools/export_tiling.py <bin> -o <某处>/tiling_rank0.json
"""
import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from moe_cost_model.config.pipeline import TILING_FIELDS, parse_tiling  # noqa: E402

DATA = PROJ / "data"


def default_out(bin_path: Path) -> Path:
    """旁置文件的落点.

    bin 在 <run>/raw/ 下, 而 /data/*/raw/ 整个被 gitignore; 所以默认落到
    <run>/ 根目录, 文件名不变 —— parse_tiling 的回落只看同目录同名 .json,
    故场景文件的 [tiling] path 也要指向 <run>/tiling_rank0.bin (不带 raw/)
    才能自动命中。--all 会一并打印需要改的 TOML 行。
    """
    if bin_path.parent.name == "raw":
        return bin_path.parent.parent / bin_path.with_suffix(".json").name
    return bin_path.with_suffix(".json")


def export_one(bin_path: Path, out: Path) -> dict:
    truth = parse_tiling(bin_path)
    assert set(truth) == set(TILING_FIELDS), "parse_tiling 字段与 TILING_FIELDS 不一致"
    payload = {k: int(truth[k]) for k in TILING_FIELDS}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    # 回读确认无损
    assert parse_tiling(out) == truth, f"{out}: 回读与 bin 不一致"
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin", nargs="?", type=Path, help="tiling_rank*.bin")
    ap.add_argument("-o", "--out", type=Path, help="输出 JSON 路径 (缺省见 default_out)")
    ap.add_argument("--all", action="store_true",
                    help=f"扫描 {DATA}/*/raw/tiling_rank0.bin 全部导出")
    args = ap.parse_args()

    if args.all:
        if args.bin or args.out:
            ap.error("--all 不能与 bin / --out 同用")
        bins = sorted(DATA.glob("*/raw/tiling_rank0.bin"))
        if not bins:
            print(f"{DATA} 下没有 */raw/tiling_rank0.bin。", file=sys.stderr)
            print("这个脚本要在有打点工件的采集机上跑 (raw/ 已 gitignore)。", file=sys.stderr)
            return 1
        for b in bins:
            out = default_out(b)
            export_one(b, out)
            print(f"{b.relative_to(PROJ)} -> {out.relative_to(PROJ)}")
        print("\n导出完毕。剩下一步:")
        print("  git add data/*/tiling_rank0.json   # 几百字节/个, 不在 gitignore 里")
        print("examples/*.toml 不用改: parse_tiling 在 raw/*.bin 缺失时会回落到上一级同名 .json。")
        return 0

    if not args.bin:
        ap.error("给一个 tiling_rank*.bin, 或用 --all")
    out = args.out or default_out(args.bin)
    export_one(args.bin, out)
    print(f"{args.bin} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
