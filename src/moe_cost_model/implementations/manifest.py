"""从 C++ / CMake 源码抽出**编译清单**, 并与 Python 侧的取值对账.

要解决的问题: 编译期参数在两个地方各写一份 —— kernel 的源码里一份, Python 的
`config/hardware.py` 与 `costs.py` 里一份 —— 而没有任何机制让它们对上。后果不是理论上的:

  * `KernelConfig.swizzle_direction` 缺省 1, 注释声称 kernel 用 `<3, 1>`, 而
    `common/mega_moe_gmm_common.h:33` 写的是 `BlockSchedulerSwizzle<3, 0>`。m 组 > 1 时
    模型的 GMM tile 遍历顺序相对 kernel 是 M/N 转置的, 实测影响墙钟 +5.0% (2026-10-05 修)。
  * `URMA_FLAG_WINDOW_TOKENS = 256` 把 kernel 的 `DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY`
    (= `MEGAMOE_TILE_M`) 抄成了字面量, 于是改 tile_m 不会带动它。
  * tiling struct 的字节偏移是手抄的 (`config/pipeline.py`), 文件头自己写明"结构变了必须
    手动同步"。

所以这里**不生成代码、不改常数**, 只做两件事:
  extract()  读源码, 给出清单 (值 + 出处 file:line)
  compare()  与 Python 侧逐项对账, 列出不一致

为什么不自动写回 Python: 一是写回会让 `provenance_sha256` 在全部 golden 上变化 (出处摘要
按常数名+取值哈希), 二是有些差异是**故意**的 (例如 `combine_meta_bytes_per_row` 缺省 16 是
"四个具名字段"的算法下界口径, kernel 的 32 是 `META_INFO_SIZE` 搬满八个槽)。故意的差异要
显式登记在 INTENTIONAL 里并写明理由, 其余一律算失配 —— 这样"忘了同步"与"有意不同"分得开。

可靠解析的范围 (见 docs: 只认固定写法, 认不出就报, 不猜):
  CMake      set(MEGAMOE_X <默认> CACHE STRING "<说明>") / option(X "<说明>" ON|OFF)
  宏缺省     #ifndef X \\n #define X <值>  这个两行惯用法
  结构常数   constexpr <类型> NAME = <字面量或简单算式>;
  模板实例化 kernel.cpp 里 MegaMoeA8W8Wave<...> 的实参表
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: 仓内 kernel 的根目录 (相对仓库根). 下面的路径都由它拼出来 —— 原先它只是个声明,
#: 而每条路径各自又写了一遍 "mega_moe/", 换目录要改十几处而这个常数改了不生效。
KERNEL_ROOT = "mega_moe"


def _k(*parts: str) -> str:
    """KERNEL_ROOT 下的一条路径 (posix 分隔符, 与清单里记录的写法一致)."""
    return "/".join((KERNEL_ROOT,) + parts)


_ARCH35 = ("op_kernel", "arch35")
_COMMON = _ARCH35 + ("common",)

#: CMake 里定义 MEGAMOE_* 的唯一文件 (其余 CMakeLists 只做 glob)
CMAKE_FILE = _k("include", "CMakeLists.txt")

#: 用两行 #ifndef/#define 惯用法给宏缺省的文件
MACRO_FILES = (
    _k(*_COMMON, "mega_moe_constants.h"),
    _k(*_COMMON, "mega_moe_gmm_common.h"),
    _k(*_ARCH35, "mega_moe_apt.cpp"),
    _k("include", "kernel.cpp"),
    _k("include", "host.cpp"),
    _k("op_host", "op_tiling", "arch35", "mega_moe_tiling.cpp"),
)

#: 要抽的结构常数 (白名单): 名字 -> 它在哪个文件。白名单而不是全抽, 因为"模型用到哪些"
#: 是一个断言: 名字消失或写法变了必须报错, 不能静默少一项。
STRUCTURAL = {
    "L1_TILE_M_256": _k(*_COMMON, "mega_moe_constants.h"),
    "L1_TILE_M_128": _k(*_COMMON, "mega_moe_constants.h"),
    "L1_TILE_N": _k(*_COMMON, "mega_moe_constants.h"),
    "ACTIVATION_N_HALF": _k(*_COMMON, "mega_moe_constants.h"),
    "META_INFO_SIZE": _k(*_COMMON, "mega_moe_constants.h"),
    "MXFP_DIVISOR_SIZE": _k(*_COMMON, "mega_moe_constants.h"),
    "MXFP_MULTI_BASE_SIZE": _k(*_COMMON, "mega_moe_constants.h"),
    "LAYERED_USABLE_UB_BYTES": _k(*_COMMON, "mega_moe_constants.h"),
    "L1_TILE_K": _k(*_COMMON, "mega_moe_gmm_common.h"),
    "GMM2_LAG_MIN_TOKEN_NUM": "mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h",
}

#: 调度器模板实例化: using BlockScheduler = ... BlockSchedulerSwizzle<Offset, Direction>
_SWIZZLE = re.compile(r"BlockSchedulerSwizzle<\s*(\d+)\s*,\s*(\d+)\s*>")
_CMAKE_SET = re.compile(
    r'^\s*set\(\s*(MEGAMOE_\w+)\s+(\S+)\s+CACHE\s+STRING\s+"([^"]*)"\s*\)', re.M)
_CMAKE_OPTION = re.compile(r'^\s*option\(\s*(\w+)\s+"([^"]*)"\s+(ON|OFF)\s*\)', re.M)
_MACRO_DEFAULT = re.compile(
    r"^#ifndef\s+(\w+)\s*\n#define\s+\1\s+([^\n/]+?)\s*$", re.M)
_CONSTEXPR = re.compile(
    r"^\s*(?:static\s+)?constexpr\s+\w+\s+(\w+)\s*=\s*([^;]+);", re.M)
#: 测量用的那份实例化 (include/kernel.cpp): 模板实参表决定了标定语料的编译点
_HARNESS_INSTANCE = re.compile(
    r"MegaMoeImpl::(MegaMoe\w+)<([^>]*)>\s*op\s*;", re.S)


@dataclass(frozen=True)
class ManifestEntry:
    """清单里的一项: 名字 / 取值 / 出处 / 种类."""

    name: str
    value: object
    source: str                 # file:line
    kind: str                   # cmake_cache | cmake_option | macro | constexpr | template
    doc: str = ""

    @property
    def as_int(self) -> Optional[int]:
        text = str(self.value).strip().rstrip("Uu")
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        # 简单算式: 只算 + - * / 与整数 (例如 248U * 1024U)
        expr = re.sub(r"([0-9]+)[Uu]+", r"\1", str(self.value))
        if re.fullmatch(r"[0-9+\-*/() ]+", expr):
            try:
                return int(eval(expr, {"__builtins__": {}}, {}))   # noqa: S307
            except Exception:
                return None
        return None


@dataclass
class CompileManifest:
    """一次抽取的结果."""

    entries: Dict[str, ManifestEntry] = field(default_factory=dict)
    harness: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def value(self, name: str, _depth: int = 0) -> Optional[int]:
        """整数取值. 右值引用别的常数时 (L1_TILE_M_256 = MEGAMOE_TILE_M) 递归解析一层.

        kernel 里这种转写很常见, 而它正是"派生关系不能丢"的地方: Python 把
        URMA_FLAG_WINDOW_TOKENS 抄成字面量 256, 于是改 tile_m 带不动它。清单保留派生,
        对账才能发现这类脱钩。
        """
        got = self.entries.get(name)
        if got is None:
            return None
        direct = got.as_int
        if direct is not None:
            return direct
        if _depth > 4:
            return None
        expr = re.sub(r"([0-9]+)[Uu]+", r"\1", str(got.value))
        names = set(re.findall(r"[A-Za-z_]\w*", expr))
        for ref in names:
            sub = self.value(ref, _depth + 1)
            if sub is None:
                return None
            expr = re.sub(rf"\b{re.escape(ref)}\b", str(sub), expr)
        if re.fullmatch(r"[0-9+\-*/() ]+", expr):
            try:
                return int(eval(expr, {"__builtins__": {}}, {}))   # noqa: S307
            except Exception:
                return None
        return None

    def to_json(self) -> str:
        return json.dumps({
            "entries": {k: {"value": str(v.value), "int": v.as_int,
                            "source": v.source, "kind": v.kind, "doc": v.doc}
                        for k, v in sorted(self.entries.items())},
            "harness_instantiation": self.harness,
            "errors": self.errors,
        }, indent=1, ensure_ascii=False, sort_keys=False)


def _read(root: Path, rel: str) -> Optional[str]:
    path = root / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def extract(root) -> CompileManifest:
    """从源码抽清单. 找不到文件或认不出写法都记进 errors, 不静默跳过."""
    root = Path(root)
    man = CompileManifest()

    cmake = _read(root, CMAKE_FILE)
    if cmake is None:
        man.errors.append(f"缺文件: {CMAKE_FILE} (MEGAMOE_* 的唯一定义处)")
    else:
        for mo in _CMAKE_SET.finditer(cmake):
            name, value, doc = mo.group(1), mo.group(2), mo.group(3)
            man.entries[name] = ManifestEntry(
                name=name, value=value, kind="cmake_cache", doc=doc,
                source=f"{CMAKE_FILE}:{_line_of(cmake, mo.start())}")
        for mo in _CMAKE_OPTION.finditer(cmake):
            name, doc, value = mo.group(1), mo.group(2), mo.group(3)
            man.entries[name] = ManifestEntry(
                name=name, value=value, kind="cmake_option", doc=doc,
                source=f"{CMAKE_FILE}:{_line_of(cmake, mo.start())}")

    for rel in MACRO_FILES:
        text = _read(root, rel)
        if text is None:
            man.errors.append(f"缺文件: {rel}")
            continue
        for mo in _MACRO_DEFAULT.finditer(text):
            name, value = mo.group(1), mo.group(2).strip()
            if name.endswith("_H") or name.endswith("_H_"):
                continue            # 头文件保护宏, 不是编译参数
            man.entries.setdefault(name, ManifestEntry(
                name=name, value=value, kind="macro",
                source=f"{rel}:{_line_of(text, mo.start())}"))

    for name, rel in STRUCTURAL.items():
        text = _read(root, rel)
        if text is None:
            man.errors.append(f"缺文件: {rel} (要从它抽 {name})")
            continue
        found = None
        for mo in _CONSTEXPR.finditer(text):
            if mo.group(1) == name:
                found = ManifestEntry(
                    name=name, value=mo.group(2).strip(), kind="constexpr",
                    source=f"{rel}:{_line_of(text, mo.start())}")
                break
        if found is None:
            man.errors.append(
                f"{rel} 里找不到 constexpr {name} —— 名字改了或写法变了, 清单不能少这一项")
        else:
            man.entries[name] = found

    gmm = _read(root, _k(*_COMMON, "mega_moe_gmm_common.h"))
    if gmm:
        mo = _SWIZZLE.search(gmm)
        if mo is None:
            man.errors.append("找不到 BlockSchedulerSwizzle<Offset, Direction> 实例化")
        else:
            line = _line_of(gmm, mo.start())
            man.entries["SWIZZLE_OFFSET"] = ManifestEntry(
                name="SWIZZLE_OFFSET", value=mo.group(1), kind="template",
                source=f"mega_moe/op_kernel/arch35/common/mega_moe_gmm_common.h:{line}",
                doc="BlockSchedulerSwizzle 的第 1 个模板实参")
            man.entries["SWIZZLE_DIRECTION"] = ManifestEntry(
                name="SWIZZLE_DIRECTION", value=mo.group(2), kind="template",
                source=f"mega_moe/op_kernel/arch35/common/mega_moe_gmm_common.h:{line}",
                doc="第 2 个模板实参; 0 = m first, 1 = n first")

    harness = _read(root, _k("include", "kernel.cpp"))
    if harness:
        mo = _HARNESS_INSTANCE.search(harness)
        if mo is None:
            man.errors.append("include/kernel.cpp 里找不到 MegaMoe* 的模板实例化")
        else:
            args = [a.strip() for a in mo.group(2).replace("\n", " ").split(",")]
            names = ("XType", "OutputType", "TopkWeightsType", "Weight1Type",
                     "QuantMode", "CombineQuantMode", "TopkWeightsPrefetch",
                     "IsGmm1Interleaved")
            man.harness = {"class": mo.group(1),
                           "source": f"mega_moe/include/kernel.cpp:"
                                     f"{_line_of(harness, mo.start())}"}
            man.harness.update({k: v for k, v in zip(names, args)})
    return man


#: **有意**不同的项: Python 名 -> (清单名, 理由)。不在这里登记的差异一律算失配。
INTENTIONAL = {
    "combine_meta_bytes_per_row": (
        "META_INFO_SIZE_BYTES",
        "模型缺省 16 = 四个具名字段 (算法下界口径); kernel 的 META_INFO_SIZE=8 个 int32 "
        "= 32B 是搬满一整条 DataCopy。profiles.MEGAMOE_A8W8 显式声明 32, 所以「复现那份"
        "实现」时两者一致; 缺省保持下界是刻意的最少假设。"),
}


def compare(man: CompileManifest, kernel=None) -> Dict[str, object]:
    """清单 vs Python 侧取值. 返回 {"mismatch": [...], "checked": n, "skipped": [...]}.

    kernel = KernelConfig (None 用缺省)。对账表写在这里而不是散在各处: 每一项都要能说出
    "Python 的哪个字段" 对 "清单的哪个名字"。
    """
    from ..config import hardware as hw
    from ..config.hardware import KernelConfig

    km = kernel if kernel is not None else KernelConfig()
    # META_INFO_SIZE 是"几个 int32", 字节数要 x4 —— 派生项先建好, 下面的对账表要用
    meta_slots = man.value("META_INFO_SIZE")
    if meta_slots is not None:
        man.entries["META_INFO_SIZE_BYTES"] = ManifestEntry(
            name="META_INFO_SIZE_BYTES", value=meta_slots * 4, kind="constexpr",
            source=man.entries["META_INFO_SIZE"].source + " (x4: int32)",
            doc="META_INFO_SIZE 个 int32 的字节数")
    pairs = [
        # (说明, Python 取值, 清单名)
        ("KernelConfig.tile_m", km.tile_m, "MEGAMOE_TILE_M"),
        ("KernelConfig.tile_n", km.tile_n, "MEGAMOE_TILE_N"),
        ("KernelConfig.l1_buf_num", km.l1_buf_num, "MEGAMOE_L1_BUF_NUM"),
        ("KernelConfig.l1_tile_k", km.l1_tile_k, "L1_TILE_K"),
        ("KernelConfig.activation_n_half", km.activation_n_half, "ACTIVATION_N_HALF"),
        ("KernelConfig.swizzle_offset", km.swizzle_offset, "SWIZZLE_OFFSET"),
        ("KernelConfig.swizzle_direction", km.swizzle_direction, "SWIZZLE_DIRECTION"),
        # 单位要对齐: META_INFO_SIZE 是"几个 int32", Python 这个字段是字节数
        ("KernelConfig.combine_meta_bytes_per_row", km.combine_meta_bytes_per_row,
         "META_INFO_SIZE_BYTES"),
        ("hardware.TILE_M", int(hw.TILE_M), "MEGAMOE_TILE_M"),
        ("hardware.TILE_N", int(hw.TILE_N), "MEGAMOE_TILE_N"),
        ("hardware.L1_TILE_K", int(hw.L1_TILE_K), "L1_TILE_K"),
        ("hardware.ACTIVATION_N_HALF", int(hw.ACTIVATION_N_HALF), "ACTIVATION_N_HALF"),
        ("hardware.MXFP_DIVISOR_SIZE", int(hw.MXFP_DIVISOR_SIZE), "MXFP_DIVISOR_SIZE"),
        ("hardware.MXFP_MULTI_BASE_SIZE", int(hw.MXFP_MULTI_BASE_SIZE),
         "MXFP_MULTI_BASE_SIZE"),
        ("hardware.TOTAL_UB_SIZE", int(hw.TOTAL_UB_SIZE), "LAYERED_USABLE_UB_BYTES"),
        ("hardware.GMM2_LAG_MIN_TOKEN_NUM", int(hw.GMM2_LAG_MIN_TOKEN_NUM),
         "GMM2_LAG_MIN_TOKEN_NUM"),
        ("hardware.LAYERED_META_BYTES_PER_ROW", int(hw.LAYERED_META_BYTES_PER_ROW),
         "META_INFO_SIZE_BYTES"),
        ("hardware.URMA_FLAG_WINDOW_TOKENS", int(hw.URMA_FLAG_WINDOW_TOKENS),
         "L1_TILE_M_256"),
        # prefetch 的 epilogue 行块高度: kernel 的 EPILOGUE_TILE_M 真值分支
        ("hardware.EPILOGUE_TILE_M_PREFETCH", int(hw.EPILOGUE_TILE_M_PREFETCH),
         "L1_TILE_M_128"),
        ("hardware.META_BYTES_PER_ROW", int(hw.META_BYTES_PER_ROW),
         "META_INFO_SIZE_BYTES"),
    ]
    mismatch, skipped, checked = [], [], 0
    for label, py_value, entry_name in pairs:
        want = man.value(entry_name)
        if want is None:
            skipped.append(f"{label}: 清单里没有 {entry_name} 或它不是整数")
            continue
        checked += 1
        field_name = label.split(".", 1)[-1]
        if int(py_value) == want:
            continue
        reason = INTENTIONAL.get(field_name)
        if reason and reason[0] == entry_name:
            skipped.append(f"{label}: 有意不同 ({py_value} vs {want}) — {reason[1]}")
            continue
        mismatch.append({
            "python": label, "python_value": int(py_value),
            "manifest": entry_name, "manifest_value": want,
            "source": man.entries[entry_name].source})
    return {"mismatch": mismatch, "checked": checked, "skipped": skipped,
            "errors": list(man.errors)}
