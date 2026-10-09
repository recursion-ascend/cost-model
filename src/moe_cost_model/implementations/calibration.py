"""按域登记的标定值: 一个数只在它量过的地方有效.

要替换的现状 —— 全部标定常数是 `config/hardware.py` 的模块级全局量, 一套数覆盖所有
实现、所有编译点、所有形状、所有拓扑。而那些常数自己的注释早就写明了它们不是这样的:

  BW_L1_GM 51900     "H 扫差分; B=64 域单点, 并发未扫"; 按核数重拟是 28 核 45300 / 18 核 37000
  BW_REMOTE_WRITE    三个形状给出 9.5 / 7.8 / 4.5 GB/s 每核, 相差 2.1 倍
  URMA_GET_LAT_US    "4 卡 / 3 条流; 超过 world-1 > 3 未验证"
  T_RANK_SYNC_RTT_US 缺省 1.6-2.0, h6144 是 2.2-2.5 (随形状变)
  BW_UNPERMUTE_AGG   两个尺度 B=64 与 B=1024 之间 +18%

也就是说: **换一套编排/拓扑/形状, 这些数未必还成立**, 而模型此前没有地方记这件事, 更没有
地方在越域时提醒。本模块给每个标定值配一个 CalibrationDomain (实现 id + 编译指纹 + 形状域 +
拓扑), 查表时报出三种结果:

  in_domain      实际运行落在量过的范围内
  out_of_domain  某些维度越界 —— 给出是哪些维, 以及当时量的范围
  undeclared     某些维度**从没声明过范围** —— 与越界分开报: "量过但不在范围内" 与
                 "从没说过这一维" 是两种不同的不确定性

**不做自动外推**: 越域时不给一个"修正后的值"。模型没有那条规律 (BW_L1_GM 的核数依赖只有
两个点, BW_REMOTE_WRITE 的形状依赖只有三个点), 凭两三个点造一条曲线再用它去外推, 比直接
说"这里超出标定域"更坏。

种子数据的来源: 每条 evidence 都指向 `config/hardware.py` 里那条常数的注释与 data/ 下的 run
目录名。域的边界也取自那些注释 —— 没写的维度就留 None (未声明), 不替它编一个范围。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..config.hardware import (BW_L1_GM, BW_LOCAL_GM, BW_REMOTE_WRITE,
                               BW_UNPERMUTE_AGG, T_RANK_SYNC_RTT_US,
                               URMA_GET_BW_SINGLE, URMA_GET_LAT_US)
from .compile import CompileConfig
from .identity import (CalibrationDomain, ImplementationId, RuntimeTopology,
                       ShapeDomain)
from .megamoe import A8W8WaveV1, LayeredV1

#: 标定语料的编译点: include/kernel.cpp 写死 CombineQuantMode=COMBINE_NO_QUANT 与
#: IsGmm1Interleaved=false, MEGAMOE_* 全取 CMake 缺省 —— 所以全部实测值共用这一个指纹。
#: 这不是假设: tools/compile_manifest.py 从源码抽出来的 harness 实例化就是这组。
#: combine_meta_bytes_per_row 取 32 而不是模型缺省的 16: 打点跑的是 kernel, 而 kernel 搬满
#: META_INFO_SIZE = 8 个 int32 = 32B (common/mega_moe_constants.h)。写 16 会让语料的指纹与
#: profiles.MEGAMOE_A8W8 的不同, 于是"复现那份实现"的场景查标定时全部报 wrong_key ——
#: 那是种子数据的错, 不是真的换了二进制。
CORPUS_COMPILE = CompileConfig(combine_meta_bytes_per_row=32,
                               provenance="include/kernel.cpp (标定语料)")

#: 标定语料的拓扑: data/ 下六个 run 都是 4 卡 x 28 核 (config.json5 的 device.aic_cores)。
CORPUS_TOPOLOGY = RuntimeTopology(world_size=4, active_cores=28, ranks_per_server=4)

#: 语料覆盖的形状: bs ∈ {36, 128, 8192}, h = 5120, topk = 6, 每卡 3 或 16 个本地专家
#: (data/*/config.json5)。
#:
#: **hidden_dim 是 2I 不是 I**: 那批 run 的 config.json5 写 `intermediate: 4608`, 同一处
#: 注明 `hiddenDim = 2I = 9216`, 六个场景文件也都写 hidden_dim = 9216。本模型的
#: hidden_dim 字段是 GMM1 的输出宽度, 即 2I (SwiGLU 的 gate+up 两半)。
#: hidden_dim 是 2I (9216), 不是 I —— 写成 I 会让这一层把**自己的语料**判成越域,
#: 而它的全部职责就是"一个数只在它量过的地方有效"。
#:
#: 注意 examples/scenario_basic.toml 用的 h=6144/hidden_dim=4096 **仍在这个域外** ——
#: 这正是要能查出来的那种情况。
CORPUS_SHAPE = ShapeDomain(token_num=(36, 8192), h=(5120, 5120),
                           hidden_dim=(9216, 9216), topk=(6, 6),
                           local_experts=(3, 16))

#: 语料里一个**代表性的实际形状** (不是范围)。工具与测试都从这里取, 不各自再写一份 ——
CORPUS_POINT = {"token_num": 128, "h": 5120, "hidden_dim": 9216, "topk": 6,
                "local_experts": 3}


@dataclass(frozen=True)
class CalibrationRecord:
    """一条标定值 + 它的适用域 + 证据."""

    name: str
    value: float
    domain: CalibrationDomain
    label: str = ""                 # 原常数的出处标签 (measured:/assumed:/derived:)
    #: 同一个量在别的条件下的其它观测 (条件描述 -> 值)。记下来是为了让"它随什么变"
    #: 看得见, 而不是埋在注释里。
    observations: Tuple[Tuple[str, float], ...] = ()

    @property
    def spread(self) -> Optional[float]:
        """其它观测与本值的最大比值 (>1). None = 只有一个观测.

        它是"这个数有多不稳"的直接度量: BW_REMOTE_WRITE 的 spread 是 2.11, 比模型平时
        争论的差异大得多。
        """
        if not self.observations:
            return None
        values = [v for _, v in self.observations] + [self.value]
        lo, hi = min(values), max(values)
        return (hi / lo) if lo > 0 else None


@dataclass(frozen=True)
class Lookup:
    """一次查表的结果."""

    record: CalibrationRecord
    verdict: str                      # in_domain | out_of_domain | undeclared | wrong_key
    out_of_range: Tuple[str, ...] = ()
    undeclared: Tuple[str, ...] = ()
    topology_mismatch: Tuple[str, ...] = ()
    note: str = ""

    @property
    def usable(self) -> bool:
        """落在域内才算"可直接用". 其余要么越域要么没声明, 调用方必须知道."""
        return self.verdict == "in_domain"


class CalibrationTable:
    """按 (实现 id, 编译指纹) 分组的标定值表."""

    def __init__(self, records: Sequence[CalibrationRecord] = ()):
        self._by_key: Dict[Tuple[str, str], Dict[str, CalibrationRecord]] = {}
        for rec in records:
            self.add(rec)

    def add(self, record: CalibrationRecord) -> None:
        key = record.domain.key
        self._by_key.setdefault(key, {})[record.name] = record

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted({n for group in self._by_key.values() for n in group}))

    def records(self) -> Tuple["CalibrationRecord", ...]:
        """表里的全部记录, 按 (名字, 域键) 排序。审计这张表本身要用它."""
        return tuple(sorted(
            (rec for group in self._by_key.values() for rec in group.values()),
            key=lambda r: (r.name, r.domain.key)))

    def lookup(self, name: str, *, implementation: ImplementationId,
               compile_fingerprint: str, shape: Optional[Dict[str, int]] = None,
               topology: Optional[Dict[str, int]] = None) -> Optional[Lookup]:
        """查一个标定值. 返回 None = 这个键下没有这条记录 (而不是悄悄回落到全局量)."""
        group = self._by_key.get((implementation.key, compile_fingerprint))
        if group is None or name not in group:
            # 同一个实现的其它编译点上有没有? 有就明确说"键不对", 不默默换一个用。
            for (impl_key, fp), other in self._by_key.items():
                if impl_key == implementation.key and name in other:
                    return Lookup(record=other[name], verdict="wrong_key",
                                  note=f"该值是在编译指纹 {fp} 上量的, 当前是 "
                                       f"{compile_fingerprint} —— 不同的二进制")
            return None
        rec = group[name]
        shape = shape or {}
        topology = topology or {}
        ok, bad = rec.domain.shape.covers(**shape)
        missing = rec.domain.shape.undeclared(**shape)
        topo_bad = rec.domain.topology.mismatch(**topology)
        if bad or topo_bad:
            return Lookup(record=rec, verdict="out_of_domain", out_of_range=bad,
                          undeclared=missing, topology_mismatch=topo_bad)
        if missing:
            return Lookup(record=rec, verdict="undeclared", undeclared=missing)
        return Lookup(record=rec, verdict="in_domain")

    def audit(self, *, implementation: ImplementationId, compile_fingerprint: str,
              shape: Optional[Dict[str, int]] = None,
              topology: Optional[Dict[str, int]] = None) -> Dict[str, Lookup]:
        """把表里全部记录对给定运行条件查一遍 (给报告用)."""
        out: Dict[str, Lookup] = {}
        for name in self.names():
            got = self.lookup(name, implementation=implementation,
                              compile_fingerprint=compile_fingerprint,
                              shape=shape, topology=topology)
            if got is not None:
                out[name] = got
        return out


def _corpus_domain(impl: ImplementationId, evidence: Tuple[str, ...] = (),
                   **shape_overrides) -> CalibrationDomain:
    shape = replace(CORPUS_SHAPE, **shape_overrides) if shape_overrides else CORPUS_SHAPE
    return CalibrationDomain(implementation=impl,
                            compile_fingerprint=CORPUS_COMPILE.fingerprint,
                            shape=shape, topology=CORPUS_TOPOLOGY, evidence=evidence)


def default_table() -> CalibrationTable:
    """把 config/hardware.py 里**自己写明了标定条件**的那些常数登记进来.

    只登记注释里说清了条件的: 其余常数留在模块里不动 (登记一个条件不明的值, 等于给它
    编一个域)。每条的 observations 取自同一段注释里记下的其它观测。
    """
    a8w8 = A8W8WaveV1().identity()
    layered = LayeredV1().identity()
    runs = ("data/20260930_154158_112575_bs36_h5120_i4608_k6_cyclic_noshared",
            "data/20260930_154407_446172_bs128_h5120_i4608_k6_cyclic_noshared",
            "data/20260930_145851_854111_bs8192_h5120_i4608_k6_cyclic_noshared")
    records = [
        CalibrationRecord(
            name="bw_l1_gm", value=float(BW_L1_GM), label="measured",
            domain=_corpus_domain(a8w8, evidence=runs + ("hardware.BW_L1_GM 的注释",)),
            # 同一个量按并发核数重拟的另外两个观测 —— 它随拓扑变, 幅度 1.40 倍
            observations=(("28 核并发重拟", 45300.0), ("18 核并发重拟", 37000.0))),
        CalibrationRecord(
            name="bw_remote_write", value=float(BW_REMOTE_WRITE), label="measured",
            domain=_corpus_domain(a8w8, evidence=runs),
            # 三个形状各自反扣出的每核跨卡写带宽; 2.11 倍的离散度
            observations=(("bs36", 9500.0), ("bs128 (全程有争用)", 7800.0),
                          ("bs8192", 4500.0))),
        CalibrationRecord(
            name="bw_unpermute_agg", value=float(BW_UNPERMUTE_AGG), label="measured",
            domain=_corpus_domain(a8w8, evidence=runs),
            observations=(("h6144 (缺省形状, +18%)", float(BW_UNPERMUTE_AGG) * 1.18),)),
        CalibrationRecord(
            name="t_rank_sync_rtt_us", value=float(T_RANK_SYNC_RTT_US), label="measured",
            domain=_corpus_domain(a8w8, evidence=runs),
            observations=(("缺省形状 r1/r3", 1.8), ("h6144 r1/r3", 2.35))),
        CalibrationRecord(
            name="bw_local_gm", value=float(BW_LOCAL_GM), label="measured",
            # 唯一一个标定条件窄到只靠硬件就能说清的: 单核大块 MTE, 无其它流量。
            # 所以形状与拓扑都不声明 (留 None = 未声明, 不是"任意")。
            domain=CalibrationDomain(
                implementation=a8w8, compile_fingerprint=CORPUS_COMPILE.fingerprint,
                evidence=("hardware.BW_LOCAL_GM 的注释: MTE 大尺寸拟合, 单核无竞争",))),
        CalibrationRecord(
            name="urma_get_lat_us", value=float(URMA_GET_LAT_US), label="measured",
            domain=CalibrationDomain(
                implementation=layered,
                compile_fingerprint=CompileConfig(
                    comm_mode="urma", provenance="URMA 探针").fingerprint,
                shape=ShapeDomain(h=(5120, 5120)),
                topology=RuntimeTopology(world_size=4, concurrent_streams=3),
                evidence=("hardware.URMA_GET_LAT_US 的注释: 4 卡 / CANN 9.1.0 / "
                          "6336B 行 / 512B 对齐槽 / 单核 channel-owner / 3 条流",))),
        CalibrationRecord(
            name="urma_get_bw_single", value=float(URMA_GET_BW_SINGLE), label="measured",
            domain=CalibrationDomain(
                implementation=layered,
                compile_fingerprint=CompileConfig(
                    comm_mode="urma", provenance="URMA 探针").fingerprint,
                shape=ShapeDomain(h=(5120, 5120)),
                topology=RuntimeTopology(world_size=4, concurrent_streams=3),
                evidence=("同上探针; drain-chunk=8 的下界",))),
    ]
    return CalibrationTable(records)


def audit_run(result_implementation: Dict[str, object], shape: Dict[str, int],
              table: Optional[CalibrationTable] = None) -> Dict[str, Lookup]:
    """给一次运行的 rank_results[i]["implementation"] 做标定域审计.

    用法: 跑完取 rank_results[0]["implementation"], 连形状一起传进来, 得到"这次用到的
    标定值里, 哪些落在它们量过的域内"。
    """
    table = table or default_table()
    ident = ImplementationId.parse(str(result_implementation["id"]))
    topo = dict(result_implementation.get("topology") or {})
    return table.audit(implementation=ident,
                       compile_fingerprint=str(
                           result_implementation["compile_fingerprint"]),
                       shape=shape, topology=topo)
