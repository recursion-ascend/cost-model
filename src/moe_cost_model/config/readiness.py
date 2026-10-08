"""第 0 层: 就绪粒度 —— 消费者沿共享轴分成几段独立就绪.

语义 (与任何 kernel 实现无关)
-----------------------------
一条生产者->消费者的边上, 若两边共有一条轴 (生产者切分的轴 ∩ 消费者的某条轴),
且消费者能沿这条轴**增量消费**, 那么消费者就不必等生产者把整条轴产完:

    把共享轴分成 S 段, 第 j 段只挂覆盖第 j 段范围的那些产出。

S=1 就是等齐, S 越大越细、开工越早。段界必须落在共享轴的**自然块**边界上
(自然块由边自己给: GMM2 的 K 轴是 kL1 块, 别的边见 config/links.EDGE_AXES),
因为"半块就绪"在硬件上没有对应物 —— 累加器按块攒, 标志按块置。

取值: 四个档, 没有哨兵
----------------------
    "whole"        等齐 (缺省, 最少假设: 不声称实现能在只拿到部分轴时起步)
    N (>= 2 的整数) 按自然块数**均分** N 段
    "per_chunk"    最细: 一个自然块一段
    "first_chunk"  首块一段 + 其余一段

整数只表示"均分几段", 不再有特例: 0 与 1 都**报错**而不是另作他解
(0 曾经是"最细", 1 曾经是"等齐")。理由是旧编码不单调 ——
1=等齐 / 2=首块+其余 / 0=最细 / >=块数=最细, 字面"段数"只在 3..n-1 成立,
读者没法从数字本身判断哪个更细。现在"更细"就是"段数更多", 单调:

    whole(1 段)  <=  N 段 (min(N, 块数))  <=  per_chunk(块数)

分段的代价本模型**不计**: 段越多, 实现侧每段要多走一次标志等待/轮询。
它由 ``StageLink.segment_sync_us`` 显式表达, 缺省 0 = **未标定**, 不是"量过是零"
—— 所以缺省下"分得越细越好"是上界, 不是结论 (见 analysis.sensitivity 的
UNCERTAIN_INPUTS)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: 四个档的档名。EVEN 没有字面写法 —— 均分要给段数, 写整数。
WHOLE = "whole"
PER_CHUNK = "per_chunk"
FIRST_CHUNK = "first_chunk"
EVEN = "even"
#: 场景文件里能写的字符串档 (均分写整数, 所以 EVEN 不在其中)
NAMED: Tuple[str, ...] = (WHOLE, PER_CHUNK, FIRST_CHUNK)
KINDS: Tuple[str, ...] = NAMED + (EVEN,)


@dataclass(frozen=True)
class Readiness:
    """消费者沿共享轴分几段就绪. 用四个构造器取值, 不要直接填 kind."""

    kind: str = WHOLE
    #: 只有 kind == EVEN 时有意义 (均分几段); 其余档为 None
    segments: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"readiness: 不认识的档 {self.kind!r}; 只能是 {KINDS}")
        if self.kind == EVEN:
            if not isinstance(self.segments, int) or isinstance(self.segments, bool):
                raise ValueError(f"readiness: 均分的段数必须是整数, 得到 {self.segments!r}")
            if self.segments < 2:
                raise ValueError(
                    f'readiness: 均分至少 2 段 (得到 {self.segments}); '
                    f'一段就是 "{WHOLE}"')
        elif self.segments is not None:
            raise ValueError(f'readiness: "{self.kind}" 不带段数, 得到 {self.segments!r}')

    # ---------------------------------------------------------------- 构造器
    @classmethod
    def whole(cls) -> "Readiness":
        """等齐: 一段, 覆盖整条共享轴."""
        return cls(WHOLE)

    @classmethod
    def n(cls, segments: int) -> "Readiness":
        """按自然块数均分 segments 段 (segments >= 2)."""
        return cls(EVEN, int(segments))

    @classmethod
    def per_chunk(cls) -> "Readiness":
        """最细: 一个自然块一段."""
        return cls(PER_CHUNK)

    @classmethod
    def first_chunk(cls) -> "Readiness":
        """首块一段 + 其余一段."""
        return cls(FIRST_CHUNK)

    # ---------------------------------------------------------------- 取值
    @property
    def spelling(self):
        """场景文件里的写法 (字符串档名, 或均分的段数)."""
        return self.segments if self.kind == EVEN else self.kind

    def __str__(self) -> str:
        return str(self.spelling)

    @property
    def is_whole(self) -> bool:
        return self.kind == WHOLE

    def chunk_counts(self, n_chunks: int) -> Tuple[int, ...]:
        """每段吃几个自然块. 和 == n_chunks, 长度 = 实际段数.

        余数规则**写死**: 均分除不尽时, 多出来的块给**前面**的段
        (9 块分 4 段 -> 3,2,2,2)。段数超过块数时就是每块一段 —— 块是最小单位,
        再分就是"半块就绪", 硬件上没有对应物。
        """
        if n_chunks < 1:
            raise ValueError(f"n_chunks 必须 >= 1, 得到 {n_chunks}")
        if self.kind == WHOLE:
            return (n_chunks,)
        if self.kind == PER_CHUNK:
            return (1,) * n_chunks
        if self.kind == FIRST_CHUNK:
            return (1,) if n_chunks == 1 else (1, n_chunks - 1)
        want = min(int(self.segments), n_chunks)
        per, rem = divmod(n_chunks, want)
        return tuple(per + (1 if j < rem else 0) for j in range(want))

    def segment_count(self, n_chunks: int) -> int:
        """这条边上实际分成几段 (受块数上限)."""
        return len(self.chunk_counts(n_chunks))


def parse_readiness(value) -> Readiness:
    """场景文件 / Python 里写的取值 -> Readiness. 不认识的直接报错."""
    if isinstance(value, Readiness):
        return value
    if isinstance(value, str):
        if value in NAMED:
            return Readiness(value)
        raise ValueError(
            f'不认识 {value!r}; 可写 {" / ".join(NAMED)} '
            f"或 >= 2 的整数 (均分几段)")
    if isinstance(value, bool):
        raise ValueError(f"布尔值不是段数, 得到 {value!r}")
    if isinstance(value, int):
        if value == 0:
            raise ValueError(
                '0 不再是取值 —— 最细写 "per_chunk"。'
                "(0 曾经表示最细, 与 granularity 的 0=整片、dispatch 的 0=沿用 "
                "tiling 三处相反, 已去掉)")
        if value == 1:
            raise ValueError(
                '整数只表示"均分几段", 必须 >= 2; 一段就是 "whole" (缺省)')
        if value < 0:
            raise ValueError(f"段数不能为负, 得到 {value}")
        return Readiness.n(value)
    raise ValueError(
        f'应为 {" / ".join(NAMED)} 之一或 >= 2 的整数, 得到 {value!r}')


def chunk_count(extent: int, chunk: int) -> int:
    """共享轴有几个自然块 (向上取整)."""
    if extent <= 0:
        raise ValueError(f"共享轴长度必须为正, 得到 {extent}")
    if chunk <= 0:
        raise ValueError(f"自然块大小必须为正, 得到 {chunk}")
    return -(-extent // chunk)


def segment_spans(readiness: Readiness, extent: int, chunk: int):
    """沿共享轴的分段区间 [(lo, hi), ...]: 首段从 0, 末段到 extent, 无缺口无重叠.

    段界一律落在自然块边界上 (最后一块可能不满)。extent 与 chunk 都以**轴的单位**
    计 (GMM2 的 K 是元素数, 自然块是 kL1; 换一条边就换一对单位, 本函数不关心)。
    """
    counts = readiness.chunk_counts(chunk_count(extent, chunk))
    out, lo = [], 0
    for take in counts:
        hi = lo + take
        out.append((lo * chunk, min(hi * chunk, extent)))
        lo = hi
    return out
