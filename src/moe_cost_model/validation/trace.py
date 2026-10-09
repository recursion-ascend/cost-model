"""读实测 trace (Chrome Trace Event 格式), 并把它翻成模型的词表.

数据事实 (data/ 下 6 个 run x 4 rank = 24 个文件):
  * 16 个文件完整可解析 (bs36 与 bs128 两组, 各 4 rank);
  * 8 个文件**被截断** —— 两个 bs8192 run 的全部 rank, 都在 7602176 字节处断在一条记录
    中间 (同一个字节数, 说明是采集侧的写入上限, 不是随机损坏)。
    截断的文件照样能用: 按记录边界回退到最后一条完整记录即可, 但**必须报告**它是截断的,
    否则"事件数比模型少"会被当成模型的问题。

能比的与不能比的 (这一条决定步骤 6 的范围, 所以写在最前面):
  能比   逐 stage 事件数、逐核分布、(wave, expert) 的分组、波数
  不能比 **字节数**: trace 的 args 只有 rank/local_id/payload/cycles/wave/expert,
         没有任何搬运字节字段。所以"申报字节 vs 实测字节"这项对不了 —— 不是没做, 是
         数据里没有。
  不能比 **buffer 生命周期**: trace 里只有它的影子 (WAIT_GMM1_BUFFER 这类等待事件),
         没有槽位的取/还记录。

stage 名对照 (trace 名 -> 模型 stage): 见 STAGE_MAP。trace 的名字形如 "GMM1·w0",
中点后面是波号, 所以取名字要按中点切。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

#: trace 事件名 (中点前的部分) -> 模型 stage 名。只映**模型也发的**那些;
#: 其余 (INIT / INPUT_QUANT / WAIT_* / SYNC_RESET ...) 要么是模型不建图的前导,
#: 要么是等待事件, 单独归类, 不硬塞进某个 stage。
STAGE_MAP = {
    "GMM1": "gmm1",
    "ACT_QUANT": "activation",
    "GMM2": "gmm2",
    "COMBINE": "combine",
    "DISPATCH_XFER": "dispatch",
    "DISPATCH_LOCAL": "dispatch",
    "DISPATCH_SCHEDULE": "dispatch_call",
    "COUNTS_EXPORT": "epilogue",
    "UNPERMUTE": "epilogue",
    "OUTPUT_BUFFER_INIT": "epilogue",
    "FINALIZE": "epilogue",
}

#: 等待类事件: 它们是**同步与缓冲的影子**, 不是 stage。模型里对应的是边上的延迟或槽位
#: 约束, 不是独立事件 —— 所以计数单独出, 不与 stage 混。
WAIT_PREFIX = "WAIT_"

#: 模型完全不建图的前导/杂项 (见 config/hardware 的 T_INPUT_QUANT_* 注释: 前导已移出 DAG)
NOT_MODELLED = frozenset({
    "INIT", "INPUT_QUANT", "INPUT_BUFFER_INIT", "DISPATCH_BUFFER_INIT",
    "TOKEN_COUNT_PREPARE", "ROUTE_SEND", "SYNC_RESET", "KERNEL",
})

_THREAD = re.compile(r"^(AIC|AIV)(\d*)-(\d+)$")


@dataclass(frozen=True)
class TraceEvent:
    """一条实测事件, 已翻成模型词表."""

    raw_name: str
    stage: Optional[str]            # 模型 stage; None = 不对应任何 stage
    engine: str                     # AIC / AIV0 / AIV1
    core: int
    start_us: float
    duration_us: float
    wave: Optional[int]
    expert: Optional[int]
    is_wait: bool


@dataclass
class TraceFile:
    """一个 trace 文件的读取结果."""

    path: str
    rank: Optional[int]
    events: Tuple[TraceEvent, ...] = ()
    truncated: bool = False
    recovered_records: int = 0
    note: str = ""

    def by_stage(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ev in self.events:
            if ev.stage is None:
                continue
            out[ev.stage] = out.get(ev.stage, 0) + 1
        return dict(sorted(out.items()))

    def by_core(self, stage: str) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for ev in self.events:
            if ev.stage == stage:
                out[ev.core] = out.get(ev.core, 0) + 1
        return dict(sorted(out.items()))

    def waves(self, stage: str) -> Tuple[int, ...]:
        return tuple(sorted({ev.wave for ev in self.events
                             if ev.stage == stage and ev.wave is not None}))

    def waits(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ev in self.events:
            if ev.is_wait:
                out[ev.raw_name] = out.get(ev.raw_name, 0) + 1
        return dict(sorted(out.items()))


def _records(text: str) -> Tuple[List[dict], bool, int]:
    """从 traceEvents 数组里取记录; 截断时回退到最后一条完整记录.

    为什么不用 json.load 就算了: 两个 bs8192 run 的 8 个文件都在同一个字节数处断在记录
    中间, 直接解析会整文件失败。按花括号配平扫一遍可以救回前面所有完整记录 —— 但要把
    "这是截断的"一起返回, 不能让它看起来像一个完整但事件较少的 run。
    """
    try:
        whole = json.loads(text)
        return list(whole.get("traceEvents", [])), False, 0
    except json.JSONDecodeError:
        pass
    start = text.find("[", text.find("traceEvents"))
    if start < 0:
        return [], True, 0
    out: List[dict] = []
    depth = 0
    begin = -1
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and begin >= 0:
                try:
                    out.append(json.loads(text[begin:i + 1]))
                except json.JSONDecodeError:
                    pass
                begin = -1
    return out, True, len(out)


def read_trace(path) -> TraceFile:
    """读一个 trace 文件 -> TraceFile (截断也能读, 但会标记)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    records, truncated, recovered = _records(text)
    # 线程名按 (pid, tid) 建键: tid 是**每 pid 独立**的, 只按 tid 建会被后一个 pid 覆盖。
    threads: Dict[Tuple[object, object], str] = {}
    processes: Dict[object, str] = {}
    rank: Optional[int] = None
    for rec in records:
        if rec.get("ph") != "M":
            continue
        if rec.get("name") == "thread_name":
            threads[(rec.get("pid"), rec.get("tid"))] = str(
                rec.get("args", {}).get("name", ""))
        elif rec.get("name") == "process_name":
            processes[rec.get("pid")] = str(rec.get("args", {}).get("name", ""))

    # 一个文件里可能有**同一次运行的多个视图** (本仓的 trace 有两个 pid:
    # "完整流水" 与 "隐藏 WAIT", 事件逐位相同)。全读进来等于每个事件数两遍 ——
    # 两个视图都算会让结构比对里实测条数恒为模型的 2 倍。只取一个: 事件最多的那个
    # (完整视图必然 >= 隐藏视图), 并列取 pid 最小的。
    per_pid: Dict[object, int] = {}
    for rec in records:
        if rec.get("ph") == "X":
            per_pid[rec.get("pid")] = per_pid.get(rec.get("pid"), 0) + 1
    view = None
    skipped_views: List[str] = []
    if per_pid:
        view = sorted(per_pid, key=lambda k: (-per_pid[k], str(k)))[0]
        skipped_views = [f"{processes.get(k, k)} ({per_pid[k]} 条)"
                         for k in sorted(per_pid, key=str) if k != view]

    events: List[TraceEvent] = []
    for rec in records:
        if rec.get("ph") != "X" or rec.get("pid") != view:
            continue
        name = str(rec.get("name", ""))
        base = name.split("·", 1)[0].strip()
        thread = threads.get((rec.get("pid"), rec.get("tid")), "")
        mo = _THREAD.match(thread)
        if mo is None:
            continue
        kind, idx, core = mo.group(1), mo.group(2), int(mo.group(3))
        engine = "AIC" if kind == "AIC" else f"AIV{idx or '0'}"
        args = rec.get("args", {}) or {}
        if rank is None and args.get("rank") is not None:
            rank = int(args["rank"])
        events.append(TraceEvent(
            raw_name=base, stage=STAGE_MAP.get(base),
            engine=engine, core=core,
            start_us=float(rec.get("ts", 0.0)), duration_us=float(rec.get("dur", 0.0)),
            wave=(int(args["wave"]) if args.get("wave") is not None else None),
            # DISPATCH_* 用 dispatch_expert 这个键, 其余用 expert
            expert=(int(args["expert"]) if args.get("expert") is not None
                    else (int(args["dispatch_expert"])
                          if args.get("dispatch_expert") is not None else None)),
            is_wait=base.startswith(WAIT_PREFIX)))
    notes: List[str] = []
    if truncated:
        notes.append(f"文件被截断 (在 {len(text)} 字节处断在一条记录中间); "
                     f"已按记录边界救回 {recovered} 条, 事件数因此是**下界**")
    if skipped_views:
        notes.append(f"文件含多个视图, 只读了 {processes.get(view, view)!r}; "
                     f"跳过 {', '.join(skipped_views)}")
    note = "; ".join(notes)
    return TraceFile(path=str(path), rank=rank, events=tuple(events),
                     truncated=truncated, recovered_records=recovered, note=note)


def read_run(run_dir) -> Dict[int, TraceFile]:
    """读一个 run 目录下全部 rank 的 trace, 按 rank 号返回."""
    run_dir = Path(run_dir)
    out: Dict[int, TraceFile] = {}
    for path in sorted(run_dir.glob("*_trace_rank*.json")):
        mo = re.search(r"_trace_rank(\d+)\.json$", path.name)
        if mo is None:
            continue
        got = read_trace(path)
        out[int(mo.group(1))] = got
    return out


def read_run_config(run_dir) -> Dict[str, object]:
    """读 run 的 config.json5 -> 形状与拓扑事实 (模型侧要用它建场景).

    为什么从这里读而不是从 tiling 真值: data/*/raw/ 整个被 gitignore, 6 个 run 的
    tiling_rank0.bin 都不在仓里 (config/pipeline.py 提到的 .json 旁置文件也没有), 所以
    examples/*.toml 那六个场景在干净克隆上**跑不起来**。config.json5 在仓里, 它给的
    bs/h/hidden/topk/ep/dtype/aic_cores 足够建出模型侧。
    json5 的注释与尾逗号用正则清掉 —— 只认这两种偏离, 认不出就报错, 不猜。
    """
    run_dir = Path(run_dir)
    path = run_dir / "config.json5"
    if not path.exists():
        raise FileNotFoundError(f"{run_dir} 下没有 config.json5")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"#[^\n]*", "", text)        # 这些文件用 # 行尾注释
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    data = json.loads(text)
    case = dict(data.get("case", {}))
    device = dict(data.get("device", {}))
    build = dict(data.get("build", {}))
    world = int(case.get("ep", 0))
    experts = int(case.get("experts", 0))
    # 字段名对照 (config.json5 的写法 -> 模型的写法):
    #   tokens       每卡 token 数            -> token_num_per_rank
    #   hidden       输入维                   -> h          (目录名里的 h5120)
    #   intermediate 专家中间维 I             -> hidden_dim = **2I** (SwiGLU 的 gate+up
    #                两半), 目录名里的 i4608 是 I; config.json5 自己也注明 hiddenDim = 2I。
    #                直接把 I 当 hidden_dim 返回会让
    #                tools/compare_trace_structure.py 按这个 cfg 建出来的模型只有一半的
    #                GMM1 n-tile (9 个而不是 18 个) —— 那正是"模型与 trace tile 数差 4x"
    #                里的一个 2 倍。原始 I 仍以 intermediate 键给出。
    #   ep           rank 数                  -> world
    #   experts      **全局**路由专家数        -> 每卡 experts/ep
    return {
        "tokens": int(case.get("tokens", 0)),
        "h": int(case.get("hidden", 0)),
        "intermediate": int(case.get("intermediate", 0)),       # I, 原样
        "hidden_dim": 2 * int(case.get("intermediate", 0)),     # 2I, 模型的口径
        "topk": int(case.get("topk", 0)),
        "world": world,
        "experts_total": experts,
        "local_experts": (experts // world if world else 0),
        "shared_experts": int(case.get("shared_experts", 0) or 0),
        "routing": str(case.get("routing", "")),
        "dtype": str(case.get("dtype", "")),
        "seed": int(case.get("seed", 0) or 0),
        "aic_cores": int(device.get("aic_cores", 0)),
        "aiv_cores": int(device.get("aiv_cores", 0)),
        "platform": str(build.get("platform", "")),
    }
