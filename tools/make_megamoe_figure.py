# -*- coding: utf-8 -*-
"""生成 fig-megamoe-timing.svg: cost model 怎么建图, 怎么算时间.

(a) 建图  (b) 一个事件算多久  (c) 排成时间线

(a) 的事件图与 (c) 的甘特图都不是手画的: 本脚本跑一个小实例, 把 build_events
拿到的真图与 scheduler 排出的真时间线画出来, 所以图改不动模型、模型改了图跟着变。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_cost_model import (DispatchMechanisticLatency, InstancePolicy,  # noqa: E402
                            KernelConfig, build_analytical_costs,
                            simulate_routing_counts)
from moe_cost_model.model import (A8W8WaveCostModel, MegaMoeShape,  # noqa: E402
                                  ModelOptions)

# ---------------------------------------------------------------- 小实例
# 选 1 个 AIC/AIV0/AIV1: 核多了全是依赖等待, 调度器看着像没干活;
# 核少了三种等待 (等前驱/等核/等槽) 才同时出现在一张甘特图里。
WORLD, LOCAL, TOKENS, TOPK = 2, 2, 256, 2
H, HIDDEN, AIC, TILE = 512, 1024, 1, 256
ROWS = TOKENS * TOPK // WORLD // LOCAL          # 每专家每源卡的行数
KERNEL = KernelConfig(tile_m=TILE, tile_n=TILE)
COSTS = build_analytical_costs(h=H, kernel=KERNEL,
                              dispatch_mechanistic=DispatchMechanisticLatency(),
                              cube_mac_per_us=2.7e7)


def instance():
    C = [[[ROWS] * WORLD for _ in range(LOCAL)] for _ in range(WORLD)]
    res = simulate_routing_counts(
        routing_counts=C, token_num_per_rank=TOKENS, h=H, hidden_dim=HIDDEN,
        aic_num=AIC, costs=COSTS, topk=TOPK, kernel=KERNEL,
        policy=InstancePolicy())
    r = res["rank_results"][res["slowest_rank"]]
    m = A8W8WaveCostModel(COSTS, ModelOptions())
    shape = MegaMoeShape(
        expert_tokens=tuple(ROWS * WORLD for _ in range(LOCAL)),
        token_num=TOKENS, h=H, hidden_dim=HIDDEN, aic_num=AIC, rank_id=0,
        expert_source_tokens=tuple(tuple(ROWS for _ in range(WORLD))
                                   for _ in range(LOCAL)),
        topk=TOPK, kernel=KERNEL, policy=InstancePolicy())
    raw, _ = m.build_events(shape)
    return res["kernel_total_us"], r, raw


TOTAL, RANK, RAW = instance()
N_EV, N_ED = len(RAW), sum(len(e.deps) for e in RAW)
SCHED = {e.name: e for e in RANK["events"]}
MOE = ("dispatch", "gmm1", "activation", "gmm2", "combine")


def ev(short):
    return SCHED["R0." + short]


# ---------------------------------------------------------------- SVG 原语
F = "'Times New Roman', Times, 'Liberation Serif', FreeSerif, serif"
F_CJK = ("'Noto Serif CJK SC', 'Source Han Serif SC', 'WenQuanYi Zen Hei', "
         "SimSun, serif")
W, H_SVG = 396, 380
o = []
A = o.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H_SVG}pt" '
  f'viewBox="0 0 {W} {H_SVG}" font-family="{F}" fill="#000">')
A('''  <defs>
    <marker id="a" viewBox="0 0 8 6" refX="7.2" refY="3" markerWidth="4.6"
            markerHeight="3.4" orient="auto"><path d="M0 0 L8 3 L0 6 z" fill="#000"/></marker>
    <marker id="ag" viewBox="0 0 8 6" refX="7.2" refY="3" markerWidth="4.2"
            markerHeight="3.2" orient="auto"><path d="M0 0 L8 3 L0 6 z" fill="#777"/></marker>
  </defs>''')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _is_cjk(c):
    o = ord(c)
    return (0x3000 <= o <= 0x303F or 0x4E00 <= o <= 0x9FFF
            or 0x3400 <= o <= 0x4DBF or 0xFF00 <= o <= 0xFFEF)


def _split_cjk(t):
    """按中文/非中文切段 — 两边要挂不同字体."""
    out, buf, cur = [], [], None
    for c in t:
        k = _is_cjk(c)
        if cur is None or k == cur:
            buf.append(c)
        else:
            out.append(("".join(buf), cur))
            buf = [c]
        cur = k
    if buf:
        out.append(("".join(buf), bool(cur)))
    return out


_ADV = {" ": 0.25, ".": 0.26, ",": 0.26, ":": 0.29, ";": 0.29, "(": 0.33,
        ")": 0.33, "/": 0.28, "-": 0.33, "=": 0.56, "+": 0.56, "_": 0.50}


def _adv(c):
    """字符宽度 (em). Times 系衬线的近似值 —— 只用来定位, 不参与排版."""
    if _is_cjk(c):
        return 1.0
    if c in _ADV:
        return _ADV[c]
    if c.isdigit():
        return 0.50
    if "A" <= c <= "Z":
        return 0.68
    if "a" <= c <= "z":
        return 0.48
    return 0.55


def _width(toks, size):
    return sum(size * (0.72 if sub else 1.0) * _adv(c)
               for t, _it, sub in toks for c in t)


def tx(x, y, s, size=5.6, anchor="start", fill=None, weight=None,
       halo=False):
    """排版标记: |...| 斜体 (数学变量), ~...~ 下标; 其余正体 (中文/函数名/数字)."""
    toks, buf, it, sub = [], [], False, False
    for c in s:
        if c in "|~":
            if buf:
                toks.append(("".join(buf), it, sub))
                buf = []
            if c == "|":
                it = not it
            else:
                sub = not sub
        else:
            buf.append(c)
    if buf:
        toks.append(("".join(buf), it, sub))
    runs, cur = [], 0.0
    for t, i_, s_ in toks:
        want = size * 0.30 if s_ else 0.0
        for piece, cjk in _split_cjk(t):
            dy, cur = want - cur, want
            a = f' dy="{dy:.2f}"' if abs(dy) > 1e-9 else ""
            a += f' font-size="{size * 0.72:.2f}"' if s_ else ""
            a += ' font-style="italic"' if i_ and not cjk else ""
            a += f' font-family="{F_CJK}"' if cjk else ""
            runs.append(f"<tspan{a}>{esc(piece)}</tspan>")
    f = f' fill="{fill}"' if fill else ""
    w = f' font-weight="{weight}"' if weight else ""
    # 不用 text-anchor: 自己量宽度左对齐。多 tspan + text-anchor 在部分渲染器
    # (cairo) 下会逐 tspan 各自对齐, 字就叠在一起。
    wd = _width(toks, size)
    if anchor != "start":
        x = x - wd / 2.0 if anchor == "middle" else x - wd
    if halo:
        A(f'  <rect x="{x - 1:.2f}" y="{y - size * 0.82:.2f}" '
          f'width="{wd + 2:.2f}" height="{size * 1.08:.2f}" fill="#fff" '
          f'stroke="none"/>')
    A(f'  <text x="{x:.2f}" y="{y:.2f}" font-size="{size}"{f}{w}>'
      + "".join(runs) + "</text>")


def rect(x, y, w, h, fill="none", sw=0.5, dash=None, stroke="#000"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    A(f'  <rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def ln(x1, y1, x2, y2, sw=0.45, dash=None, stroke="#000"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    A(f'  <path d="M{x1:.2f} {y1:.2f} L{x2:.2f} {y2:.2f}" fill="none" '
      f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arr(x1, y1, x2, y2, sw=0.5, dash=None, stroke="#000", m="a"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    A(f'  <path d="M{x1:.2f} {y1:.2f} L{x2:.2f} {y2:.2f}" fill="none" '
      f'stroke="{stroke}" stroke-width="{sw}"{d} marker-end="url(#{m})"/>')


def curve(x1, y1, cx, cy, x2, y2, sw=0.45, dash=None, stroke="#000", m=None):
    mk = f' marker-end="url(#{m})"' if m else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    A(f'  <path d="M{x1:.2f} {y1:.2f} Q{cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}" '
      f'fill="none" stroke="{stroke}" stroke-width="{sw}"{d}{mk}/>')


def head(x, y, letter, title):
    tx(x, y, "|" + letter + "|", 7.2)
    tx(x + 13, y, title, 6.3)


# ============================= (a) 建图 =============================
head(3, 11, "(a)", "建图: 路由计数 → tile → 事件图")

# -- 路由计数张量 (三层堆叠) --
tx(8, 22, "路由计数", 5.4, fill="#444")
for dx, dy in ((16, 26), (12, 30)):
    rect(dx, dy, 30, 22, fill="#fff", sw=0.4, stroke="#999")
rect(8, 34, 30, 22, fill="#fff", sw=0.55)
for i in (1, 2):
    ln(8 + i * 10, 34, 8 + i * 10, 56, 0.3, stroke="#bbb")
    ln(8, 34 + i * 7.33, 38, 34 + i * 7.33, 0.3, stroke="#bbb")
tx(8, 64, "|C|[r][e][s]", 5.4)

arr(48, 45, 62, 45, 0.5, stroke="#777", m="ag")
tx(55, 41, "Σ~s~", 5.0, "middle", fill="#444")

# -- 每专家行数 (长短不一的条) --
tx(66, 22, "每专家行数", 5.4, fill="#444")
for i, w in enumerate((40, 26, 48)):
    rect(66, 30 + i * 9, w, 7, fill="#dcdcdc", sw=0.45)
    tx(66 + w + 2, 35.4 + i * 9, f"e{i}", 4.4, fill="#666")
tx(66, 64, "|m|~e~", 5.4)

arr(124, 45, 138, 45, 0.5, stroke="#777", m="ag")
tx(131, 41, "切 tile", 5.0, "middle", fill="#444")

# -- tile 网格: 行方向切 m-group, 列方向切 n-tile; 末组不满 --
GX, GY, CW = 142, 28, 17
for j in range(3):
    for i, chh in enumerate((9, 9, 5)):
        yy = GY + sum((9, 9, 5)[:i])
        rect(GX + j * CW, yy, CW, chh,
             fill="#e4e4e4" if (i, j) == (0, 0) else "#fff", sw=0.42)
tx(GX + CW / 2, GY + 6.2, "tile", 4.3, "middle")
ln(GX - 3.5, GY, GX - 3.5, GY + 9, 0.45)
ln(GX - 5, GY, GX - 2, GY, 0.45)
ln(GX - 5, GY + 9, GX - 2, GY + 9, 0.45)
tx(GX - 6, GY + 6, "TILE_M", 4.3, "end")
ln(GX, GY - 3.5, GX + CW, GY - 3.5, 0.45)
tx(GX + CW / 2, GY - 5, "TILE_N", 4.3, "middle")
tx(GX + 3 * CW + 3, GY + 44, "末组不满", 4.3, fill="#666")
ln(GX + 3 * CW, GY + 20.5, GX + 3 * CW + 2, GY + 42, 0.35, stroke="#888")
tx(GX, 64, "一个 tile = 一个事件", 5.4)

# -- tile 数与波宽 --
FX = 222
for i, s in enumerate((
        "m-group 数 = ceil(|m|~e~ / TILE_M)",
        "GMM1 的 n-tile 数 = ceil((|hidden_dim|/2) / TILE_N)",
        "GMM2 的 n-tile 数 = ceil(|h| / TILE_N),  |K|~2~ = |hidden_dim|/2",
        "(/2 是 SwiGLU 的 gate+up 两块)")):
    tx(FX, 30 + i * 9, s, 5.1 if i < 3 else 4.6,
       fill="#666" if i == 3 else None)
tx(FX, 64, "一波投放 mgw 个 m-group (|P| = AIC 数; 波策略按逻辑 tile 计, 不除 2):", 4.6)
tx(FX, 71, "mgw = max(ceil(|P p|~1~/ceil(|hidden_dim|/TILE_N)),"
           " ceil(|P p|~2~/ceil(|h|/TILE_N)))", 4.5)

# -- 真事件图 (build_events 的输出, 取其中一个专家) --
COLS = [("dispatch", 10, 36), ("", 56, 14), ("GMM1", 82, 36),
        ("ACT", 128, 36), ("GMM2", 174, 36), ("combine", 220, 36)]
YA, YB, NH = 82, 100, 9
for nm, x, w in COLS:
    if nm:
        tx(x + w / 2, 78, nm, 5.2, "middle")
D_FILL = {"dispatch": "#d6d6d6", "GMM1": "#fff", "ACT": "#b9b9b9",
          "GMM2": "#8e8e8e", "combine": "#ececec"}
for nm, x, w in COLS:
    if not nm:
        rect(x, YA, w, YB + NH - YA, fill="#fff", sw=0.5, dash="1.6 1.2")
        tx(x + w / 2, YA + 12, "组", 4.4, "middle")
        tx(x + w / 2, YA + 19, "就绪", 4.4, "middle")
        continue
    for k, yy in enumerate((YA, YB)):
        rect(x, yy, w, NH, fill=D_FILL[nm], sw=0.5)
        tag = ("s0", "s1")[k] if nm == "dispatch" else ("n0", "n1")[k]
        tx(x + w / 2, yy + 6, tag, 4.5, "middle",
           fill="#fff" if nm == "GMM2" else None)
CY = (YA + NH / 2, YB + NH / 2)
arr(46, CY[0], 56, YA + 8, 0.45)
arr(46, CY[1], 56, YB + 1, 0.45)
arr(70, YA + 8, 82, CY[0], 0.45)
arr(70, YB + 1, 82, CY[1], 0.45)
for k in (0, 1):
    arr(118, CY[k], 128, CY[k], 0.45)          # GMM1 -> ACT  (1:1)
    arr(210, CY[k], 220, CY[k], 0.45)          # GMM2 -> combine
for a in (0, 1):                                # ACT -> GMM2  (沿 K 汇入)
    for b in (0, 1):
        arr(164, CY[a], 174, CY[b], 0.4)
tx(169, 97.5, "沿 |K| 汇入", 4.3, "middle", fill="#333", halo=True)
# UB 槽: GMM1 取, ACT 还 -> 深度满时下一个 GMM1 要等 ACT 归还 (不是边)
curve(146, YA + NH, 124, 96, 100, YB, 0.5, dash="1.8 1.3", stroke="#555", m="ag")
tx(123, 94.6, "槽", 4.3, "middle", fill="#555", halo=True)
for x, w, core in ((10, 36, "AIV1"), (82, 36, "AIC"), (128, 36, "AIV0"),
                   (174, 36, "AIC"), (220, 36, "AIV1")):
    tx(x + w / 2, 117, core, 4.8, "middle", fill="#666")
tx(28, 123.5, "一段 = 一个源卡的一批", 4.3, "middle", fill="#666")
tx(262, 85, "实线 = 数据依赖", 4.7, fill="#444")
tx(262, 92.5, "虚线 = UB 槽: GMM1 取, ACT 还", 4.7, fill="#444")
tx(262, 100, "核在建图时就定死 (静态钉核)", 4.7, fill="#444")
tx(262, 107.5, f"此处 ×{LOCAL} 专家 × ceil(|m|~e~/TILE_M) 组", 4.7, fill="#444")
tx(262, 115, f"共 {N_EV} 事件 / {N_ED} 边", 4.7, fill="#444")


# -- 两个真事件的记录: 取/还 不对称, 所以"槽"写不成边 --
RAWBY = {e.name: e for e in RAW}


def rec(short):
    e = RAWBY["R0." + short]
    sem = lambda ts: ", ".join(t.split(":c")[0] for t, _ in ts) or "—"
    return (short.split(".S0.")[-1], e.resources[0],
            e.deps[0].split(".")[-3:] and e.deps[0], sem(e.acquires),
            sem(e.releases), f"{e.duration_us:.3f}")


tx(10, 130, "事件记录 (真值):", 5.2)
CX = (10, 76, 104, 150, 216, 288)
for i, hd in enumerate(("事件", "核", "前驱", "取", "还", "时长 µs")):
    tx(CX[i], 138, hd, 4.5, fill="#777")
ln(10, 140, 320, 140, 0.35, stroke="#aaa")
for r, short in enumerate(("W0.E0.S0.gmm1.m0.n0", "W0.E0.S0.act.m0.n0")):
    name, core, dep, acq, rel, dur = rec(short)
    dep = ("组就绪" if "dispatch_ready" in dep
           else dep.split(".S0.")[-1] if ".S0." in dep else dep)
    for i, v in enumerate((name, core, dep, acq, rel, dur)):
        tx(CX[i], 147 + r * 7, v, 4.6)
tx(326, 144, "UB:gmm1act 跨两个", 4.6, fill="#444")
tx(326, 150.5, "事件持有 → 不是边", 4.6, fill="#444")


# ===================== (b) 一个事件算多久 =====================
head(3, 166, "(b)", "一个事件算多久: 按物理性质分四类")
BX = 72
rows = [
    ("dispatch 段", 177, ["|d| = max(|T|~lat~, min(|n|,|D|)·|τ|) + max(0, |n|−|D|)·|τ|,"
                          "   |τ| = |b|~row~/|BW|"]),
    ("GMM tile", 188, ["|d| = max(|L|, |C|) 若 L1 深度 ≥ 2;   |d| = |L| + |C| + ceil(|K|/|K|~L1~)·|τ|~r~"
                       " 若深度 = 1",
                       "|L| = |A| + |B|;   GMM1: |A| = |mK|/|BW|,  |B| = |w|~b~|KN f|/|BW|~b~,"
                       "  |C| = 2|mNK|/|R|",
                       "GMM2: |A| = |mK|~2~/|BW| (仅物化),  |B| = |K|~2~|N|/|BW|~b~,"
                       "  |C| = |mNK|~2~/|R|"]),
    ("ACT tile", 214, ["|d| = |T|~0~ + (|mN|/|V|)·|β|/|BW|~UB~"]),
    ("combine tile", 224, ["|d| = |m|(|e|~in~|N| + meta)/|BW|~loc~ "
                           "+ (|m|−|r|)|e|~out~|N|/|BW|~loc~ + |r e|~out~|N|/|BW|~rmt~"]),
]
for lbl, y, fs in rows:
    tx(10, y, lbl, 5.3, fill="#333")
    for i, f in enumerate(fs):
        tx(BX, y + i * 8, f, 5.4)
tx(10, 235, "取 max 只在双缓冲真能重叠处; |A|/|B| 两股载入抢同一条 HBM→L1 通路, 故相加。"
            "|r| = 跨卡行数 (由路由精确数出)。", 4.8, fill="#444")


# ===================== (c) 排时间线 =====================
head(3, 252, "(c)", "排时间线: 先取依赖与核都就绪的时刻, 再推到槽有余量的时刻")
tx(10, 262, "|t|~base~(|e|) = max{ max~p→e~ [ end(|p|) + |λ|~pe~ ],"
            "   max~res(e)~ free(|r|) }", 5.5)
tx(10, 271, "start(|e|) = min{ |t| ≥ |t|~base~(|e|) : |e| 要占的每个槽在 |t| 都有余量 }", 5.5)
tx(258, 271, "(只在已登记的归还时刻上搜)", 4.8, fill="#444")
tx(10, 280, "end(|e|) = start(|e|) + |d|(|e|);   每步取 start 最小的事件;"
            "   |T| = 最后一个 combine 的 end (尾段照排但不计入)", 5.5)

# -- 真甘特图 (scheduler 的输出) --
GX0, GX1 = 44, 390
SC = (GX1 - GX0) / TOTAL
ROWS_R = ["R0.AIV1:0", "R0.AIC:0", "R0.AIV0:0"]
RY = {r: 290 + i * 20 for i, r in enumerate(ROWS_R)}
BH = 11
FILL = {"dispatch": "#d6d6d6", "gmm1": "#ffffff", "activation": "#b9b9b9",
        "gmm2": "#8e8e8e", "combine": "#ececec"}


def X(t):
    return GX0 + t * SC


for r in ROWS_R:
    tx(GX0 - 3, RY[r] + 7.4, r.replace("R0.", ""), 4.8, "end", fill="#444")
    ln(GX0, RY[r] + BH + 0.4, GX1, RY[r] + BH + 0.4, 0.3, stroke="#ddd")
for e in sorted(RANK["events"], key=lambda e: e.start_us):
    st = str(e.meta.get("stage", ""))
    if st not in MOE or not e.resources:
        continue
    r = e.resources[0]
    if r not in RY:
        continue
    w = (e.end_us - e.start_us) * SC
    rect(X(e.start_us), RY[r], w, BH, fill=FILL[st], sw=0.4)
    if w >= 15:
        tx(X(e.start_us) + w / 2, RY[r] + 7.2,
           {"dispatch": "disp", "gmm1": "GMM1", "activation": "ACT",
            "gmm2": "GMM2", "combine": "comb"}[st], 4.4, "middle",
           fill="#fff" if st == "gmm2" else None)

# 三种等待各标一处, 数都来自 ScheduledEvent 自己的归因字段
g0, g1 = ev("W0.E0.S0.gmm1.m0.n0"), ev("W0.E0.S0.gmm1.m0.n1")
cb = ev("W0.E0.S0.combine.m0.n1")
AY = RY["R0.AIC:0"]
rect(X(0), AY, X(g0.start_us) - X(0), BH, fill="none", sw=0.5, dash="1.5 1.1",
     stroke="#333")
tx((X(0) + X(g0.start_us)) / 2, AY + 7.2,
   f"等前驱 {g0.dependency_wait_us:.1f}", 4.3, "middle")
rect(X(g0.end_us), AY, X(g1.start_us) - X(g0.end_us), BH, fill="none", sw=0.5,
     dash="1.5 1.1", stroke="#333")
tx((X(g0.end_us) + X(g1.start_us)) / 2, AY + 7.2,
   f"等槽 {g1.capacity_wait_us:.1f}", 4.3, "middle")
BY = RY["R0.AIV1:0"] + BH + 3.4
ln(X(cb.dependency_ready_us), BY - 2, X(cb.dependency_ready_us), BY, 0.45)
ln(X(cb.start_us), BY - 2, X(cb.start_us), BY, 0.45)
ln(X(cb.dependency_ready_us), BY, X(cb.start_us), BY, 0.45)
tx(X(cb.start_us) + 2.5, BY + 1.8,
   f"等核 {cb.resource_queue_us:.1f} (核被占着, 没有空档)", 4.3, fill="#333")

# 时间轴
ln(GX0, 350, GX1, 350, 0.5)
for t in range(0, int(TOTAL) + 1, 25):
    ln(X(t), 350, X(t), 352.6, 0.45)
    tx(X(t), 357.6, str(t), 4.6, "middle")
tx(GX1, 357.6, "µs", 4.6, "end", fill="#444")
ln(X(TOTAL), 288, X(TOTAL), 350, 0.5, dash="2 1.4", stroke="#333")
tx(X(TOTAL) - 2, 287, f"|T| = {TOTAL:.1f}", 5.2, "end")

# 图例 + 实例
LX = 10
for st, lb in (("dispatch", "dispatch"), ("gmm1", "GMM1"),
               ("activation", "ACT"), ("gmm2", "GMM2"), ("combine", "combine")):
    rect(LX, 362, 7, 5.4, fill=FILL[st], sw=0.4)
    tx(LX + 9, 366.5, lb, 4.6)
    LX += 9 + len(lb) * 2.5 + 7
rect(LX, 362, 7, 5.4, fill="none", sw=0.5, dash="1.5 1.1", stroke="#333")
tx(LX + 9, 366.5, "等待", 4.6)
tx(10, 375.5, f"小实例: {LOCAL} 专家 × {ROWS * WORLD} 行, |h|={H}, "
              f"|hidden_dim|={HIDDEN}, TILE={TILE}², 1×AIC/AIV0/AIV1 "
              f"→ {N_EV} 事件 / {N_ED} 边 / |T| = {TOTAL:.2f} µs。"
              "核取 1 个是为了让三种等待同时出现。", 4.7, fill="#444")

A("</svg>")
out = Path(__file__).resolve().parents[1] / "fig-megamoe-timing.svg"
out.write_text("\n".join(o), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size} bytes)  T={TOTAL:.3f}us "
      f"events={N_EV} edges={N_ED}")
