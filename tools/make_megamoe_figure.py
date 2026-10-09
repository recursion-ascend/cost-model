# -*- coding: utf-8 -*-
"""生成 fig-megamoe-timing.svg: MegaMoE 性能时间是怎么算出来的.

三栏: (a) 从路由到 tile 到事件, 依赖怎么连; (b) 每个事件记哪些约束;
(c) 调度怎么排, 端到端时间怎么出来。用算子开发的说法, 不用自造术语。
"""
F = 'Times New Roman, Times, serif'
W, H = 400, 548
o = []
A = o.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" '
  f'viewBox="0 0 {W} {H}" font-family="{F}" fill="#000">')
A('''  <defs>
    <marker id="a" viewBox="0 0 8 6" refX="7.4" refY="3" markerWidth="5.2" markerHeight="3.8" orient="auto">
      <path d="M0 0 L8 3 L0 6 z" fill="#000"/></marker>
    <marker id="ag" viewBox="0 0 8 6" refX="7.4" refY="3" markerWidth="4.8" markerHeight="3.4" orient="auto">
      <path d="M0 0 L8 3 L0 6 z" fill="#666"/></marker>
  </defs>''')


def tx(x, y, s, size=6.0, anchor='start', style='normal', fill=None):
    f = f' fill="{fill}"' if fill else ''
    st = ' font-style="italic"' if style == 'italic' else ''
    A(f'  <text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{st}{f}>{s}</text>')


def rect(x, y, w, h, fill='none', sw=0.6, dash=None, stroke='#000'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
      f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def ln(x1, y1, x2, y2, sw=0.5, dash=None, stroke='#000'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'  <path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arr(x1, y1, x2, y2, sw=0.6, dash=None, stroke='#000', m='a'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'  <path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{stroke}" '
      f'stroke-width="{sw}"{d} marker-end="url(#{m})"/>')


# ===================== (a) 从路由到 tile 到事件 =====================
tx(2, 10, '(a)', 7.4, style='italic')
tx(14, 10, '路由决定每个专家收多少行, 切 tile 决定有多少份活, 一份活就是一个事件', 6.6)

tx(8, 26, '路由计数', 6.0)
rect(8, 30, 52, 24, sw=0.6)
tx(34, 40, 'C[卡][专家][源卡]', 5.4, 'middle')
tx(34, 49, '→ 每专家收到的行数', 5.2, 'middle', fill='#333')
arr(62, 42, 76, 42, stroke='#666', m='ag')

tx(78, 26, '按 TILE_M / TILE_N 切', 6.0)
gx, gy, cw, ch = 78, 30, 18, 12
for i in range(2):
    for j in range(3):
        rect(gx + j * cw, gy + i * ch, cw, ch,
             fill='#e8e8e8' if (i, j) == (0, 0) else 'none', sw=0.5)
tx(gx + cw / 2, gy + ch / 2 + 2, 'tile', 4.8, 'middle')
ln(gx, gy + 2 * ch + 3, gx + 3 * cw, gy + 2 * ch + 3, 0.4, stroke='#666')
tx(gx + 1.5 * cw, gy + 2 * ch + 10, 'n-tile 方向 (TILE_N)', 5.0, 'middle', fill='#333')
tx(gx - 2, gy + ch, 'm-group', 5.0, 'end', fill='#333')
tx(gx - 2, gy + ch + 7, '(TILE_M)', 5.0, 'end', fill='#333')

tx(142, 34, '一个 tile = 一份活 = 一个事件', 5.6)
tx(142, 42, 'GMM1 的 n-tile 数按 hidden_dim/2 算 (SwiGLU 两块权重),', 5.4, fill='#333')
tx(142, 49, 'GMM2 按 h 算, 两者不同', 5.4, fill='#333')
tx(142, 59, 'dispatch 不是一行一个事件 —— 行在软流水里重叠,', 5.4, fill='#333')
tx(142, 66, '一段连续行 (同源卡同目的) 作为一份活', 5.4, fill='#333')

tx(8, 78, '波: m-group 分批投放, 一波装几个由"每核至少领到几个 tile"反推', 6.0)
tx(8, 88, '粒度: 也可以把相邻几个 tile 合成一份活 —— 少几次同步, 但这份活只能占一个核, 且要等里面最后一个 tile 算完', 5.6, fill='#333')

# 依赖判定
tx(8, 106, '依赖边 = 同一专家、同一块行列范围, 前一阶段交给后一阶段', 6.2)
rect(20, 114, 32, 12, sw=0.6)
tx(36, 122.5, 'GMM1', 5.4, 'middle')
tx(36, 133, '行 [0,256) 列 [0,256)', 5.0, 'middle', fill='#333')
arr(54, 120, 76, 120)
rect(78, 114, 32, 12, sw=0.6)
tx(94, 122.5, 'ACT', 5.4, 'middle')
tx(94, 133, '行 [0,256) 列 [0,256)', 5.0, 'middle', fill='#333')
tx(65, 116, '✓', 6.2, 'middle')

rect(196, 114, 32, 12, sw=0.6)
tx(212, 122.5, 'GMM1', 5.4, 'middle')
tx(212, 133, '行 [256,512) 列 [0,256)', 5.0, 'middle', fill='#333')
ln(230, 120, 252, 120, 0.6, dash='2 1.6', stroke='#999')
ln(238, 116, 244, 124, 0.6)
ln(244, 116, 238, 124, 0.6)
rect(254, 114, 32, 12, sw=0.6)
tx(270, 122.5, 'ACT', 5.4, 'middle')
tx(270, 133, '行 [0,256) 列 [0,256)', 5.0, 'middle', fill='#333')
tx(241, 110, '行范围不同', 5.0, 'middle', fill='#333')
tx(292, 119, '列对上但行没对上, 是并排的两块', 5.2, fill='#333')
tx(292, 126, '数据, 不该连', 5.2, fill='#333')

tx(8, 146, 'GMM2 是唯一一个可以只等一部分的: 它沿 K 累加, 第 j 段只等覆盖这段 K 的 ACT 算完; 其余三条边都要整块对齐。', 5.6)

# ===================== (b) 每个事件记哪些约束 =====================
ln(0, 158, W, 158, 0.4, stroke='#bbb')
tx(2, 172, '(b)', 7.4, style='italic')
tx(14, 172, '每个事件记五件事, 其中两件不是"等谁算完"', 6.6)

rect(10, 180, 96, 58, sw=0.7)
tx(58, 190, '一个事件记什么', 6.2, 'middle')
for i, (a_, b_) in enumerate([('落哪个核', 'AIC / AIV0 / AIV1'),
                              ('等哪些前驱', '上一阶段同块数据'),
                              ('占几个缓冲槽', 'L1 / UB 的槽位'),
                              ('是否必须同核', 'Fixpipe 直给时'),
                              ('算多久', '见 (c) 下方四类公式')]):
    tx(16, 201 + i * 8, a_, 5.4)
    tx(58, 201 + i * 8, b_, 5.4, fill='#333')

tx(122, 190, '缓冲槽 (深度 D): GMM1 写 UB, ACT 读完才释放', 6.0)
rect(122, 196, 32, 12, sw=0.6)
tx(138, 204.5, 'GMM1', 5.4, 'middle')
rect(180, 196, 32, 12, sw=0.6)
tx(196, 204.5, 'ACT', 5.4, 'middle')
arr(156, 202, 178, 202, 0.5, dash='1.8 1.4', stroke='#444')
tx(167, 198, '槽', 5.2, 'middle')
tx(122, 217, '第 i+D 个 GMM1 要等第 i 个 ACT 把槽释放 —— 约束的两端随在飞数滑动,', 5.4, fill='#333')
tx(122, 224, '不是固定的一对事件, 所以不能写成依赖边。D=1 就是没有双缓冲。', 5.4, fill='#333')
tx(122, 234, '必须同核: GMM1 的结果经 Fixpipe 直给配对的 AIV0, 这条通路只在绑定对内存在,', 5.4, fill='#333')
tx(122, 241, '所以 ACT 的核号被 GMM1 钉住, 不是可以挑的。', 5.4, fill='#333')

# ===================== (c) 调度与端到端时间 =====================
ln(0, 252, W, 252, 0.4, stroke='#bbb')
tx(2, 266, '(c)', 7.4, style='italic')
tx(14, 266, '每个事件排在哪: 前驱算完、核空出来、槽也空出来, 三个条件里最晚的那个', 6.6)

rx0, rx1 = 62, 348
sc = (rx1 - rx0) / 170.0
rows = [('AIV1', [('dispatch 段', 0, 26, 'run'), ('combine', 132, 164, 'run')]),
        ('AIC',  [('GMM1 tile', 26, 76, 'run'), ('GMM1 tile', 96, 146, 'run')]),
        ('AIV0', [('ACT', 76, 92, 'run')]),
        ('AIC',  [('GMM2 tile', 92, 132, 'run')])]
y = 280
rowy = {}
for k, (nm, bars) in enumerate(rows):
    rowy[k] = y
    tx(rx0 - 5, y + 6.4, nm, 5.8, 'end')
    ln(rx0, y + 9.6, rx1, y + 9.6, 0.4, stroke='#ccc')
    for lb, s, e, kind in bars:
        bx, bw = rx0 + s * sc, (e - s) * sc
        rect(bx, y, bw, 9, fill='#dcdcdc', sw=0.5)
        tx(bx + bw / 2, y + 6.3, lb, 5.2, 'middle')
    y += 13.5

# 等槽
wx0, wx1 = rx0 + 76 * sc, rx0 + 96 * sc
rect(wx0, rowy[1], wx1 - wx0, 9, dash='1.5 1.5', stroke='#999', sw=0.5)
tx((wx0 + wx1) / 2, rowy[1] + 6.3, '等槽', 4.8, 'middle', style='italic', fill='#555')
ln(rx0 + 92 * sc, rowy[2] + 9.6, rx0 + 92 * sc, rowy[1] + 9.6, 0.5, dash='1.6 1.4', stroke='#444')
arr(rx0 + 92 * sc, rowy[1] + 9.6, rx0 + 96 * sc, rowy[1] + 6, 0.5, dash='1.6 1.4', stroke='#444')
tx(rx0 + 98 * sc, rowy[1] + 20, 'ACT 读完释放槽, 下一个 GMM1 才能开始', 5.2, fill='#333')

# 同核
ln(rx0 + 78 * sc, rowy[1] + 9.6, rx0 + 78 * sc, rowy[2], 0.5, stroke='#444')
tx(rx0 + 80 * sc, rowy[2] - 2, '同核 (Fixpipe)', 5.0, fill='#333')

# 端到端
ln(rx0 + 164 * sc, 276, rx0 + 164 * sc, rowy[3] + 14, 0.7)
tx(rx0 + 164 * sc + 2, 274, '端到端耗时 = 最后一个 combine 结束', 5.8)
ln(rx0, rowy[3] + 16, rx1 + 6, rowy[3] + 16, 0.5)
tx(rx0, rowy[3] + 23, '0', 5.4, 'middle')
tx(rx1 + 8, rowy[3] + 18, '时间', 5.6, style='italic')

tx(14, 362, '排的办法: 对每个还没排的事件, 算出它最早能开始的时刻 = max(前驱都算完, 它要的核空出来); 如果它还要占缓冲槽,', 5.8)
tx(14, 370, '再推迟到槽空出来的那一刻。所有能开始的事件里, 谁最早就先排谁, 一个一个排完整张图。', 5.8)
tx(14, 380, '每个事件被挡的时间分成等前驱 / 等核 / 等槽三份, 加起来就是它开始的时刻 —— 逐阶段汇总就能看出瓶颈是依赖链太长、', 5.8)
tx(14, 388, '核不够, 还是缓冲太浅, 这三件事的改法完全不同。', 5.8)

# 四类时长
tx(14, 406, '一份活算多久: 按物理性质分四类', 6.4)
rect(14, 412, 372, 50, dash='2.5 1.5', stroke='#666', sw=0.5)
rows2 = [('dispatch 一段',
          'max(一次往返延迟, 槽内重叠的行的字节时间) + 超出槽数的行按每行节拍串上去'),
         ('GMM1 / GMM2 一个 tile',
          '双缓冲取 max(载入, 计算); 单缓冲是载入 + 计算 + 每个 K 块的重启'),
         ('ACT 一个 tile',
          '向量启动开销 + 元素数/向量宽度 x 每向量的 UB 字节 / UB 带宽'),
         ('combine 一个 tile',
          '读回 + 本卡写 + 跨卡写 (两条带宽差一个量级, 必须分开) + 落点跨度项')]
yy = 422
for a_, b_ in rows2:
    tx(20, yy, a_, 5.6)
    tx(104, yy, b_, 5.6, fill='#333')
    yy += 9
tx(20, 458, '载入与计算取 max 不是相加, 是因为它们重叠; 但 GMM 的两股载入 (激活 + 权重) 之间是相加, '
   '这一条由对照实验定的。', 5.4, fill='#333')

# MegaMoE 的具体选择
tx(14, 478, 'MegaMoE 在这套算法里的具体选择', 6.4)
rect(14, 484, 372, 48, dash='2.5 1.5', stroke='#666', sw=0.5)
tx(20, 494, '五段 dispatch → GMM1 → ACT → GMM2 → combine; GMM1/GMM2 上 AIC, ACT 上 AIV0, '
   'dispatch 与 combine 上 AIV1。', 5.6)
tx(20, 503, 'GMM1→ACT 这条边既占 UB 槽又必须同核 (Fixpipe 直给); ACT→GMM2 缺省经显存往返, '
   '所以 GMM2 的载入要算激活那一股。', 5.6)
tx(20, 512, 'tile 到核的分配在建图时就定死 (编译期按游标分块), 不是运行时抢 —— 换成运行时抢是另一种实现, '
   '要另付一次原子操作。', 5.6)
tx(20, 521, '这些选择换一个取值就重跑一次, 两次的时间差就是这个选择的代价。', 5.6)
tx(20, 530, '', 5.6)

A('</svg>')
open('fig-megamoe-timing.svg', 'w').write('\n'.join(o) + '\n')
