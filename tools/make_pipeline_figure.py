# -*- coding: utf-8 -*-
"""Regenerate fig-modelling-pipeline.svg — the method figure for this cost model.

Why a generator and not a hand-edited SVG: every box in the figure names a real
layer with a real file behind it, and those paths move. Editing coordinates by
hand is how a figure drifts away from the code it claims to describe.

Panel (a) is the layering declared in implementations/adapter.py's module
docstring; panel (b) is the scheduling recurrence in scheduler/engine.py;
panel (c) is analysis/bounds.py (Bounds.lower_us / binding / BoundViolation)
and analysis/idle.py (forced vs avoidable, and avoidable being an upper bound).
"""
F = 'Times New Roman, Times, serif'
W, H = 396, 492
o = []
A = o.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" '
  f'viewBox="0 0 {W} {H}" font-family="{F}" fill="#000">')
A('''  <defs>
    <marker id="a" viewBox="0 0 8 6" refX="7.4" refY="3" markerWidth="5.4" markerHeight="4" orient="auto">
      <path d="M0 0 L8 3 L0 6 z" fill="#000"/></marker>
    <marker id="ag" viewBox="0 0 8 6" refX="7.4" refY="3" markerWidth="5" markerHeight="3.6" orient="auto">
      <path d="M0 0 L8 3 L0 6 z" fill="#666"/></marker>
  </defs>''')


def tx(x, y, s, size=6.0, anchor='start', style='normal', fill=None):
    f = f' fill="{fill}"' if fill else ''
    st = ' font-style="italic"' if style == 'italic' else ''
    A(f'  <text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{st}{f}>'
      f'{s}</text>')


def rect(x, y, w, h, fill='none', sw=0.6, dash=None, stroke='#000'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
      f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def ln(x1, y1, x2, y2, sw=0.5, dash=None, stroke='#000'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'  <path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{stroke}" '
      f'stroke-width="{sw}"{d}/>')


def arr(x1, y1, x2, y2, sw=0.6, dash=None, stroke='#000', m='a'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'  <path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{stroke}" '
      f'stroke-width="{sw}"{d} marker-end="url(#{m})"/>')


def brace(x0, x1, y, down=True, depth=4):
    s = 1 if down else -1
    ln(x0, y, x0, y + s * depth, 0.45, stroke='#777')
    ln(x0, y + s * depth, x1, y + s * depth, 0.45, stroke='#777')
    ln(x1, y + s * depth, x1, y, 0.45, stroke='#777')


# ===================== panel (a): the layers, and who knows a stage name =====
tx(2, 10, '(a)', 7.4, style='italic')
tx(14, 10, 'the layers an operator passes through.  A stage name is declared once and '
           'stops at the DAG; below it nothing reads one.', 6.5)

cols = [
    ('input facts', ['shape and workload', 'runtime: cards, topology', 'compile: binary, tiling'],
     'shape.py · runtime.py · compile.py'),
    ('adapter', ['the orchestration of', 'one implementation: wave', 'advance, which stage lands',
                 'on which core, where the', 'intermediate lives, when', 'to synchronise'],
     'implementations/adapter.py'),
    ('builder', ['expands that orchestration', 'tile by tile into events;', 'fills the five fields and',
                 'costs each one'],
     'builders/base.py + gmm1,', 'activation, gmm2, comm'),
    ('event DAG', ['the single IR; the five', 'fields are the whole', 'expressiveness ceiling'],
     'scheduler/events.py'),
    ('scheduler', ['places events on resources', 'and nothing else; reads', 'no stage name'],
     'scheduler/engine.py'),
    ('analysis', ['makespan, the three lower', 'bounds, the idle split'],
     'analysis/bounds.py, idle.py'),
]
bx0, bw, bgap, by, bh = 6, 59, 5.2, 22, 54
for k, c in enumerate(cols):
    title, lines, path = c[0], c[1], c[2]
    x = bx0 + k * (bw + bgap)
    rect(x, by, bw, bh, fill='#fafafa' if k in (1, 2) else 'none', sw=0.6)
    tx(x + bw / 2, by + 9, title, 6.5, 'middle')
    for i, s in enumerate(lines):
        tx(x + 3, by + 17.5 + i * 6.0, s, 5.1)
    tx(x + 3, by + bh - 7.5, path, 4.7, style='italic', fill='#444')
    if len(c) > 3:
        tx(x + 3, by + bh - 2.5, c[3], 4.7, style='italic', fill='#444')
    if k:
        arr(x - bgap, by + bh / 2, x - 0.8, by + bh / 2)
# who knows a stage name
brace(bx0, bx0 + 3 * bw + 2 * bgap - 0.5, by + bh + 1)
brace(bx0 + 3 * (bw + bgap), bx0 + 6 * bw + 5 * bgap, by + bh + 1)
tx(bx0 + (3 * bw + 2 * bgap) / 2, by + bh + 13, 'stage names live here only: vocabulary, '
   'wave plan, colocation rules', 5.5, 'middle')
tx(bx0 + 3 * (bw + bgap) + (3 * bw + 2 * bgap) / 2, by + bh + 13,
   'stage-name blind — one exception, PriorityByStage,', 5.5, 'middle')
tx(bx0 + 3 * (bw + bgap) + (3 * bw + 2 * bgap) / 2, by + bh + 19.5,
   'which is why it carries work_conserving = False', 5.5, 'middle', style='italic')

# builder expansion
ln(bx0 + 2 * (bw + bgap) + bw / 2, by + bh + 5, bx0 + 2 * (bw + bgap) + bw / 2, 108,
   0.5, dash='1.8 1.4', stroke='#666')
arr(bx0 + 2 * (bw + bgap) + bw / 2, 108, 200, 113, 0.5, dash='1.8 1.4', stroke='#666', m='ag')
tx(6, 118, 'what the builders emit for the MegaMoE vocabulary (five stages, one event per tile):', 6.0)
names = ['dispatch', 'gmm1', 'activation', 'gmm2', 'combine']
sub = ['peerwrite', 'cube', 'vector', 'cube', 'reduce + write']
unit = ['m-group', 'm-tile', 'm-tile', 'm-tile', 'expert']
gw, gap2, gx0, gy = 63, 10, 10, 128
for k, (nm, sb, un) in enumerate(zip(names, sub, unit)):
    gx = gx0 + k * (gw + gap2)
    rect(gx, gy, gw, 30, dash='2 1.4', stroke='#666', sw=0.5)
    tx(gx + gw / 2, gy - 3, f'{nm}', 6.2, 'middle')
    for j in range(3):
        rect(gx + 11 + j * 14, gy + 5, 11, 8, fill='#e6e6e6', sw=0.5)
    tx(gx + gw / 2, gy + 21, f'× n tiles, unit {un}', 5.3, 'middle')
    tx(gx + gw / 2, gy + 27, sb, 5.1, 'middle', style='italic', fill='#444')
    if k:
        arr(gx - gap2, gy + 9, gx - 1, gy + 9)
        tx(gx - gap2 / 2 - 0.5, gy + 6, ['m', 'm', 'm', 'exp'][k - 1], 5.2, 'middle', style='italic')
tx(198, 172, 'tile cost  d(e) = max( MAC(e) / rate_cube ,  bytes(e) / BW(link) )'
   '   — compute and movement are two costs of one event, never two boxes', 6.0, 'middle')

# ===================== panel (b): the scheduler's only job =====================
ln(0, 182, W, 182, 0.4, stroke='#bbb')
tx(2, 196, '(b)', 7.4, style='italic')
tx(14, 196, 'the scheduler reads the five fields and nothing else.  Contention, '
            'colocation and buffer credits are what make time.', 6.5)

gx0, gx1 = 54, 322
rows = [
    ('AIC 0', [('gmm1 t0', 0, 46, 'c'), ('act t0', 48, 62, 'v'), ('gmm2 t0', 64, 108, 'c'),
               ('gmm1 t3', 112, 158, 'c')]),
    ('AIC 1', [('gmm1 t1', 6, 52, 'c'), ('act t1', 54, 68, 'v'), ('gmm2 t1', 70, 114, 'c'),
               ('gmm1 t4', 118, 164, 'c')]),
    ('AIC 27', [('gmm1 t2', 12, 58, 'c'), ('act t2', 60, 74, 'v'), ('gmm2 t2', 76, 120, 'c'),
                ('idle', 124, 164, 'i')]),
    ('link', [('dispatch w0', 0, 40, 'l'), ('dispatch w1', 44, 84, 'l'), ('combine', 126, 176, 'l')]),
]
sc = (gx1 - gx0) / 184.0
y = 208
rowy = {}
for nm, bars in rows:
    rowy[nm] = y
    tx(gx0 - 4, y + 6.3, nm, 5.8, 'end')
    ln(gx0, y + 9.6, gx1 + 4, y + 9.6, 0.4, stroke='#ccc')
    for lb, s, e, kind in bars:
        bx, bwd = gx0 + s * sc, (e - s) * sc
        if kind == 'i':
            rect(bx, y, bwd, 9, dash='1.5 1.5', stroke='#999', sw=0.5)
            tx(bx + bwd / 2, y + 6.3, 'forced idle', 5.2, 'middle', style='italic', fill='#555')
        else:
            fill = {'c': '#d7d7d7', 'v': '#f0f0f0', 'l': '#ffffff'}[kind]
            rect(bx, y, bwd, 9, fill=fill, sw=0.5)
            tx(bx + bwd / 2, y + 6.3, lb, 5.2, 'middle')
    y += 13.5
tx(gx0 - 4, y + 4, 'engine', 5.2, 'end', style='italic', fill='#444')
tx(gx0, y + 4, 'cube  =  dark,   vector  =  light,   link  =  outline.   An event may hold '
   'several resources at once; t_base then takes max over all of them.', 5.5)

# time axis
ty = y + 12
ln(gx0, ty, gx1 + 14, ty, 0.5)
tx(gx0, ty + 7, '0', 5.6, 'middle')
ln(gx0 + 176 * sc, ty - 3, gx0 + 176 * sc, ty + 3, 0.5)
tx(gx0 + 176 * sc, ty + 7.5, 'makespan  T̂ = max end(e)', 5.8, 'middle')
tx(gx1 + 16, ty + 2, 'time', 5.8, style='italic')
tx(14, ty + 20, 'start(e)  =  least  t ≥  max { max', 6.2)
tx(107, ty + 22, 'p∈P(e)', 4.9)
tx(125, ty + 20, '[ end(p) + λ(e,p) ] ,   max', 6.2)
tx(205, ty + 22, 'r∈R(e)', 4.9)
tx(223, ty + 20, 'free(r) }   that the credits A(e) admit.', 6.2)
tx(14, ty + 29, 'A concrete resource takes the max of the set; a pool placeholder such as '
   'AIC:* takes the min, which is what late binding means.', 6.2)
tx(14, ty + 39, 'activation carries colocate_with, so it must take the core its gmm1 tile '
   'already holds: a free core elsewhere cannot help it, which is why', 5.7)
tx(14, ty + 46, 'pinned roles are excluded from the idle test.  The wave boundary is a credit, '
   'not an edge: dispatch w1 releases an L1 slot and gmm1 t3 is admitted.', 5.7)

# ===================== panel (c): accept or reject =====================
ln(0, 348, W, 348, 0.4, stroke='#bbb')
tx(2, 362, '(c)', 7.4, style='italic')
tx(14, 362, 'a schedule is only read off once it clears two tests that do not depend on '
            'the orchestration being modelled.', 6.5)

ax0, ax1, ay = 66, 290, 392
ln(ax0 - 6, ay, ax1 + 20, ay, 0.5)
tx(ax0 - 10, ay + 2.2, 'cost', 6.0, 'end', style='italic')
for fr, nm, note in [(0.08, 'compute_us', 'MAC / (cores · rate)'),
                     (0.34, 'bandwidth_us', 'bytes / BW_L1_GM'),
                     (0.60, 'dependency_us', 'longest d-weighted path')]:
    x = ax0 + fr * (ax1 - ax0)
    ln(x, ay - 5, x, ay + 5, 0.5)
    tx(x, ay - 8, nm, 5.7, 'middle')
    tx(x, ay + 12, note, 5.2, 'middle', style='italic', fill='#333')
xT = ax0 + 0.92 * (ax1 - ax0)
ln(xT, ay - 10, xT, ay + 5, 0.9)
tx(xT, ay - 13, 'T̂', 7.0, 'middle')
tx(xT, ay + 12, 'reported makespan', 5.2, 'middle', style='italic', fill='#333')
ln(ax0 + 0.60 * (ax1 - ax0), ay - 2.6, xT, ay - 2.6, 0.5)
tx((ax0 + 0.60 * (ax1 - ax0) + xT) / 2, ay - 4.6, 'what the orchestration costs', 5.2,
   'middle', style='italic')
tx(ax1 + 24, ay - 4, 'lower_us = max of the three.', 5.5)
tx(ax1 + 24, ay + 3, 'All three move with the shape', 5.5)
tx(ax1 + 24, ay + 10, 'alone, so T̂ under any of them', 5.5)
tx(ax1 + 24, ay + 17, 'raises BoundViolation.', 5.5)

rect(6, 414, 186, 40, dash='2.5 1.5', stroke='#666', sw=0.5)
tx(11, 424, 'idle on a pooled role, per core:', 5.9)
tx(11, 432, 'forced_idle_us  —  nothing was ready; admissible', 5.9)
tx(11, 440, 'avoidable_idle_us  —  something was ready while a', 5.9)
tx(11, 448, 'core sat free  ⇒  WorkConservationViolation', 5.9)
rect(200, 414, 190, 40, dash='2.5 1.5', stroke='#666', sw=0.5)
tx(205, 424, 'avoidable is an upper bound, not a ledger: under', 5.9)
tx(205, 432, 'phase pipelining the phase group inflates it, and', 5.9)
tx(205, 440, 'roles pinned by colocate_with or core_group are', 5.9)
tx(205, 448, 'excluded by derivation from the graph, not by hand.', 5.9)

# sweep
rect(86, 464, 224, 15, sw=0.6)
tx(198, 474, 'change exactly one knob — tile size, wave split, steal driver, binding — '
   'and re-run', 5.9, 'middle')
arr(86, 471.5, 46, 453.5, 0.6, dash='2.5 1.5', stroke='#444', m='ag')
ln(46, 471.5, 36, 471.5, 0.6, dash='2.5 1.5', stroke='#444')
ln(36, 471.5, 36, 60, 0.6, dash='2.5 1.5', stroke='#444')
arr(36, 60, 64, 49, 0.6, dash='2.5 1.5', stroke='#444', m='ag')
ln(310, 471.5, 330, 471.5, 0.6, dash='2.5 1.5', stroke='#444')
arr(330, 471.5, 330, 455, 0.6, dash='2.5 1.5', stroke='#444', m='ag')
tx(198, 488, 'ΔT̂ is the cost of that one choice: both runs are scheduled from the same '
   'declarations and pass the same two tests, so neither is fitted to the other.', 6.0, 'middle')

A('</svg>')
open('fig-modelling-pipeline.svg', 'w').write('\n'.join(o) + '\n')
