# -*- coding: utf-8 -*-
F = 'Times New Roman, Times, serif'
W, H = 396, 404
o = []
A = o.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" '
  f'viewBox="0 0 {W} {H}" font-family="{F}" fill="#000">')
A('''  <defs>
    <marker id="a" viewBox="0 0 8 6" refX="7.4" refY="3" markerWidth="5.5" markerHeight="4" orient="auto">
      <path d="M0 0 L8 3 L0 6 z" fill="#000"/></marker>
    <marker id="ag" viewBox="0 0 8 6" refX="7.4" refY="3" markerWidth="5" markerHeight="3.6" orient="auto">
      <path d="M0 0 L8 3 L0 6 z" fill="#666"/></marker>
  </defs>''')

def tx(x, y, s, size=6.2, anchor='start', style='normal', fill=None, weight='normal'):
    f = f' fill="{fill}"' if fill else ''
    w = f' font-weight="{weight}"' if weight != 'normal' else ''
    st = f' font-style="{style}"' if style != 'normal' else ''
    A(f'  <text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{st}{f}{w}>{s}</text>')

def rect(x, y, w, h, fill='none', sw=0.6, dash=None, stroke='#000'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
      f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

def arr(x1, y1, x2, y2, sw=0.6, dash=None, stroke='#000', m='a'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'  <path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{stroke}" '
      f'stroke-width="{sw}"{d} marker-end="url(#{m})"/>')

def line(x1, y1, x2, y2, sw=0.5, dash=None, stroke='#000'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'  <path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{stroke}" stroke-width="{sw}"{d}/>')

# ======================= panel (a): shape -> tile events =======================
tx(2, 11, '(a)', 7.6, style='italic')
tx(14, 11, 'one tiling choice lowers the operator into tile events; a stage is a vocabulary entry, not a box in the code', 6.6)

# declarations column
rect(4, 18, 86, 62, dash='2.5 1.5', stroke='#666', sw=0.5)
tx(47, 27, 'declared, never hard-coded', 6.0, 'middle', style='italic', fill='#333')
for i, s in enumerate(['shape  bs/rank, h, I, topk, routing',
                       'hardware  28 AIC, 2 AIV/core,',
                       '   BW_L1_GM, BW_UB, BW_REMOTE',
                       'orchestration  tile size, wave',
                       '   split, steal, late binding',
                       'stage vocabulary  per implementation']):
    tx(8, 36 + i*7.4, s, 5.9)
arr(90, 49, 110, 49, stroke='#666', m='ag')
tx(100, 46, 'lower', 5.8, 'middle', style='italic', fill='#333')

# five stages
names = ['dispatch', 'gmm1', 'activation', 'gmm2', 'combine']
sub = ['peerwrite', 'cube', 'vector', 'cube', 'reduce+write']
unit = ['m-group', 'm-tile', 'm-tile', 'm-tile', 'expert']
x0, gw, gap = 114, 48, 10
for k, (nm, sb, un) in enumerate(zip(names, sub, unit)):
    gx = x0 + k*(gw+gap)
    rect(gx, 30, gw, 38, dash='2 1.4', stroke='#666', sw=0.5)
    tx(gx+gw/2, 26, nm, 6.4, 'middle')
    tx(gx+gw/2, 65, sb, 5.6, 'middle', style='italic', fill='#333')
    # tiles
    for j in range(3):
        rect(gx+6+j*12, 38, 9, 8, fill='#e8e8e8', sw=0.5)
    tx(gx+gw/2, 57, f'× n tiles · unit {un}', 5.6, 'middle')
    if k:
        arr(gx-gap, 42, gx-1, 42)
        ax = ['m', 'm', 'm', 'expert'][k-1]
        tx(gx-gap/2-0.5, 39, ax, 5.6, 'middle', style='italic')
tx(200, 78, 'one event per tile:  d(e) = max( MAC(e)/rate_cube ,  bytes(e)/BW(link) )   — compute and movement are two costs of the same event, not two boxes',
   6.0, 'middle')

# ======================= panel (b): events contend for resources =======================
line(0, 90, W, 90, 0.4, stroke='#bbb')
tx(2, 104, '(b)', 7.6, style='italic')
tx(14, 104, 'the list scheduler places each event on a resource; contention, colocation and buffer credits are what make time, not a formula', 6.6)

gx0, gx1 = 52, 330
rows = [('AIC 0', [('gmm1 t0', 0, 46), ('gmm2 t0', 52, 96), ('gmm1 t3', 100, 150)]),
        ('AIC 1', [('gmm1 t1', 6, 52), ('gmm2 t1', 58, 104), ('gmm1 t4', 108, 158)]),
        ('AIC 27', [('gmm1 t2', 12, 58), ('gmm2 t2', 64, 110), ('idle', 114, 158)]),
        ('AIV', [('permute', 0, 30), ('act t0', 46, 62), ('act t1', 66, 82), ('act t2', 86, 102)]),
        ('link', [('dispatch w0', 0, 40), ('dispatch w1', 44, 84), ('combine', 120, 170)])]
scale = (gx1-gx0)/178.0
y = 116
for nm, bars in rows:
    tx(gx0-4, y+6.2, nm, 5.9, 'end')
    line(gx0, y+9.5, gx1, y+9.5, 0.4, stroke='#ccc')
    for lb, s, e in bars:
        bx, bw = gx0+s*scale, (e-s)*scale
        if lb == 'idle':
            rect(bx, y, bw, 9, dash='1.5 1.5', stroke='#999', sw=0.5)
            tx(bx+bw/2, y+6.2, 'forced idle', 5.4, 'middle', style='italic', fill='#555')
        else:
            fill = '#d9d9d9' if nm.startswith('AIC') else ('#efefef' if nm == 'AIV' else '#fff')
            rect(bx, y, bw, 9, fill=fill, sw=0.5)
            tx(bx+bw/2, y+6.2, lb, 5.4, 'middle')
    y += 13
# semaphore arc: dispatch w1 -> gmm1 t3
line(gx0+84*scale, 177, gx0+84*scale, 123, 0.5, dash='1.8 1.4', stroke='#444')
arr(gx0+84*scale, 123, gx0+100*scale, 121, 0.5, dash='1.8 1.4', stroke='#444')
tx(gx0+92*scale, 118, 'L1 slot released ⇒ next wave admitted', 5.5, 'middle', style='italic', fill='#333')
# time axis
line(gx0, 182, gx1+6, 182, 0.5)
tx(gx0, 189, '0', 5.6, 'middle')
line(gx1-8*scale, 179, gx1-8*scale, 185, 0.5)
tx(gx1-8*scale, 189, 'makespan  T̂ = max end(e)', 5.8, 'middle')
tx(gx1+8, 180, 'time', 5.8, style='italic')
tx(14, 199, 'start(e) = least t ≥ max{ max',  6.0)
tx(80, 201, 'p∈P(e)', 4.8)
tx(97, 199, '[ end(p)+λ(e,p) ] ,  max', 6.0)
tx(173, 201, 'r∈R(e)', 4.8)
tx(190, 199, 'free(r) } admitted by the credits A(e) still holds;  concrete resources take the max, pool placeholders the min (late binding).', 6.0)

# ======================= panel (c): what is read off, and the guardrails =======================
line(0, 208, W, 208, 0.4, stroke='#bbb')
tx(2, 222, '(c)', 7.6, style='italic')
tx(14, 222, 'the schedule is only accepted if it sits above every shape-determined bound and leaves no recoverable idle', 6.6)

# bound axis
ax0, ax1, ay = 60, 300, 248
line(ax0, ay, ax1+18, ay, 0.5)
tx(ax0-4, ay+2.2, 'cost', 6.0, 'end', style='italic')
ticks = [(0.10, 'LB', 'compute', 'MAC / (28 · rate)'),
         (0.36, 'LB', 'bandwidth', 'bytes / BW'),
         (0.62, 'LB', 'chain', 'longest d-weighted path')]
for fr, a, b, note in ticks:
    x = ax0 + fr*(ax1-ax0)
    line(x, ay-5, x, ay+5, 0.5)
    tx(x, ay-8, a, 6.0, 'middle')
    tx(x+5.5, ay-6.5, b, 4.8, 'middle')
    tx(x, ay+13, note, 5.4, 'middle', style='italic', fill='#333')
xT = ax0 + 0.90*(ax1-ax0)
line(xT, ay-9, xT, ay+5, 0.8)
tx(xT, ay-12, 'T̂', 7.0, 'middle')
tx(xT, ay+13, 'reported makespan', 5.4, 'middle', style='italic', fill='#333')
line(ax0+0.62*(ax1-ax0), ay-2.5, xT, ay-2.5, 0.5)
tx((ax0+0.62*(ax1-ax0)+xT)/2, ay-4.5, 'orchestration cost', 5.4, 'middle', style='italic')
tx(ax1+22, ay+2.2, 'the three bounds move only with the shape;', 5.6)
tx(ax1+22, ay+9, 'T̂ below any of them ⇒ BoundViolation', 5.6)

# idle decomposition
rect(4, 262, 182, 34, dash='2.5 1.5', stroke='#666', sw=0.5)
tx(10, 272, 'idle on a pooled role is split per core:', 5.9)
tx(10, 280, 'forced  (no ready work existed)  — admissible', 5.9)
tx(10, 288, 'avoidable  (ready work existed)  ⇒ WorkConservationViolation', 5.9)
rect(196, 262, 196, 34, dash='2.5 1.5', stroke='#666', sw=0.5)
tx(202, 272, 'colocation-pinned roles are excluded by derivation,', 5.9)
tx(202, 280, 'not by exception: a role carrying colocate_with or', 5.9)
tx(202, 288, 'core_group cannot take the ready tile it sees.', 5.9)

# sweep loop
rect(96, 308, 204, 18, sw=0.6)
tx(198, 319.5, 'change exactly one knob — tile size, wave split, steal driver, binding — and re-run', 6.0, 'middle')
arr(96, 317, 60, 317, 0.6, dash='2.5 1.5', stroke='#444', m='ag')
line(60, 317, 40, 317, 0.6, dash='2.5 1.5', stroke='#444')
line(40, 317, 40, 56, 0.6, dash='2.5 1.5', stroke='#444')
arr(40, 56, 46, 49, 0.6, dash='2.5 1.5', stroke='#444', m='ag')
line(300, 317, 320, 317, 0.6, dash='2.5 1.5', stroke='#444')
arr(320, 317, 320, 262, 0.6, dash='2.5 1.5', stroke='#444', m='ag')
tx(198, 334, 'ΔT̂ between two runs is the cost of that one choice; the bounds and the idle test hold for both, so the difference is comparable and not an artefact of the simulator.',
   6.0, 'middle')
tx(198, 344, 'No run is calibrated against the other: both are scheduled from the same declarations, so a win here is a win in the orchestration, not in the fit.', 6.0, 'middle')
A('</svg>')
open('fig-modelling-pipeline.svg', 'w').write('\n'.join(o) + '\n')
