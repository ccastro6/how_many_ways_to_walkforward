"""Build every figure of the paper as an HTML page with inline SVG (figures/html/).

Run from the replication folder:
    python figures/build_figures.py && node figures/render.js
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from svgplot import (AQUA, AXIS, BLUE, GRID, INK, INK2, INK3, ORANGE, SURFACE, VIOLET, YELLOW,  # noqa: E402
                     MAGENTA, Panel, esc, legend_html, nice_ticks, page)

W = 624                      # 6.5 in at 96 px/in = LaTeX \textwidth
EMP = ROOT / "results" / "empirical"
MC = ROOT / "results" / "mc"
OUT = HERE / "html"
SCHEME_COLOR = {"rolling": BLUE, "growing": AQUA, "expanding": VIOLET}
SCHEME_NAME = {"rolling": "Rolling", "growing": "Growing-rolling", "expanding": "Expanding"}


def save(name, body, width=W):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.html").write_text(page(body, width, name))
    print("wrote", name)


def svg_wrap(w, h, inner):
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{inner}</svg>'


# ----------------------------------------------------------------- Figure 1
def _bracket(x0, x1, y, text, color=INK2, up=True):
    """Horizontal dimension bracket with a centred label (SVG string)."""
    d = -4 if up else 4
    return (f'<path d="M{x0:.1f},{y + d:.1f} V{y:.1f} H{x1:.1f} V{y + d:.1f}" fill="none" stroke="{color}" stroke-width="1"/>'
            f'<text class="label" x="{(x0 + x1) / 2:.1f}" y="{y - 6 if up else y + 14:.1f}" text-anchor="middle" '
            f'style="font-style:italic;font-family:Stix,serif;font-size:11px">{text}</text>')


def fig_design():
    """The three schemes of the unified family on a common time axis."""
    parts = []
    LABEL_W = 124
    x0, x1 = LABEL_W + 6, W - 10
    T = 100.0
    sx = lambda v: x0 + v / T * (x1 - x0)
    origin, s, h, nblk = 0.0, 40.0, 12.0, 4
    schemes = [
        ("Rolling", "fixed length L", dict(L0=26, G=0, g=0, exp=False)),
        ("Growing-rolling", "starts at L, gains G per block", dict(L0=18, G=4, g=0, exp=False)),
        ("Expanding", "start fixed at the origin", dict(L0=None, G=0, g=0, exp=True)),
        ("Rolling with gap", "g days left out before each test", dict(L0=26, G=0, g=4, exp=False)),
    ]
    BAR, ROW, GAP_ROWS = 7, 10, 18
    parts.append('<text class="ptitle" x="0" y="12">One family, three schemes</text>')
    parts.append('<text class="psub" x="0" y="25">Block b trains on the blue window and trades the chosen rule, unchanged, over the orange test block. '
                 'The test blocks are the same for every design.</text>')
    top = 50
    # vertical guides at the common block boundaries
    y_axis = top + len(schemes) * (nblk * ROW + GAP_ROWS) + 2
    for b in range(nblk + 1):
        xx = sx(s + b * h)
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top - 6}" y2="{y_axis:.1f}" stroke="{GRID}" stroke-dasharray="2 3"/>')
    y = top
    for si, (name, sub, pr) in enumerate(schemes):
        # light row band
        band_h = nblk * ROW + 6
        parts.append(f'<rect x="0" y="{y - 4}" width="{W}" height="{band_h}" fill="{"#f6f6f3" if si % 2 == 0 else SURFACE}"/>')
        parts.append(f'<text class="label" x="0" y="{y + 12}" style="font-weight:600;font-size:10.5px">{esc(name)}</text>')
        parts.append(f'<text class="label2" x="0" y="{y + 24}">{esc(sub)}</text>')
        for b in range(nblk):
            ts, te = s + b * h, s + (b + 1) * h
            tre = ts - pr["g"]
            trs = origin if pr["exp"] else max(origin, tre - (pr["L0"] + pr["G"] * b))
            yy = y + b * ROW
            parts.append(f'<rect x="{sx(trs):.1f}" y="{yy}" width="{sx(tre) - sx(trs):.1f}" height="{BAR}" rx="1.5" fill="#9ec5f4"/>')
            if pr["g"]:
                parts.append(f'<rect x="{sx(tre):.1f}" y="{yy}" width="{sx(ts) - sx(tre):.1f}" height="{BAR}" fill="url(#hatch)"/>')
            parts.append(f'<rect x="{sx(ts) + 0.7:.1f}" y="{yy}" width="{sx(te) - sx(ts) - 1.4:.1f}" height="{BAR}" rx="1.5" fill="{ORANGE}"/>')
            parts.append(f'<text class="tick" x="{sx(te) + 5:.1f}" y="{yy + BAR - 0.5:.1f}" style="font-size:9px;fill:{INK3}">b = {b}</text>')
        # dimension brackets on the first block of the first and last schemes
        if si == 0:
            yb = y - 7
            parts.append(_bracket(sx(s - pr["L0"]), sx(s), yb, "L"))
            parts.append(_bracket(sx(s), sx(s + h), yb, "h"))
        if pr["g"]:
            parts.append(_bracket(sx(s - pr["g"]), sx(s), y - 7, "g"))
        y += nblk * ROW + GAP_ROWS
    parts.append(f'<line x1="{x0}" x2="{x1}" y1="{y_axis:.1f}" y2="{y_axis:.1f}" stroke="{AXIS}"/>')
    ticks = [(origin, "origin o"), (s, "first test day s"), (s + h, "s + h"), (s + 2 * h, "s + 2h"), (s + 3 * h, "s + 3h"), (s + 4 * h, "s + 4h")]
    for v, lab in ticks:
        xx = sx(v)
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y_axis:.1f}" y2="{y_axis + 4:.1f}" stroke="{AXIS}"/>')
        anchor = "start" if v == origin else "middle"
        parts.append(f'<text class="tick" x="{xx:.1f}" y="{y_axis + 15:.1f}" text-anchor="{anchor}">{esc(lab)}</text>')
    parts.append(f'<text class="axlabel" x="{x1:.1f}" y="{y_axis + 30:.1f}" text-anchor="end">trading days →</text>')
    ya_end = y_axis + 36

    Ht = ya_end + 4
    defs = ('<defs><pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="4" height="4" fill="{SURFACE}"/><line x1="0" y1="0" x2="0" y2="4" stroke="{INK3}" stroke-width="1.2"/></pattern></defs>')
    legend = legend_html([("training window", "#9ec5f4", "wash"), ("test block", ORANGE, "wash"),
                          ("gap (embargo)", INK3, "wash")])
    save("fig1_design", legend + svg_wrap(W, int(Ht), defs + "".join(parts)))
# ----------------------------------------------------------------- empirical
def fig_us_fan():
    cur = pd.read_csv(EMP / "curves_us_1926.csv")
    des = pd.read_csv(EMP / "designs_us_1926.csv")
    months = pd.PeriodIndex(cur["month"], freq="M")
    years = np.asarray(months.year + (months.month - 0.5) / 12, dtype=float)
    labels = list(des["label"])
    scheme = dict(zip(des["label"], des["scheme"]))
    log10 = math.log(10)
    pw, ph, gap = (W - 56 - 34) / 2, 240, 34
    svg = []
    for j, (suffix, tag, sub) in enumerate(((" net", "(a) After 10 bp per unit traded", "growth of $1 invested above T-bills, net of costs"),
                                            ("", "(b) Before costs", "same, gross"))):
        cols = [lab + suffix for lab in labels]
        Y = cur[cols].to_numpy() / log10
        bh = cur["buy_hold"].to_numpy() / log10
        lo = math.floor(min(Y.min(), bh.min()))
        hi = math.ceil(max(Y.max(), bh.max()))
        p = Panel(52 + j * (pw + gap), 40, pw, ph)
        p.set((years.min(), years.max()), (lo, hi))
        step = 1 if hi - lo <= 6 else 2
        ticks = list(range(lo, hi + 1, step))
        def fmt(v):
            x = 10 ** v
            if x < 1:
                return f"{x:g}"
            if x < 10_000:
                return f"{x:,.0f}"
            if x < 1e6:
                return f"{x / 1e3:,.0f}K"
            if x < 1e9:
                return f"{x / 1e6:,.0f}M"
            return f"{x / 1e9:,.0f}B"
        p.grid_y(ticks, fmt=fmt, label="dollars, log scale" if j == 0 else None)
        p.band(years.min(), 1982, lo, hi, "#8a8984", 0.07)
        p.hline(0, INK2, dash="4 3")
        p.text((years.min() + 1982) / 2, lo, "before index futures (1982)", cls="label2 halo", anchor="middle", dy=-6)
        order = sorted(range(len(labels)), key=lambda k: {"growing": 0, "expanding": 1, "rolling": 2}[scheme[labels[k]]])
        for k in order:
            p.line(years, Y[:, k], SCHEME_COLOR[scheme[labels[k]]], width=0.8, opacity=0.6)
        p.line(years, bh, INK, width=2.0)
        p.axis_x([1940, 1960, 1980, 2000, 2020])
        p.title(tag, sub)
        svg.append(p.svg())
    legend = legend_html([("buy-and-hold (US market)", INK, "line"), ("rolling designs", BLUE, "line"),
                          ("growing-rolling designs", AQUA, "line"), ("expanding designs", VIOLET, "line"),
                          ("$1 (break-even)", INK2, "dash")])
    save("fig_us_fan", legend + svg_wrap(W, 40 + ph + 24, "".join(svg)))


def _beeswarm(vals, ys, r=3.0, max_off=60.0):
    """Greedy beeswarm: (x offset, y) in px for points of radius r, kept within +/- max_off.

    Points that cannot be placed without overlap in their own row are nudged up or down by
    a fraction of a radius, so a column of 56 identical values forms a compact block."""
    order = np.argsort(vals)
    placed, out = [], [None] * len(vals)
    step = 2 * r * 0.95
    M = int(max_off // step)
    for k in order:
        y0 = ys(vals[k])
        best, best_d = None, -1.0
        for dyi in range(0, 7):
            for sdy in ((0,) if dyi == 0 else (1, -1)):
                y = y0 + sdy * dyi * r * 0.9
                for m in range(0, M + 1):
                    for sgn in ((0,) if m == 0 else (1, -1)):
                        x = sgn * m * step
                        d = min(((x - px) ** 2 + (y - py) ** 2) ** 0.5 for px, py in placed) if placed else 1e9
                        if d >= 2 * r * 0.98:
                            out[k] = (x, y); placed.append((x, y)); break
                        if d > best_d:
                            best, best_d = (x, y), d
                    if out[k] is not None: break
                if out[k] is not None: break
            if out[k] is not None: break
        if out[k] is None:
            out[k] = best; placed.append(best)
    return out


def fig_eras():
    e = pd.read_csv(EMP / "eras.csv")
    es = pd.read_csv(EMP / "eras_summary.csv")
    eras = ["1930-1959", "1960-1989", "1990-2026"]
    LEFT, GAP, TOP, PH = 132, 16, 48, 236
    pw = (W - LEFT - 8 - 2 * GAP) / 3
    lo = math.floor(e[["t", "t_net"]].min().min() - 0.5)
    hi = math.ceil(e[["t", "t_net"]].max().max() + 0.5)
    parts = []
    stat_rows = [("designs with t > 1.645 (of 56)", []), ("average-t p-value", [])]
    for i, era in enumerate(eras):
        p = Panel(LEFT + i * (pw + GAP), TOP, pw, PH).set((0, 2), (lo, hi))
        p.raw(f'<rect x="{p.x:.1f}" y="{p.y:.1f}" width="{p.w:.1f}" height="{p.h:.1f}" fill="#f9f9f7"/>')
        p.grid_y([t for t in range(-8, 9, 2) if lo <= t <= hi],
                 fmt=(lambda t: f"{t:g}".replace("-", "−")) if i == 0 else (lambda t: ""),
                 label="Newey-West t-statistic vs buy-and-hold" if i == 0 else None)
        p.hline(0, AXIS)
        p.hline(1.645, INK2, dash="4 3", label="t = 1.645" if i == 2 else None, label_side="right")
        sub = e[e["era"] == era].reset_index(drop=True)
        for j, (col, tag) in enumerate((("t", "gross"), ("t_net", "net"))):
            vals = sub[col].to_numpy()
            cx = j + 0.5
            pos = _beeswarm(vals, p.ys, r=3.0, max_off=p.w / 4 - 5)
            for k in range(len(vals)):
                x_px, y_px = pos[k]
                col = SCHEME_COLOR[sub.loc[k, "scheme"]]
                p.raw(f'<circle cx="{p.xs(cx) + x_px:.2f}" cy="{y_px:.2f}" r="3" fill="{col}" stroke="{SURFACE}" stroke-width="1.2"/>')
            st = es[(es["era"] == era) & (es["costs"] == tag)].iloc[0]
            pv = "<0.001" if st["p_avgt"] < 0.001 else f"{st['p_avgt']:.3f}"
            stat_rows[0][1].append((p.xs(cx), f"{int(st['n_sig_raw'])}", st["n_sig_raw"] > 0))
            stat_rows[1][1].append((p.xs(cx), pv, st["p_avgt"] < 0.05))
        p.axis_x([0.5, 1.5], labels=["before costs", "after costs"])
        p.raw(f'<text class="ptitle" x="{p.x + p.w / 2:.1f}" y="{p.y - 10:.1f}" text-anchor="middle">{era.replace("-", "–")}</text>')
        parts.append(p.svg())
    # statistics strip under the panels, with row headers at the left so every number is labelled
    ys0 = TOP + PH + 40
    parts.append(f'<line x1="{LEFT}" x2="{W - 8}" y1="{ys0 - 13}" y2="{ys0 - 13}" stroke="{GRID}"/>')
    for rI, (head, cells) in enumerate(stat_rows):
        yy = ys0 + rI * 15
        parts.append(f'<text class="label2" x="0" y="{yy}" text-anchor="start">{esc(head)}</text>')
        for xx, txt, strong in cells:
            w = "600" if strong else "400"
            parts.append(f'<text class="label" x="{xx:.1f}" y="{yy}" text-anchor="middle" style="font-weight:{w}">{esc(txt)}</text>')
    legend = legend_html([("rolling design", BLUE, "circle"), ("growing-rolling design", AQUA, "circle"),
                          ("expanding design", VIOLET, "circle"), ("one dot = one design", INK3, "ring")])
    save("fig_eras", legend + svg_wrap(W, ys0 + 30, "".join(parts)))



# ----------------------------------------------------------------- dependence
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]   # sequential blue, steps 100-700


def _ramp(u):
    """Colour of the sequential blue ramp at u in [0, 1] (linear interpolation between steps)."""
    u = min(max(u, 0.0), 1.0) * (len(BLUE_RAMP) - 1)
    i = min(int(u), len(BLUE_RAMP) - 2)
    f = u - i
    a, b = BLUE_RAMP[i], BLUE_RAMP[i + 1]
    rgb = [round(int(a[k:k + 2], 16) * (1 - f) + int(b[k:k + 2], 16) * f) for k in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(rgb)


def fig_corr():
    f = MC / "corr_t_garch_t.csv"
    if not f.exists():
        print("skip fig_corr (no MC results)")
        return
    from wfo.splits import default_grid
    grid = default_grid()
    labels = [c.label for c in grid]
    C = pd.read_csv(f, index_col=0).loc[labels, labels].to_numpy()
    dep = pd.read_csv(MC / "summary_dependence.csv")
    dep = dep[(dep["dgp"] == "garch_t")].iloc[0]
    K = len(labels)
    lo = math.floor(C[np.triu_indices(K, 1)].min() * 20) / 20
    LEFT, TOP, SIZE = 150, 34, 400
    cell = SIZE / K
    parts = []
    parts.append(f'<text class="ptitle" x="0" y="12">Correlation of the 56 design t-statistics across simulated markets</text>')
    parts.append(f'<text class="psub" x="0" y="25">GARCH null, benchmark buy-and-hold. Mean pairwise correlation {dep["mean_corr_t"]:.2f}, minimum {dep["min_corr_t"]:.2f}; '
                 f'with cash as the benchmark: {dep["mean_corr_t_cash"]:.2f} and {dep["min_corr_t_cash"]:.2f}.</text>')
    for i in range(K):
        for j in range(K):
            u = (C[i, j] - lo) / (1 - lo)
            parts.append(f'<rect x="{LEFT + j * cell:.2f}" y="{TOP + i * cell:.2f}" width="{cell + 0.3:.2f}" height="{cell + 0.3:.2f}" fill="{_ramp(u)}"/>')
    # group separators and labels: 4 blocks of h (14 designs each), each split by g (7 each)
    hs = [c.h for c in grid]
    gs = [c.g for c in grid]
    for k in range(1, K):
        if hs[k] != hs[k - 1]:
            y = TOP + k * cell
            parts.append(f'<line x1="{LEFT}" x2="{LEFT + SIZE}" y1="{y:.1f}" y2="{y:.1f}" stroke="{SURFACE}" stroke-width="1.6"/>')
            parts.append(f'<line y1="{TOP}" y2="{TOP + SIZE}" x1="{y - TOP + LEFT:.1f}" x2="{y - TOP + LEFT:.1f}" stroke="{SURFACE}" stroke-width="1.6"/>')
        elif gs[k] != gs[k - 1]:
            y = TOP + k * cell
            parts.append(f'<line x1="{LEFT}" x2="{LEFT + SIZE}" y1="{y:.1f}" y2="{y:.1f}" stroke="{SURFACE}" stroke-width="0.8" stroke-dasharray="2 2"/>')
            parts.append(f'<line y1="{TOP}" y2="{TOP + SIZE}" x1="{y - TOP + LEFT:.1f}" x2="{y - TOP + LEFT:.1f}" stroke="{SURFACE}" stroke-width="0.8" stroke-dasharray="2 2"/>')
    # block labels
    k0 = 0
    for k in range(1, K + 1):
        if k == K or hs[k] != hs[k0]:
            yc = TOP + (k0 + k) / 2 * cell
            xc = LEFT + (k0 + k) / 2 * cell
            parts.append(f'<text class="label" x="{LEFT - 8}" y="{yc + 4:.1f}" text-anchor="end">h = {hs[k0]}</text>')
            parts.append(f'<text class="label" x="{xc:.1f}" y="{TOP + SIZE + 14}" text-anchor="middle">h = {hs[k0]}</text>')
            # g sub-labels on the left
            parts.append(f'<text class="tick" x="{LEFT - 8}" y="{TOP + (k0 + 3.5) * cell + 3:.1f}" text-anchor="end" style="fill:{INK3}">g = 0</text>')
            parts.append(f'<text class="tick" x="{LEFT - 8}" y="{TOP + (k0 + 10.5) * cell + 3:.1f}" text-anchor="end" style="fill:{INK3}">g = 21</text>')
            k0 = k
    parts.append(f'<text class="label2" x="0" y="{TOP + 10}">within each block:</text>')
    parts.append(f'<text class="label2" x="0" y="{TOP + 22}">R252, R504, R1008,</text>')
    parts.append(f'<text class="label2" x="0" y="{TOP + 34}">G252, G504, G1008,</text>')
    parts.append(f'<text class="label2" x="0" y="{TOP + 46}">EXP (Table 2)</text>')
    parts.append(f'<text class="axlabel" x="{LEFT + SIZE / 2:.1f}" y="{TOP + SIZE + 30}" text-anchor="middle">design (test length h, gap g, scheme and training length)</text>')
    # colour bar
    bx, bw = LEFT + SIZE + 22, 12
    nseg = 40
    for q in range(nseg):
        u0 = q / nseg
        y = TOP + SIZE - (q + 1) / nseg * SIZE
        parts.append(f'<rect x="{bx}" y="{y:.2f}" width="{bw}" height="{SIZE / nseg + 0.3:.2f}" fill="{_ramp(u0)}"/>')
    for v in nice_ticks(lo, 1.0, 5):
        if v < lo - 1e-9 or v > 1 + 1e-9:
            continue
        y = TOP + SIZE - (v - lo) / (1 - lo) * SIZE
        parts.append(f'<line x1="{bx + bw}" x2="{bx + bw + 3}" y1="{y:.1f}" y2="{y:.1f}" stroke="{AXIS}"/>')
        parts.append(f'<text class="tick" x="{bx + bw + 6}" y="{y + 3.5:.1f}">{v:.2f}</text>')
    parts.append(f'<text class="label2" x="{bx}" y="{TOP - 6}">correlation</text>')
    save("fig_corr", svg_wrap(W, TOP + SIZE + 40, "".join(parts)))

# ----------------------------------------------------------------- Monte Carlo
PROC = [  # key, label, color, shape, dash
    ("best_naive_cash", "best of K, naive t > 1.645, against cash", MAGENTA, "square", None),
    ("best_naive", "best of K, naive, against buy-and-hold", ORANGE, "circle", None),
    ("boot_max", "bootstrap max-t (Reality Check)", AQUA, "diamond", None),
    ("boot_avg", "bootstrap average-t", BLUE, "circle", None),
]
PROC_POWER = [
    ("best_naive_sizeadj", "best of 56, size-adjusted (oracle critical value)", ORANGE, "circle"),
    ("boot_max", "bootstrap max-t (Reality Check)", AQUA, "diamond"),
    ("boot_avg", "bootstrap average-t", BLUE, "circle"),
]


def fig_size():
    if not (MC / "summary_procedures.csv").exists():
        print("skip fig_size (no MC results)")
        return
    sp = pd.read_csv(MC / "summary_procedures.csv")
    sd = pd.read_csv(MC / "summary_by_design.csv")
    parts = []
    # panel (a): per-design size, garch null
    pa = Panel(52, 44, 200, 230)
    nulls = sd[sd["dgp"] == "garch_t"].copy()
    g_all = sp[(sp["dgp"] == "garch_t") & (sp["K"] <= 56) & sp["procedure"].isin([k for k, *_ in PROC])]
    ymax = max(0.20, math.ceil(g_all["reject"].max() * 10) / 10)   # axis top, in steps of 10%
    lo = min(0.03, math.floor(nulls["reject"].min() * 100) / 100)
    hi = max(0.07, math.ceil(nulls["reject"].max() * 100) / 100)
    pa.set((0, 4), (lo, hi))
    ticks_a = [round(lo + 0.01 * i, 2) for i in range(int(round((hi - lo) / 0.01)) + 1)]
    pa.grid_y(ticks_a, fmt=lambda v: f"{v * 100:.0f}%", label="false-positive rate")
    reps = int(sp[sp["dgp"] == "garch_t"]["reps"].iloc[0]) if "reps" in sp else 5000
    se = math.sqrt(0.05 * 0.95 / reps)
    pa.band(0, 4, 0.05 - 1.96 * se, 0.05 + 1.96 * se, INK3, 0.18)
    pa.hline(0.05, INK2, dash="4 3")
    hs = sorted(nulls["h"].unique())
    for i, h in enumerate(hs):
        sub = nulls[nulls["h"] == h].reset_index(drop=True)
        for k in range(len(sub)):
            jx = i + 0.5 + ((k % 7) - 3) * 0.07
            pa.dot(jx, sub.loc[k, "reject"], SCHEME_COLOR[sub.loc[k, "scheme"]], r=2.8)
    pa.axis_x([0.5 + i for i in range(len(hs))], labels=[f"{h}" for h in hs], label="test-block length h (days)")
    for i, (nm, col) in enumerate((("rolling", BLUE), ("growing-rolling", AQUA), ("expanding", VIOLET))):
        yy = hi - 0.0015 - i * 0.003
        pa.dot(0.25, yy, col, r=2.8)
        pa.text(0.25, yy, nm, cls="label2", anchor="start", dx=7, dy=3.5)
    pa.title("(a) Each design fixed in advance", f"GARCH null; one dot per design; axis {lo * 100:.0f}–{hi * 100:.0f}%")
    parts.append(pa.svg())
    # panel (b): FPR vs K
    pb = Panel(318, 44, W - 318 - 120, 230)
    g = sp[(sp["dgp"] == "garch_t")]
    Ks = sorted(g["K"].unique())
    Ks = [k for k in Ks if k <= 56]
    pb.set((math.log(1), math.log(56)), (0, ymax))
    pb.grid_y([0.10 * i for i in range(int(round(ymax / 0.10)) + 1)], fmt=lambda v: f"{v * 100:.0f}%")
    pb.band(0, math.log(56), 0.05 - 1.96 * se, 0.05 + 1.96 * se, INK3, 0.18)
    pb.hline(0.05, INK2, dash="4 3")
    ends = []
    for key, lab, col, shp, _ in PROC:
        s = g[g["procedure"] == key].sort_values("K")
        s = s[s["K"].isin(Ks)]
        if s.empty:
            continue
        xs = np.log(s["K"].to_numpy())
        pb.line(xs, s["reject"].to_numpy(), col, 2)
        pb.dots(xs, s["reject"].to_numpy(), col, 3.6, shp)
        ends.append([pb.ys(s["reject"].iloc[-1]), f"{s['reject'].iloc[-1] * 100:.1f}%", col])
    # end labels with a leader when they would collide (min 11 px apart)
    ends.sort(key=lambda e: e[0])
    placed = [e[0] for e in ends]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + 11)
    over = placed[-1] - (pb.y + pb.h + 4) if placed else 0
    if over > 0:
        placed = [y - over for y in placed]
    xe = pb.xs(math.log(56))
    for (y0, txt, col), y1 in zip(ends, placed):
        if abs(y1 - y0) > 1:
            pb.raw(f'<line x1="{xe + 5:.1f}" y1="{y0:.1f}" x2="{xe + 12:.1f}" y2="{y1:.1f}" stroke="{AXIS}" stroke-width="1"/>')
        pb.raw(f'<text class="label" x="{xe + 14:.1f}" y="{y1 + 3.5:.1f}">{txt}</text>')
    pb.axis_x([math.log(k) for k in Ks], labels=[str(k) for k in Ks], label="number of designs tried, K")
    pb.title("(b) Reporting the best of K designs", "GARCH null; bootstrap tests shown against buy-and-hold")
    parts.append(pb.svg())
    legend = legend_html([(lab, col, shp) for _, lab, col, shp, _ in PROC] +
                         [("nominal level 5%", INK2, "dash"), ("Monte Carlo 95% band around 5%", "#d6d5d0", "wash")])
    save("fig_size", legend + svg_wrap(W, 44 + 230 + 40, "".join(parts)))


def fig_power():
    if not (MC / "summary_procedures.csv").exists():
        print("skip fig_power (no MC results)")
        return
    sp = pd.read_csv(MC / "summary_procedures.csv")
    sd = pd.read_csv(MC / "summary_by_design.csv")
    parts = []
    specs = [("ar1", "(a) Short-run momentum: AR(1) coefficient φ", lambda v: f"{v:.2f}", "φ"),
             ("regime", "(b) Persistent trends: annual drift μ", lambda v: f"{v * 100:.0f}%", "μ")]
    pw = (W - 60 - 40) / 2
    N_OOS = 4013
    for f in MC.glob("meta_*.json"):
        N_OOS = int(json.loads(f.read_text())["n_oos"])
        break
    for j, (dgp, title, fmt, sym) in enumerate(specs):
        p = Panel(52 + j * (pw + 40), 44, pw, 220)
        null = sp[(sp["dgp"] == "garch_t") & (sp["K"] == 56)].copy()
        null["param"] = 0.0
        s = pd.concat([null, sp[(sp["dgp"] == dgp) & (sp["K"] == 56)]])
        params = sorted(s["param"].unique())
        p.set((params[0], params[-1]), (0, 1))
        p.grid_y([0, 0.2, 0.4, 0.6, 0.8, 1.0], fmt=lambda v: f"{v * 100:.0f}%", label="rejection rate" if j == 0 else None)
        p.hline(0.05, INK2, dash="4 3")
        # single-design envelope
        d0 = sd[sd["dgp"] == "garch_t"][["label", "reject"]].assign(param=0.0)
        dd = pd.concat([d0, sd[sd["dgp"] == dgp][["label", "reject", "param"]]])
        env = dd.groupby("param")["reject"].agg(["min", "median", "max"]).reindex(params)
        p.area(params, env["min"].to_numpy(), env["max"].to_numpy(), INK3, 0.16)
        p.line(params, env["median"].to_numpy(), INK2, 1.5, dash="2 2")
        for key, lab, col, shp in PROC_POWER:
            q = s[s["procedure"] == key].sort_values("param")
            if q.empty:
                continue
            p.line(q["param"], q["reject"], col, 2)
            p.dots(q["param"], q["reject"], col, 3.4, shp)
        p.axis_x(params, fmt=fmt, label=f"strength of planted predictability, {sym}")
        nrep = int(s[s["procedure"] == "boot_avg"]["reps"].min()) if "reps" in s else 0
        p.title(title, f"K = 56 designs, {nrep:,} replications per point")
        parts.append(p.svg())
    legend = legend_html([(lab, col, shp) for k, lab, col, shp in PROC_POWER] +
                         [("nominal level 5%", INK2, "dash"), ("single design: median (dashed) and range", "#d6d5d0", "wash")])
    save("fig_power", legend + svg_wrap(W, 44 + 220 + 40, "".join(parts)))


def main():
    fig_design()
    fig_us_fan()
    fig_eras()
    fig_corr()
    fig_size()
    fig_power()


if __name__ == "__main__":
    main()
