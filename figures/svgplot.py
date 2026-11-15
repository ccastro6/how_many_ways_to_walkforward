"""A tiny SVG plotting helper for print figures.

Charts are written as inline SVG inside an HTML page styled with ``style.css``;
``render.js`` (Playwright + Chromium) prints each page to a vector PDF at its exact
size. Everything is plain text, so figures are reproducible and diffable.

Units are CSS pixels at 96 px per inch, so a 624 px wide figure is 6.5 in wide and
is included in LaTeX at 100% scale (text and lines keep their designed sizes).
"""
from __future__ import annotations

import html
import math
from dataclasses import dataclass, field

# Reference categorical palette (light mode), slots in fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, AXIS, SURFACE = "#e4e3df", "#bdbcb6", "#ffffff"
BLUE_WASH, ORANGE_WASH = "#cde2fb", "#fbdccf"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + 1e-9 * step:
        out.append(round(v, 10))
        v += step
    return out


@dataclass
class Scale:
    d0: float
    d1: float
    r0: float
    r1: float
    log: bool = False

    def __call__(self, v: float) -> float:
        if self.log:
            v, a, b = math.log(v), math.log(self.d0), math.log(self.d1)
        else:
            a, b = self.d0, self.d1
        return self.r0 + (v - a) / (b - a) * (self.r1 - self.r0)


@dataclass
class Panel:
    """One plotting area with its own scales, drawn at (x, y) inside the figure SVG."""
    x: float
    y: float
    w: float
    h: float
    xs: Scale = None
    ys: Scale = None
    parts: list = field(default_factory=list)

    def set(self, xlim, ylim, ylog=False, xlog=False):
        self.xs = Scale(xlim[0], xlim[1], self.x, self.x + self.w, xlog)
        self.ys = Scale(ylim[0], ylim[1], self.y + self.h, self.y, ylog)
        return self

    # -------------------------------------------------------------- frame
    def grid_y(self, ticks, fmt=lambda v: f"{v:g}", label=None, zero=None):
        for t in ticks:
            yy = self.ys(t)
            self.parts.append(f'<line x1="{self.x:.1f}" x2="{self.x + self.w:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                              f'stroke="{GRID}" stroke-width="1"/>')
            self.parts.append(f'<text class="tick" x="{self.x - 6:.1f}" y="{yy + 3.5:.1f}" text-anchor="end">{esc(fmt(t))}</text>')
        if zero is not None:
            yy = self.ys(zero)
            self.parts.append(f'<line x1="{self.x:.1f}" x2="{self.x + self.w:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                              f'stroke="{AXIS}" stroke-width="1"/>')
        if label:
            cy = self.y + self.h / 2
            self.parts.append(f'<text class="axlabel" transform="translate({self.x - 38:.1f},{cy:.1f}) rotate(-90)" '
                              f'text-anchor="middle">{esc(label)}</text>')

    def axis_x(self, ticks, fmt=lambda v: f"{v:g}", label=None, labels=None):
        yb = self.y + self.h
        self.parts.append(f'<line x1="{self.x:.1f}" x2="{self.x + self.w:.1f}" y1="{yb:.1f}" y2="{yb:.1f}" '
                          f'stroke="{AXIS}" stroke-width="1"/>')
        for i, t in enumerate(ticks):
            xx = self.xs(t)
            self.parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{yb:.1f}" y2="{yb + 4:.1f}" stroke="{AXIS}" stroke-width="1"/>')
            txt = labels[i] if labels else fmt(t)
            self.parts.append(f'<text class="tick" x="{xx:.1f}" y="{yb + 15:.1f}" text-anchor="middle">{esc(txt)}</text>')
        if label:
            self.parts.append(f'<text class="axlabel" x="{self.x + self.w / 2:.1f}" y="{yb + 32:.1f}" '
                              f'text-anchor="middle">{esc(label)}</text>')

    def title(self, text, sub=None):
        self.parts.append(f'<text class="ptitle" x="{self.x:.1f}" y="{self.y - 14:.1f}">{esc(text)}</text>')
        if sub:
            self.parts.append(f'<text class="psub" x="{self.x:.1f}" y="{self.y - 2:.1f}">{esc(sub)}</text>')

    # -------------------------------------------------------------- marks
    def hline(self, v, color=INK2, dash=None, width=1.0, label=None, label_side="right"):
        yy = self.ys(v)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{self.x:.1f}" x2="{self.x + self.w:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                          f'stroke="{color}" stroke-width="{width}"{d}/>')
        if label:
            if label_side == "right":
                self.parts.append(f'<text class="note" x="{self.x + self.w - 2:.1f}" y="{yy - 4:.1f}" text-anchor="end">{esc(label)}</text>')
            else:
                self.parts.append(f'<text class="note" x="{self.x + 4:.1f}" y="{yy - 4:.1f}">{esc(label)}</text>')

    def band(self, x0, x1, y0, y1, color, opacity=0.12):
        X0, X1 = sorted((self.xs(x0), self.xs(x1)))
        Y0, Y1 = sorted((self.ys(y0), self.ys(y1)))
        self.parts.append(f'<rect x="{X0:.1f}" y="{Y0:.1f}" width="{X1 - X0:.1f}" height="{Y1 - Y0:.1f}" '
                          f'fill="{color}" fill-opacity="{opacity}"/>')

    def line(self, xs, ys, color, width=2.0, dash=None, opacity=1.0):
        pts = " ".join(f"{self.xs(a):.2f},{self.ys(b):.2f}" for a, b in zip(xs, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
                          f'stroke-linejoin="round" stroke-linecap="round" stroke-opacity="{opacity}"{d}/>')

    def area(self, xs, lo, hi, color, opacity=0.12):
        up = [f"{self.xs(a):.2f},{self.ys(b):.2f}" for a, b in zip(xs, hi)]
        dn = [f"{self.xs(a):.2f},{self.ys(b):.2f}" for a, b in zip(xs[::-1], lo[::-1])]
        self.parts.append(f'<polygon points="{" ".join(up + dn)}" fill="{color}" fill-opacity="{opacity}" stroke="none"/>')

    def dots(self, xs, ys, color, r=4.0, shape="circle", ring=True):
        for a, b in zip(xs, ys):
            self.dot(a, b, color, r, shape, ring)

    def dot(self, a, b, color, r=4.0, shape="circle", ring=True):
        cx, cy = self.xs(a), self.ys(b)
        st = f' stroke="{SURFACE}" stroke-width="1.5"' if ring else ""
        if shape == "circle":
            self.parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{color}"{st}/>')
        elif shape == "square":
            s = r * 1.75
            self.parts.append(f'<rect x="{cx - s / 2:.2f}" y="{cy - s / 2:.2f}" width="{s:.2f}" height="{s:.2f}" rx="1" fill="{color}"{st}/>')
        elif shape == "diamond":
            s = r * 1.35
            self.parts.append(f'<polygon points="{cx:.2f},{cy - s:.2f} {cx + s:.2f},{cy:.2f} {cx:.2f},{cy + s:.2f} {cx - s:.2f},{cy:.2f}" fill="{color}"{st}/>')
        elif shape == "triangle":
            s = r * 1.3
            self.parts.append(f'<polygon points="{cx:.2f},{cy - s:.2f} {cx + s:.2f},{cy + s * 0.8:.2f} {cx - s:.2f},{cy + s * 0.8:.2f}" fill="{color}"{st}/>')
        elif shape == "ring":
            self.parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r - 0.75}" fill="{SURFACE}" stroke="{color}" stroke-width="1.5"/>')

    def text(self, a, b, s, cls="label", anchor="start", dx=0, dy=0):
        self.parts.append(f'<text class="{cls}" x="{self.xs(a) + dx:.1f}" y="{self.ys(b) + dy:.1f}" text-anchor="{anchor}">{esc(s)}</text>')

    def raw(self, s):
        self.parts.append(s)

    def svg(self) -> str:
        return "\n".join(self.parts)


def legend_html(items) -> str:
    """items: list of (label, color, kind) with kind in line|dot|square|diamond|triangle|dash|wash."""
    out = ['<div class="legend">']
    for label, color, kind in items:
        if kind == "line":
            key = f'<svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>'
        elif kind == "dash":
            key = f'<svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="{color}" stroke-width="2" stroke-dasharray="4 3"/></svg>'
        elif kind == "wash":
            key = f'<svg width="14" height="10"><rect x="0" y="0" width="14" height="10" rx="2" fill="{color}"/></svg>'
        else:
            p = Panel(0, 0, 14, 10).set((0, 1), (0, 1))
            p.dot(0.5, 0.5, color, 4, kind, ring=False)
            key = f'<svg width="14" height="10">{p.svg()}</svg>'
        out.append(f'<span class="item">{key}<span>{esc(label)}</span></span>')
    out.append("</div>")
    return "".join(out)


def page(body: str, width: int, title: str = "figure") -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{esc(title)}</title>
<link rel="stylesheet" href="../../figures/style.css"></head>
<body><div id="fig" style="width:{width}px">{body}</div></body></html>"""
