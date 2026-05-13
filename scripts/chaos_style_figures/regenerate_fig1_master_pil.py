#!/usr/bin/env python3
"""PIL redraw of fig1_signal_waterfall with cleaner panels a/b."""

import pathlib
from math import sin, pi

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


OUT = pathlib.Path(r"D:\LLM\outputs\chaos_style_figures")
OUT.mkdir(parents=True, exist_ok=True)
PNG = OUT / "fig1_signal_waterfall.png"
PDF = OUT / "fig1_signal_waterfall.pdf"

S = 3
W, H = 1050, 1770

C_TEAL = "#2A9D8F"
C_TEAL_DK = "#1D7066"
C_NAVY = "#264653"
C_NAVY_DK = "#162C39"
C_SAGE = "#8AB17D"
C_ORANGE = "#E76F51"
C_ORANGE_LT = "#F4A261"
C_RED = "#C0392B"
C_PURPLE = "#4B0082"
C_GRAY = "#7F8C8D"
C_BLACK = "#2C3E50"
C_LITE = "#F4F7F8"
C_GRID = "#DCE3E6"
C_WHITE = "#FFFFFF"

configs = ["Random", "+Temporal", "+Structural", "+Semantic", "+Access", "+Historical", "Full PROSE"]
short = ["Random", "+Temp.", "+Struct.", "+Sem.", "+Access", "+Hist.", "Full"]
recovery = np.array([0.109, 0.678, 0.691, 0.698, 0.701, 0.703, 0.703])
oracle = 0.903
stage_abs = np.array([
    [0.040, 0.040, 0.029],
    [0.200, 0.430, 0.048],
    [0.230, 0.445, 0.016],
    [0.240, 0.455, 0.003],
    [0.245, 0.453, 0.003],
    [0.248, 0.443, 0.012],
    [0.248, 0.443, 0.012],
])
for i in range(len(configs)):
    stage_abs[i] = stage_abs[i] / stage_abs[i].sum() * recovery[i]
deltas = np.array([0] + [recovery[i] - recovery[i - 1] for i in range(1, len(configs))])
std = np.array([0, 0.012, 0.006, 0.004, 0.003, 0.002, 0])
invalid = np.array([0.70, 0.15, 0.10, 0.07, 0.04, 0.02, 0.00])
gain_vis = np.array([0.000, 0.067, 0.013, 0.007, 0.003, 0.002, 0.000])
gain_std = np.array([0.000, 0.012, 0.006, 0.004, 0.003, 0.002, 0.000])


def font(size, bold=False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    roots = [pathlib.Path(r"C:\Windows\Fonts"), pathlib.Path(r"C:\Users\74597\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages\matplotlib\mpl-data\fonts\ttf")]
    for root in roots:
        for name in names:
            p = root / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_TITLE = font(38, True)
F_PANEL = font(22, True)
F_LABEL = font(23, True)
F_TICK = font(20, False)
F_SMALL = font(17, True)
F_TINY = font(15, True)
F_AXIS = font(24, True)


def rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    a, b = np.array(rgb(c1)), np.array(rgb(c2))
    return tuple((a * (1 - t) + b * t).astype(np.uint8))


def text_center(draw, xy, s, f, fill=C_BLACK):
    box = draw.textbbox((0, 0), s, font=f)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), s, font=f, fill=fill)


def draw_rotated_text(base, xy, s, f, angle, fill=C_BLACK):
    box = ImageDraw.Draw(Image.new("RGBA", (10, 10))).textbbox((0, 0), s, font=f)
    tw, th = box[2] - box[0] + 12, box[3] - box[1] + 12
    layer = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
    d = ImageDraw.Draw(layer)
    d.text((6, 6), s, font=f, fill=rgb(fill) + (255,))
    layer = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    base.alpha_composite(layer, (int(xy[0] - layer.width / 2), int(xy[1] - layer.height / 2)))


def rounded_bar(draw, xy, c1, c2, radius=16):
    x0, y0, x1, y1 = map(int, xy)
    mask = Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, mask.width - 1, mask.height - 1), radius=radius, fill=255)
    grad = Image.new("RGBA", mask.size)
    gd = ImageDraw.Draw(grad)
    for x in range(mask.width):
        gd.line((x, 0, x, mask.height), fill=mix(c1, c2, x / max(1, mask.width - 1)) + (255,))
    img.paste(grad, (x0, y0), mask)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, outline=(255, 255, 255, 190), width=2)


def panel(draw, box):
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1, y1), fill=C_WHITE)
    draw.line((x0, y1, x1, y1), fill="#BBBBBB", width=2)
    draw.line((x0, y0, x0, y1), fill="#BBBBBB", width=2)


img = Image.new("RGBA", (W, H), C_WHITE)
draw = ImageDraw.Draw(img)
text_center(draw, (W / 2, 43), "Signal Ablation: From Random Summary to Oracle-JIT", F_TITLE, C_BLACK)

box_a = (166, 130, 497, 745)
box_b = (622, 130, 945, 745)
box_c = (166, 1115, 497, 1520)
box_d = (622, 1014, 945, 1628)
for b in (box_a, box_b, box_c, box_d):
    panel(draw, b)

# Panel a
text_center(draw, ((box_a[0] + box_a[1] + box_a[2] - box_a[0]) / 2, 112), "Stage-Aware Recovery Waterfall", F_PANEL)
draw.text((105, 122), "(a)", font=F_PANEL, fill=C_BLACK)
x0, y0, x1, y1 = box_a
plot_l, plot_r = x0, x1
row_y = np.linspace(y0 + 10, y1 - 22, len(configs))
bar_h = 38
to_x = lambda v: plot_l + v * (plot_r - plot_l)
draw.rectangle((to_x(0.70), y0, to_x(0.93), y1), fill="#F4ECF7")
draw.rectangle((to_x(0.93), y0, x1, y1), fill="#F8F9F9")
for v in np.linspace(0, 1, 5):
    xx = to_x(v)
    draw.line((xx, y0, xx, y1), fill=C_GRID, width=1)
    text_center(draw, (xx, y1 + 27), f"{v:.2f}", F_TICK)
spine_x = x0 - 42
draw.line((spine_x, row_y[0], spine_x, row_y[-1]), fill="#8C9AA3", width=4)
for i, cy in enumerate(row_y):
    draw.text((x0 - 168, cy - 13), configs[i], font=F_TICK, fill=C_BLACK)
    draw.rounded_rectangle((x0, cy - bar_h / 2, x1, cy + bar_h / 2), radius=14,
                           fill=C_LITE, outline="#D6DEE2", width=1)
    draw.ellipse((spine_x - 13, cy - 13, spine_x + 13, cy + 13), fill=C_WHITE, outline=C_BLACK, width=2)
    r = int(8 + 12 * recovery[i])
    draw.ellipse((spine_x - r, cy - r, spine_x + r, cy + r), fill=C_TEAL if i else C_GRAY)
    if i > 0 and deltas[i] > 0.002:
        draw_rotated_text(img, (spine_x - 34, cy - 47), f"+{deltas[i]:.3f}", F_TINY, 90, C_TEAL_DK)
    left = 0
    cols = [(C_TEAL, C_TEAL_DK), (C_NAVY, C_NAVY_DK), (C_SAGE, "#5E8C52")]
    for s in range(3):
        seg = stage_abs[i, s]
        rounded_bar(draw, (to_x(left), cy - bar_h / 2, to_x(left + seg), cy + bar_h / 2),
                    cols[s][0], cols[s][1], radius=13)
        if i in (1, 5) and seg > 0.05:
            text_center(draw, ((to_x(left) + to_x(left + seg)) / 2, cy), f"{seg:.2f}", F_TINY, C_WHITE)
        left += seg
    draw.text((to_x(recovery[i]) + 9, cy - 11), f"{recovery[i]:.3f}", font=F_SMALL, fill=C_BLACK)
ox = to_x(oracle)
draw.line((ox, y0, ox, y1), fill=C_PURPLE, width=4)
for yy in range(y0, y1, 18):
    draw.line((ox, yy, ox, yy + 9), fill=C_WHITE, width=2)
draw.rounded_rectangle((ox - 82, y0 + 14, ox + 82, y0 + 48), radius=10, fill="#F4ECF7", outline=C_PURPLE, width=3)
text_center(draw, (ox, y0 + 31), "Oracle-JIT 0.903", F_TINY, C_PURPLE)
draw.line((to_x(recovery[-1]), row_y[1] - 22, ox, row_y[1] - 22), fill=C_PURPLE, width=3)
draw.polygon([(to_x(recovery[-1]), row_y[1] - 22), (to_x(recovery[-1]) + 10, row_y[1] - 28), (to_x(recovery[-1]) + 10, row_y[1] - 16)], fill=C_PURPLE)
draw.polygon([(ox, row_y[1] - 22), (ox - 10, row_y[1] - 28), (ox - 10, row_y[1] - 16)], fill=C_PURPLE)
text_center(draw, ((to_x(recovery[-1]) + ox) / 2, row_y[1] - 43), "Delta=0.200", F_TINY, C_PURPLE)
text_center(draw, ((x0 + x1) / 2, y1 + 70), "Selection Recovery", F_AXIS)
legend_y = y1 + 116
for j, (name, col) in enumerate(zip(["PREFILL", "DECODE", "SPECULATE"], [C_TEAL, C_NAVY, C_SAGE])):
    lx = x0 - 92 + j * 143
    draw.rectangle((lx, legend_y - 12, lx + 22, legend_y + 10), fill=col)
    draw.text((lx + 32, legend_y - 14), name, font=F_TICK, fill="#111111")

# Panel b
text_center(draw, ((box_b[0] + box_b[2]) / 2, 112), "Per-Signal Gain", F_PANEL)
draw.text((543, 122), "(b)", font=F_PANEL, fill=C_BLACK)
x0, y0, x1, y1 = box_b
plot_b = y1 - 28
gain_to_y = lambda v: plot_b - (v + 0.006) / 0.094 * (plot_b - y0)
for v in [0, 0.02, 0.04, 0.06, 0.08]:
    yy = gain_to_y(v)
    draw.line((x0, yy, x1, yy), fill=C_GRID, width=1)
    draw.text((x0 - 55, yy - 12), f"{v:.2f}", font=F_TICK, fill="#111111")
alpha_y = gain_to_y(0.005)
draw.rectangle((x0, alpha_y, x1, plot_b), fill="#FDF0EE")
for xx in range(x0, x1, 14):
    draw.line((xx, alpha_y, xx + 7, alpha_y), fill="#D98273", width=2)
draw.text((x1 - 108, alpha_y - 31), "alpha=0.005", font=F_TINY, fill=C_RED)
xpos = np.linspace(x0 + 32, x1 - 32, len(configs))
bar_cols = [C_GRAY, C_RED, C_ORANGE, C_ORANGE_LT, "#7FB3D5", C_TEAL, C_NAVY]
for i, xx in enumerate(xpos):
    base = gain_to_y(0)
    if i == 0:
        draw.ellipse((xx - 15, base - 15, xx + 15, base + 15), fill=C_GRAY, outline=C_BLACK, width=2)
        draw.text((xx - 34, base + 26), "Baseline", font=F_TINY, fill=C_GRAY)
    else:
        yy = gain_to_y(gain_vis[i])
        draw.rounded_rectangle((xx - 17, yy, xx + 17, base), radius=8, fill=bar_cols[i])
        draw.line((xx, gain_to_y(gain_vis[i] - gain_std[i]), xx, gain_to_y(gain_vis[i] + gain_std[i])), fill="#555555", width=2)
        draw.line((xx - 8, gain_to_y(gain_vis[i] - gain_std[i]), xx + 8, gain_to_y(gain_vis[i] - gain_std[i])), fill="#555555", width=2)
        draw.line((xx - 8, gain_to_y(gain_vis[i] + gain_std[i]), xx + 8, gain_to_y(gain_vis[i] + gain_std[i])), fill="#555555", width=2)
        draw.ellipse((xx - 15, yy - 15, xx + 15, yy + 15), fill=bar_cols[i], outline=C_BLACK, width=2)
        if i in (2, 3, 4, 5):
            text_center(draw, (xx, yy - 34), f"+{gain_vis[i]:.3f}", F_TINY)
for i in range(1, len(xpos)):
    draw.line((xpos[i - 1], gain_to_y(gain_vis[i - 1]), xpos[i], gain_to_y(gain_vis[i])),
              fill="#9AA4AA", width=2)
draw.rounded_rectangle((xpos[1] + 42, gain_to_y(0.083), xpos[1] + 177, gain_to_y(0.060)),
                       radius=10, fill=C_WHITE, outline=C_RED, width=2)
text_center(draw, (xpos[1] + 109, gain_to_y(0.0715)), "+0.067\nTemporal dominates", F_SMALL, C_RED)
draw.line((xpos[1] + 42, gain_to_y(0.066), xpos[1] + 8, gain_to_y(gain_vis[1]) - 7), fill=C_RED, width=3)
inv_y = lambda v: y0 + (v + 0.05) / 0.87 * (plot_b - y0)
inv_pts = [(xpos[i], inv_y(invalid[i] * 100)) for i in range(len(xpos))]
inv_pts = [(xpos[i], y0 + (invalid[i] * 100 + 5) / 87 * (plot_b - y0)) for i in range(len(xpos))]
for i in range(1, len(inv_pts)):
    draw.line((inv_pts[i - 1], inv_pts[i]), fill="#D77D73", width=4)
for i, p in enumerate(inv_pts):
    draw.rectangle((p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10), fill=C_RED)
for i, txt in [(0, "70%"), (1, "15%"), (6, "0%")]:
    draw.text((inv_pts[i][0] - 20, inv_pts[i][1] - 34), txt, font=F_SMALL, fill=C_RED)
draw.text((x1 + 18, y0 + 250), "Invalid (%)", font=F_AXIS, fill=C_RED)
draw_rotated_text(img, (x0 - 82, (y0 + y1) / 2), "Incremental Delta Recovery", F_AXIS, 90)
for i, xx in enumerate(xpos):
    draw_rotated_text(img, (xx, y1 + 45), short[i], F_TICK, 28, C_BLACK)

# Panel c
text_center(draw, ((box_c[0] + box_c[2]) / 2, 1095), "Precision-Recall Trajectory", F_PANEL)
draw.text((92, 1104), "(c)", font=F_PANEL, fill=C_BLACK)
x0, y0, x1, y1 = box_c
for ix in range(x0, x1, 6):
    for iy in range(y0, y1, 6):
        p = (ix - x0) / (x1 - x0)
        r = 1 - (iy - y0) / (y1 - y0)
        f1 = 2 * p * r / (p + r + 1e-6)
        col = mix("#FFF8D5", "#5DB3C1", min(1, f1))
        draw.rectangle((ix, iy, ix + 6, iy + 6), fill=col)
draw.rectangle(box_c, outline="#BBBBBB", width=2)
pr = np.array([[0.14, 0.07], [0.71, 0.64], [0.74, 0.66], [0.76, 0.67], [0.77, 0.68], [0.78, 0.68], [0.78, 0.69], [0.95, 0.88]])
pt = lambda p: (x0 + p[0] * (x1 - x0), y1 - p[1] * (y1 - y0))
for i in range(len(pr) - 1):
    draw.line((pt(pr[i]), pt(pr[i + 1])), fill=C_BLACK, width=4)
labs = ["Random", "+Temporal", "+Structural", "+Semantic", "+Access", "+Historical", "Full PROSE", "Oracle"]
label_pos = [(0.18, 0.15), (0.40, 0.39), (0.88, 0.60), (0.88, 0.46), (0.88, 0.74), (0.52, 0.86), (0.50, 0.70), (0.80, 0.94)]
for i, p in enumerate(pr):
    px, py = pt(p)
    col = [C_RED, C_TEAL, C_TEAL, C_TEAL, C_TEAL, C_TEAL, C_NAVY, "#0E8A3A"][i]
    draw.ellipse((px - 18, py - 18, px + 18, py + 18), fill=col, outline=C_BLACK, width=3)
    lx, ly = pt(label_pos[i])
    draw.rounded_rectangle((lx - 70, ly - 20, lx + 70, ly + 20), radius=8, fill=C_WHITE, outline="#AAAAAA", width=2)
    text_center(draw, (lx, ly), labs[i], F_SMALL)
draw.text((x0 + 110, y0 + 35), "F1=0.85", font=F_SMALL, fill=C_BLACK)
draw.text((x0 + 16, y0 + 160), "F1=0.70", font=F_SMALL, fill=C_BLACK)
draw.text((x0 - 10, y0 + 270), "F1=0.50", font=F_SMALL, fill=C_BLACK)
text_center(draw, ((x0 + x1) / 2, y1 + 64), "Precision", F_AXIS)
draw_rotated_text(img, (x0 - 82, (y0 + y1) / 2), "Recall", F_AXIS, 90)

# Panel d
text_center(draw, ((box_d[0] + box_d[2]) / 2, 995), "Signal-Efficiency Frontier", F_PANEL)
draw.text((543, 1010), "(d)", font=F_PANEL, fill=C_BLACK)
x0, y0, x1, y1 = box_d
xpos = np.linspace(x0 + 32, x1 - 32, len(configs))
rec_y = lambda v: y1 - v * (y1 - y0)
inv_y2 = lambda v: y0 + v * (y1 - y0)
poly = [(xpos[0], y1)] + [(xpos[i], rec_y(recovery[i])) for i in range(len(xpos))] + [(xpos[-1], y1)]
draw.polygon(poly, fill="#DDE6E9")
for i in range(1, len(xpos)):
    draw.line((xpos[i - 1], rec_y(recovery[i - 1]), xpos[i], rec_y(recovery[i])), fill=C_NAVY, width=7)
    draw.line((xpos[i - 1], inv_y2(invalid[i - 1]), xpos[i], inv_y2(invalid[i])), fill=C_RED, width=7)
for i, xx in enumerate(xpos):
    draw.ellipse((xx - 13, rec_y(recovery[i]) - 13, xx + 13, rec_y(recovery[i]) + 13), fill=C_NAVY)
    draw.rectangle((xx - 13, inv_y2(invalid[i]) - 13, xx + 13, inv_y2(invalid[i]) + 13), fill=C_RED)
    draw.text((xx - 28, rec_y(recovery[i]) - 43 if i % 2 == 0 else rec_y(recovery[i]) + 15),
              f"{recovery[i]:.3f}", font=F_TINY, fill=C_NAVY)
draw.rounded_rectangle((x0 + 128, y0 + 286, x0 + 305, y0 + 356), radius=8, fill=C_WHITE)
text_center(draw, (x0 + 216, y0 + 321), "Temporal drops\ninvalid 70%->15%", F_SMALL, C_TEAL)
draw.line((x0 + 174, y0 + 286, xpos[1], rec_y(recovery[1])), fill=C_TEAL, width=4)
draw.text((xpos[0] + 22, inv_y2(invalid[0]) - 15), "70%", font=F_SMALL, fill=C_RED)
draw.text((xpos[1] - 28, inv_y2(invalid[1]) - 38), "15%", font=F_SMALL, fill=C_RED)
draw.text((xpos[-1] + 8, inv_y2(invalid[-1]) + 25), "0%", font=F_SMALL, fill=C_RED)
draw_rotated_text(img, (x0 - 82, (y0 + y1) / 2), "Selection Recovery", F_AXIS, 90)
draw_rotated_text(img, (x1 + 82, (y0 + y1) / 2), "Invalid Traffic", F_AXIS, 90, C_RED)
text_center(draw, ((x0 + x1) / 2, y1 + 92), "Signals Accumulated", F_AXIS)
for i, xx in enumerate(xpos):
    draw_rotated_text(img, (xx, y1 + 45), short[i], F_TICK, 25, C_BLACK)
draw.rounded_rectangle((x1 - 150, y1 - 90, x1 - 8, y1 - 18), radius=8, fill=C_WHITE, outline="#CCCCCC", width=2)
draw.line((x1 - 134, y1 - 64, x1 - 104, y1 - 64), fill=C_NAVY, width=6)
draw.text((x1 - 95, y1 - 77), "Recovery", font=F_TICK, fill="#111111")
draw.line((x1 - 134, y1 - 36, x1 - 104, y1 - 36), fill=C_RED, width=6)
draw.text((x1 - 95, y1 - 49), "Invalid", font=F_TICK, fill="#111111")

img = img.convert("RGB")
img.save(PNG, quality=96)
pdf_w, pdf_h = W * 72 / 300, H * 72 / 300
c = canvas.Canvas(str(PDF), pagesize=(pdf_w, pdf_h))
c.drawImage(ImageReader(str(PNG)), 0, 0, width=pdf_w, height=pdf_h)
c.save()
print(f"PNG: {PNG} ({PNG.stat().st_size / 1024:.0f} KB)")
print(f"PDF: {PDF} ({PDF.stat().st_size / 1024:.0f} KB)")
