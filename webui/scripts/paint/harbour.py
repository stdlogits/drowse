"""Harbour — the link card's painting: "the harbour goes quiet as the last boats come in".

The ground is the mark's tile at card size, the depth ramp as a dusk sky over
still water (Ledger's DESIGN.md, "The marks and the cards"). The quay has a
lamp for each of the model's layers, in that layer's colour, and each lamp's
reflection is as long as one word of the line reads calm at that layer: the
dashboard's layer chart, lying on the water. The water under the steered
layers is a shade smoother. The last boat in is the mark: its sheer line is
the closed lid, and its three lights hang their reflections from it as
lashes.

    python harbour.py <scripts/social-card.json> [scale] [out.png]

The readings are illustrative, and scripts/social-card.json holds them for
this and for the type (scripts/social-card.html; scripts/social-card.mjs runs
both). Needs numpy, scipy and Pillow, and Ledger's paint (node_modules, or
LEDGER_PAINT).

Written by Claude, 2026-11-16.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("LEDGER_PAINT") or os.path.join(HERE, "../../node_modules/@stdlogits/ledger/src/paint"))
from brush import F, Canvas, mix, rect, spline, FILBERT, FLAT, LINE, LINER, ROUND, SOFT  # noqa: E402
from dusk import W, H, PAPER, LIGHT, RAMP, GLOW, ramp, wash, vignette  # noqa: E402

CARD = json.load(open(sys.argv[1], encoding="utf-8"))
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "harbour.png"
t0 = time.time()

HOR = 356.0                                                    # the waterline
READ = CARD["reading"]                                         # the word's reading, layer by layer
NL = len(READ)
STEERED = tuple(CARD["steered"])                               # the term's layers, first and last
LAMP0, LAMP1 = 262.0, 1118.0                                   # the first and last lamps


def lamp_x(L):
    return LAMP0 + (LAMP1 - LAMP0) * L / (NL - 1)


def depth(x):
    return float(np.clip((x - LAMP0) / (LAMP1 - LAMP0), 0, 1))


def sky(x, y):
    t = depth(x)
    v = float(np.clip(y / HOR, 0, 1))
    c = mix(PAPER, ramp(RAMP, t), 0.20 + 0.46 * v ** 1.7)
    return c + ramp(GLOW, t) * 0.16 * np.exp(-((HOR - y) / 70.0) ** 2)      # the last light, low


def water(x, y):
    u = float(np.clip((y - HOR) / (H - HOR), 0, 1))
    c = sky(x, HOR - (y - HOR) * 0.92) * (0.92 - 0.22 * u)
    c = c * (1 + 0.05 * np.sin(x * 0.011 + 60.0 / (y - HOR + 14.0)))
    L = (x - LAMP0) / (LAMP1 - LAMP0) * (NL - 1)
    if STEERED[0] - 0.5 < L < STEERED[1] + 0.5:                               # the steered layers: a shade smoother
        c = c + ramp(RAMP, depth(x)) * 0.045 * (1 - u) ** 0.5
    return np.clip(c, 0, 1)


cv = Canvas(W, H, SCALE, ground=tuple(sky(600, 200)), seed=40)
rng = cv.rng


def paint_sky():
    wash(cv, sky, rect(W / 2, HOR / 2 - 6, W + 80, HOR + 60), width=(44, 32), angle=(-3, 2), length=((200, 460), (180, 420)), curve=6,
         jitter=0.01)
    n = 0
    while n < 90:                                 # a few early stars, none where the type is set
        x, y = rng.uniform(0, W), rng.uniform(0, HOR - 90)
        if x < 1010 and 30 < y < 300:
            continue
        n += 1
        mag = rng.random() ** 3
        c = mix(LIGHT, ramp(GLOW, depth(x)), 0.4) * (0.5 + 0.5 * mag)
        cv.dab((x, y), 0.6 + 1.1 * mag, mix(sky(x, y), c, 0.8), soft=0.5, thick=0.2)
    # long thin clouds, lit from below
    for _ in range(7):
        x0, y0 = rng.uniform(380, 1250), rng.uniform(238, 320)
        for j in range(rng.integers(2, 4)):
            x = x0 + rng.normal(0, 80)
            y = y0 + j * rng.uniform(3, 6)
            ln = rng.uniform(160, 420)
            w = rng.uniform(4, 9)
            body = mix(sky(x + ln / 2, y), PAPER, 0.30)
            path = spline([(x, y + rng.normal(0, 1.2)), (x + ln * 0.35, y - rng.uniform(0, 3)), (x + ln * 0.7, y - rng.uniform(0, 2)),
                           (x + ln, y + rng.normal(0, 1.2))], 6)
            cv.stroke(path, w, body, FILBERT, taper=(0.35, 0.45), tip=0.05, dry=0.4, grain=1.0, streak=0.03, opacity=0.75)
            cv.stroke(path + np.array([0, w * 0.36], F), max(1.2, w * 0.3), mix(body, ramp(GLOW, depth(x + ln / 2)), 0.5),
                      ROUND, taper=(0.35, 0.45), tip=0.05, opacity=0.4)


def shore():
    # the far shore: two low arms and the town between them, asleep
    def top(x):
        a = 46 * np.exp(-((x + 40) / 190.0) ** 2) + 30 * np.exp(-((x - 1230) / 170.0) ** 2)
        return HOR - 7 - a - 5 * np.sin(x * 0.013) ** 2 - 3 * np.sin(x * 0.041 + 1.0) ** 2

    xs = np.linspace(-20, W + 20, 90)
    poly = [(x, top(x)) for x in xs] + [(W + 20, HOR + 2), (-20, HOR + 2)]
    cv.fill(np.array(poly, F), lambda x, y: mix(PAPER, ramp(RAMP, depth(x)), 0.15 + 0.10 * (y - (HOR - 50)) / 50), FLAT,
            width=9, length=(40, 120), overlap=0.35, streak=0.03, dry=0.1)
    x = 228.0
    while x < 1160:                               # roofs along the quay
        w = rng.uniform(9, 26)
        h = rng.uniform(5, 15)
        c = mix(PAPER, ramp(RAMP, depth(x)), rng.uniform(0.10, 0.17))
        cv.stroke([(x, HOR - 3 - h / 2), (x + w, HOR - 3 - h / 2)], h, c, FLAT, taper=(0, 0), tip=1.0, dry=0.05)
        x += w + rng.uniform(0, 5)
    cv.stroke([(214, HOR + 1.5), (1166, HOR + 1.5)], 5.5, mix(PAPER, ramp(RAMP, 0.5), 0.08), LINE)


def paint_water():
    whole = rect(W / 2, (HOR + H) / 2 + 4, W + 80, H - HOR + 30)
    with cv.frisket(rect(W / 2, (HOR + 4 + H + 40) / 2, W + 80, H + 40 - HOR - 4)):
        cv.fill(whole, water, FLAT, width=13, angle=0, length=(160, 520), overlap=0.45, streak=0.035, chroma=0.008,
                jitter=0.012, frisket=False, dry=0.05, wander=0.6)
        cv.fill(whole, water, FILBERT, width=7, angle=0, length=(60, 240), overlap=-1.6, streak=0.03, jitter=0.02,
                frisket=False, opacity=0.5, wander=0.8, shuffle=True)


def reflection(x, y0, length, c, w0=3.2, w1=9.0, strength=1.0, lean=0.0):
    """A light on still water: a column of short strokes, wider and fainter toward us."""
    y = y0
    while y < y0 + length:
        f = (y - y0) / max(length, 1.0)
        w = w0 + (w1 - w0) * f ** 0.8 + rng.normal(0, 0.5)
        dx = rng.normal(0, 0.4 + 1.6 * f) + lean * (y - y0)
        a = strength * (1 - f) ** 0.75 * rng.uniform(0.7, 1.0)
        if rng.random() > 0.10 + 0.25 * f:
            cv.stroke([(x + dx - w / 2, y), (x + dx + w / 2, y)], 1.9, mix(c, LIGHT, 0.35 * (1 - f)), LINE, opacity=a, soft=0.5)
        y += 2.9 + 1.6 * f


def quay_lamps():
    for L in range(NL):
        x = lamp_x(L)
        c = ramp(GLOW, L / (NL - 1))
        cv.line((x, HOR - 1), (x, HOR - 13), 1.1, mix(PAPER, ramp(RAMP, L / (NL - 1)), 0.1))
        ln = 14 + 205 * READ[L]
        cv.stroke([(x, HOR + 6), (x, HOR + 6 + ln * 0.9)], 9, c, SOFT, opacity=0.16, mode="screen", taper=(0.02, 0.7), tip=0.3)
        reflection(x, HOR + 6, ln, c, strength=0.55 + 0.4 * READ[L])
        cv.glow((x, HOR + 3), 6, c, 0.22)
        cv.glow((x, HOR - 15), 10, c, 0.50)
        cv.glow((x, HOR - 15), 4, LIGHT, 0.55)
        cv.dab((x, HOR - 15), 1.7, mix(LIGHT, c, 0.15), soft=0.4, thick=0.3)


def far_boats():
    for x, y, k in ((586, HOR + 17, 1.0), (972, HOR + 12, 0.8), (1040, HOR + 24, 1.1)):
        dk = mix(PAPER, ramp(RAMP, depth(x)), 0.08)
        cv.stroke([(x - 9 * k, y - 1), (x, y + 0.6), (x + 10 * k, y - 1.6)], 3.2 * k, dk, FILBERT, taper=(0.3, 0.3), tip=0.3, dry=0)
        cv.line((x + 1, y - 1), (x + 1, y - 13 * k), 0.9, dk)
        c = ramp(GLOW, depth(x))
        cv.glow((x - 3 * k, y - 4), 5, c, 0.4)
        cv.dab((x - 3 * k, y - 4), 1.0, LIGHT, soft=0.4, thick=0.2)
        reflection(x - 3 * k, y + 3, 16 * k, c, w0=2.0, w1=4.0, strength=0.5)


def last_boat():
    """The mark: a lid 150 across that sags 28, and three lashes under it."""
    cx, wl = 322.0, 520.0                          # amidships, and the waterline
    A, sag, rise = 78.0, 24.0, 30.0

    def sheer(u):
        return np.array([cx + A * u, wl - rise + sag * (1 - u * u)])

    us = np.linspace(-1, 1, 41)
    lid = np.array([sheer(u) for u in us])
    hull = list(lid) + [(cx + A * 0.86, wl + 1.5), (cx - A * 0.86, wl + 1.5)]
    dk = mix(PAPER, ramp(RAMP, depth(cx)), 0.07)
    # its dark double in the water first, then the boat
    with cv.layer(dk, "multiply", 0.55, blur=1.5):
        cv.fill(np.array([(cx - A * 0.9, wl + 2), (cx + A * 0.9, wl + 2), (cx + A * 0.7, wl + 15), (cx - A * 0.7, wl + 15)], F), dk, FLAT,
                width=5, length=(30, 90), overlap=0.3, frisket=True)
    cv.fill(np.array(hull, F), dk, FLAT, width=6, length=(30, 100), overlap=0.4, streak=0.02, dry=0, jitter=0.005)
    cv.line((cx + A * 0.18, wl - rise + sag - 2), (cx + A * 0.18, wl - rise - 46), 1.5, dk)          # a bare mast
    cv.line((cx + A * 0.18, wl - rise - 44), (cx + A * 0.92, wl - rise + 2), 0.8, dk, opacity=0.8)   # and its stay
    cv.line((cx + A * 0.18, wl - rise - 44), (cx - A * 0.9, wl - rise + 3), 0.8, dk, opacity=0.8)
    pale = mix(LIGHT, ramp(GLOW, depth(cx)), 0.25)
    cv.stroke(lid, 2.6, pale, LINER, taper=(0.12, 0.12), tip=0.3, opacity=0.92)                      # the lid, catching the light
    # three lights along it, and the lashes: the middle one straight down, the outer two splayed
    for u, lean in ((-0.62, -0.30), (0.0, 0.0), (0.62, 0.30)):
        p = sheer(u) + np.array([0, -3.5])
        cv.glow(p, 11, ramp(GLOW, depth(cx)), 0.50)
        cv.glow(p, 4.5, LIGHT, 0.7)
        cv.dab(p, 2.1, LIGHT, soft=0.4, thick=0.4)
        reflection(p[0] + lean * 4, wl + 5, 40 if u else 48, pale, w0=3.0, w1=5.5, strength=1.0, lean=lean)


paint_sky()
paint_water()
shore()
quay_lamps()
far_boats()
last_boat()
vignette(cv)
cv.save_card(OUT, relief=0.4, tooth=0.022)
sys.stdout.write(f"{OUT}: {cv.count} strokes, {time.time() - t0:.0f}s\n")
