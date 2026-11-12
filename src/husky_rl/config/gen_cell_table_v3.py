"""'86 Field: training grid cell -> GPS table, matched fork to fork; cells evenly spaced along each walk.

Anchors (junctions/turns) are pinned to real spots chosen by Peter, given
either as measured placemarks or as labels on the v1 overlay ("old" grid).
Every other open cell is spaced evenly along its lane between two anchors.
Topology and cell order of the training grid are unchanged.

Usage: python3 gen_cell_table_v3.py <husky_rl package dir>
Outputs: field86_cells.yaml, overlay_v3.png
"""
import ast
import math
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

# ---- training grid (read from astar_node.py, no ROS import) ---------------
_src = open(sys.argv[1] + "/astar_node.py").read()
_c = {}
for _n in ast.parse(_src).body:
    if isinstance(_n, ast.Assign) and getattr(_n.targets[0], "id", "") in (
            "OPEN_CELLS", "GRID_WIDTH", "GRID_HEIGHT"):
        _c[_n.targets[0].id] = ast.literal_eval(_n.value)
OPEN = _c["OPEN_CELLS"]
W, H = _c["GRID_WIDTH"], _c["GRID_HEIGHT"]

# ---- measured placemarks --------------------------------------------------
PTS = {
    "Star_1": (42.730705, -73.680592), "Star_2": (42.730527, -73.679088),
    "Star_3": (42.730140, -73.680727), "Star_4": (42.729932, -73.679259),
    "A": (42.730682, -73.680334), "B": (42.730575, -73.679384),
    "E": (42.729967, -73.680030), "F": (42.729919, -73.679610),
}
PIX = {  # pin tips in the Google Earth screenshot (overlay only)
    "Star_1": (234, 274), "Star_2": (668, 315), "Star_3": (209, 496),
    "Star_4": (635, 549), "A": (308, 278), "B": (582, 300),
    "E": (413, 550), "F": (535, 561),
}
LAT0, LON0 = PTS["Star_1"]
R = 6378137.0
KX = math.pi / 180 * R * math.cos(math.radians(LAT0))
KY = math.pi / 180 * R
enu = lambda p: np.array([(p[1] - LON0) * KX, (p[0] - LAT0) * KY])
gps = lambda e: (LAT0 + e[1] / KY, LON0 + e[0] / KX)
P = {k: enu(v) for k, v in PTS.items()}


# ---- v1 ("old") overlay grid, used only to look up label positions --------
def _bilinear(gx, gy):
    u, v = (gx - 1) / 14, (gy - 1) / 8
    return ((1 - u) * (1 - v) * P["Star_1"] + u * (1 - v) * P["Star_2"]
            + (1 - u) * v * P["Star_3"] + u * v * P["Star_4"])


def _inv(p):
    g = np.array([8.0, 5.0])
    for _ in range(50):
        f, h = _bilinear(*g), 1e-5
        J = np.column_stack([(_bilinear(g[0] + h, g[1]) - f) / h,
                             (_bilinear(g[0], g[1] + h) - f) / h])
        g = g + np.linalg.solve(J, p - f)
    return g


def _curve(names):
    xs = [1.0] + [_inv(P[n])[0] for n in names[1:-1]] + [15.0]
    pts = [P[n] for n in names]

    def at(gx):
        for i in range(len(xs) - 1):
            if gx <= xs[i + 1] or i == len(xs) - 2:
                t = (gx - xs[i]) / (xs[i + 1] - xs[i])
                return (1 - t) * pts[i] + t * pts[i + 1]
    return at


_top = _curve(["Star_1", "A", "B", "Star_2"])
_bot = _curve(["Star_3", "E", "F", "Star_4"])


def old(gx, gy):
    """Real position of label 'gx,gy' on the v1 overlay."""
    v = (gy - 1) / 8
    return (1 - v) * _top(gx) + v * _bot(gx)


def toward(a, b, d):
    return a + (b - a) / np.linalg.norm(b - a) * d


# ---- forks: training cell -> real position (fixed) ------------------------
FORKS = {
    (1, 1): ("Star_1", P["Star_1"]), (15, 1): ("Star_2", P["Star_2"]),
    (1, 9): ("Star_3", P["Star_3"]), (15, 9): ("Star_4", P["Star_4"]),
    (4, 1): ("A", P["A"]), (12, 1): ("B", P["B"]),
    (7, 9): ("E", P["E"]),            # = v1 label 8,9 (within ~2 m)
    (10, 9): ("F", P["F"]),           # = v1 label 12,9 (within ~2 m)
    (1, 5): ("J1", old(1, 7)),
    (4, 3): ("J2", old(3, 3)),
    (12, 6): ("J3", old(13, 7)),
}
ANCHORS = FORKS  # used by the YAML/overlay sections


def run(a, b):
    (x0, y0), (x1, y1) = a, b
    n = max(abs(x1 - x0), abs(y1 - y0))
    return [(x0 + (x1 - x0) * i // n, y0 + (y1 - y0) * i // n)
            for i in range(n + 1)]


def chain(*corners):
    out = [corners[0]]
    for a, b in zip(corners, corners[1:]):
        out += run(a, b)[1:]
    return out


# ---- paths between forks: (training cells, real shape points between ends) -
# Shape points are the turn spots Peter picked (T1..T7); they bend the path
# to follow the walk but are not pinned to any cell.
PATHS = {
    "top: Star_1-A": (chain((1, 1), (4, 1)), []),
    "top: A-B": (chain((4, 1), (12, 1)), []),
    "top: B-Star_2": (chain((12, 1), (15, 1)), []),
    "left: Star_1-J1": (chain((1, 1), (1, 5)), []),
    "left: J1-Star_3": (chain((1, 5), (1, 9)), []),
    "bottom: Star_3-E": (chain((1, 9), (7, 9)), []),
    "bottom: E-F": (chain((7, 9), (10, 9)), []),
    "bottom: F-Star_4": (chain((10, 9), (15, 9)), []),
    "mid: A-J2": (chain((4, 1), (4, 3)), []),
    "mid: J2-J1": (chain((4, 3), (3, 3), (3, 5), (2, 5), (1, 5)),
                   [("T2", old(2, 5))]),
    "mid: J2-E": (chain((4, 3), (5, 3), (5, 7), (7, 7), (7, 9)),
                  [("T4", old(5, 7)), ("T5", old(6, 8))]),
    "right: B-J3": (chain((12, 1), (12, 6)), []),
    "right: Star_2-J3": (chain((15, 1), (15, 6), (12, 6)),
                         [("T7", old(14, 4))]),
    "right: J3-F": (chain((12, 6), (10, 6), (10, 9)),
                    [("T6", old(12, 8))]),
}
LANES = {k: v[0] for k, v in PATHS.items()}


def along(poly, s):
    """Point at arc length s along a polyline."""
    for a, b in zip(poly, poly[1:]):
        seg = np.linalg.norm(b - a)
        if s <= seg or b is poly[-1]:
            return a + (b - a) * min(s / seg, 1.0)
        s -= seg


# ---- place cells evenly by arc length -------------------------------------
pos = {c: e for c, (_, e) in FORKS.items()}
path_len = {}
for name, (cells, shape) in PATHS.items():
    assert cells[0] in FORKS and cells[-1] in FORKS, name
    for a, b in zip(cells, cells[1:]):
        assert a in OPEN and b in OPEN and abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1, (name, a, b)
    poly = [FORKS[cells[0]][1]] + [p for _, p in shape] + [FORKS[cells[-1]][1]]
    L = sum(np.linalg.norm(b - a) for a, b in zip(poly, poly[1:]))
    path_len[name] = (L, poly)
    n = len(cells) - 1
    for i, c in enumerate(cells[1:-1], 1):
        assert c not in pos, f"{c} placed twice"
        pos[c] = along(poly, L * i / n)
missing = OPEN - set(pos)
extra = set(pos) - OPEN
assert not missing and not extra, (missing, extra)

edges = {frozenset((a, b)) for a in OPEN for b in OPEN
         if abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1}
lane_edges = [frozenset(e) for cs in LANES.values() for e in zip(cs, cs[1:])]
assert set(lane_edges) == edges and len(lane_edges) == len(edges)

# ---- report ---------------------------------------------------------------
print(f"{len(pos)} open cells placed, {len(edges)} grid edges, all covered once")
print(f"{'path':20s} steps  walk_m  step_m  max_straight_gap_m")
for name, cells in LANES.items():
    L, _ = path_len[name]
    n = len(cells) - 1
    gaps = [np.linalg.norm(pos[b] - pos[a]) for a, b in zip(cells, cells[1:])]
    mark = "  <-- short" if min(gaps) < 4 else ("  <-- long" if max(gaps) > 15 else "")
    print(f"{name:20s} {n:5d} {L:7.1f} {L/n:7.1f}   {min(gaps):5.1f}-{max(gaps):5.1f}{mark}")

# ---- YAML -----------------------------------------------------------------
cells_out = {}
for c in sorted(pos, key=lambda c: (c[1], c[0])):
    lat, lon = gps(pos[c])
    entry = {"lat": round(float(lat), 7), "lon": round(float(lon), 7),
             "east_m": round(float(pos[c][0]), 2),
             "north_m": round(float(pos[c][1]), 2)}
    if c in ANCHORS:
        entry["anchor"] = ANCHORS[c][0]
    cells_out[f"{c[0]},{c[1]}"] = entry
with open("field86_cells.yaml", "w") as f:
    yaml.safe_dump({
        "description": "'86 Field: training grid cell centres (open cells "
                       "only), matched fork to fork; cells evenly spaced along each walk. Generated by "
                       "gen_cell_table_v3.py.",
        "local_frame": {"origin": "Star_1", "lat0": LAT0, "lon0": LON0,
                        "axes": "east_m, north_m"},
        "grid": {"width": W, "height": H},
        "goals": {"0": "1,1", "1": "15,1", "2": "1,9", "3": "15,9"},
        "cells": cells_out}, f, sort_keys=False)

# ---- overlay --------------------------------------------------------------
names = list(PIX)
A_ = np.array([[*P[n], 1.0] for n in names])
M, *_ = np.linalg.lstsq(A_, np.array([PIX[n] for n in names], float), rcond=None)
px = lambda e: tuple(np.array([*e, 1.0]) @ M)

img = Image.open("screenshot.jpg").convert("RGB")
d = ImageDraw.Draw(img, "RGBA")
font = ImageFont.load_default()
group_col = {"top": (0, 230, 255), "left": (255, 220, 0), "bottom": (0, 255, 120),
             "mid": (255, 80, 255), "right": (255, 140, 0)}
for name, cells in LANES.items():
    col = group_col[name.split(":")[0]]
    d.line([px(pos[c]) for c in cells], fill=(*col, 230), width=3)
for c, e in pos.items():
    x, y = px(e)
    if c in ANCHORS:
        d.ellipse([x-6, y-6, x+6, y+6], fill=(255, 40, 40, 255), outline=(0, 0, 0, 255))
        d.text((x+7, y+2), ANCHORS[c][0], fill=(255, 255, 255, 255), font=font,
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    else:
        d.ellipse([x-3, y-3, x+3, y+3], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
    d.text((x+5, y-12), f"{c[0]},{c[1]}", fill=(255, 255, 0, 255), font=font,
           stroke_width=2, stroke_fill=(0, 0, 0, 255))
img.save("overlay_v3.png")
print("wrote field86_cells.yaml, overlay_v3.png")
