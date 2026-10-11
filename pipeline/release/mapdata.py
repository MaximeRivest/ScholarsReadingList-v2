"""The map of science, second format: everything the site's map engine draws, for one release.

    search/.venv/bin/python pipeline/release/mapdata.py --rows ROWS --xy XY --atlas ATLAS --corpus CORPUS --out OUT

OUT (tiles/<release>/v2) holds, all static and immutable (the release id is in the URL):

map.json     what is there: levels, which tiles exist (one bit per tile, compact()), encodings, fields and subfields,
             label font sizes. The engine reads it first.

d/z/x/y.bin  DENSITY of every paper (nothing sampled), a pyramid of 256 x 256 pixel tiles, levels
             0..DMAX (level z: the unit square is 256 * 2^z pixels wide). Each tile has a 1-pixel
             apron copied from its neighbours (258 x 258), so smooth filtering has no seams.
             gzip of three planes, row-major:
               count   uint16 LE  round(log2(1 + papers) * 3600)   (0 = empty)
               sub     uint8      dominant subfield code, inside the dominant field
               purity  uint8      share of the pixel's papers in the dominant field, * 255
             Colours are the engine's choice (per theme), from the field of `sub`.

p/z/x/y.bin  POINTS: every paper, in a quadtree (tiles.py's assign_levels) whose order is
             importance (citations, then a fixed hash): a tile keeps the CAP most cited papers of
             its square not taken by a coarser level. So level 0 is the 32,768 most cited
             papers of all science, and zooming in only adds less cited ones; a paper never
             moves. gzip of uint16 LE quadruples per point, most cited first:
               x, y    position inside the tile, / 65535
               info    year (12 bits) | citation class << 12 (3 bits: 0, 1-3, 4-15, ...,
                       >= 4096) | readable on the site << 15
               sub     subfield code
i/z/x/y.bin  the W-numbers of p/z/x/y.bin, uint64 LE, same order (fetched only on a click).

labels.json  NAMES that never collide: fields, subfields, OpenAlex topics, and the titles of
             the most cited papers. Each has the zoom range where it shows (s0 <= s < s1, s =
             screen pixels per unit square / 1024): placed greedily from the most important
             down, each label shows from the first zoom where it is big enough and overlaps no
             label already placed. Group labels sit at the densest point of their largest
             cluster (not the mean, which can fall in a neighbour's territory).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiles import assign_levels  # noqa: E402

TILE, DMAX, CAP = 256, 8, 32768
LOGQ = 3600.0
FONT = {"field": 15.0, "subfield": 13.0, "topic": 12.0, "paper": 11.5}
S_MIN = 0.45                    # the whole map in ~460 px: the smallest the explorer shows it
CHAR = 0.56                     # average glyph width / font size (sans, semibold)
T0 = time.time()


def log(m: str) -> None:
    print(f"{time.time() - T0:7.0f}s {m}", flush=True)


def splitmix_u(ids: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


# ------------------------------------------------------------------------------------ density
def level_pixels(pix: np.ndarray, sub: np.ndarray, cnt: np.ndarray, field_of: np.ndarray):
    """(pixel, subfield, count) rows sorted by pixel -> per pixel: total, dominant sub, purity."""
    starts = np.flatnonzero(np.r_[True, pix[1:] != pix[:-1]])
    upix = pix[starts]
    total = np.add.reduceat(cnt, starts)
    g = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(pix)]))
    f = field_of[sub]
    # counts per (pixel, field)
    o = np.lexsort((f, g))
    gf, ff, cf = g[o], f[o], cnt[o]
    s2 = np.flatnonzero(np.r_[True, (gf[1:] != gf[:-1]) | (ff[1:] != ff[:-1])])
    fg, fcode, fsum = gf[s2], ff[s2], np.add.reduceat(cf, s2)
    o2 = np.lexsort((-fsum, fg))                         # per pixel, the largest field first
    first = o2[np.r_[True, fg[o2][1:] != fg[o2][:-1]]]
    dom_field = np.empty(len(starts), np.int64); dom_field[fg[first]] = fcode[first]
    dom_count = np.empty(len(starts), np.int64); dom_count[fg[first]] = fsum[first]
    # dominant subfield inside the dominant field
    m = f == dom_field[g]
    gm, sm, cm = g[m], sub[m], cnt[m]
    o3 = np.lexsort((-cm, gm))
    first3 = o3[np.r_[True, gm[o3][1:] != gm[o3][:-1]]]
    dom_sub = np.empty(len(starts), np.int64); dom_sub[gm[first3]] = sm[first3]
    purity = np.round(255 * dom_count / total).astype(np.uint8)
    return upix, total, dom_sub.astype(np.uint8), purity


def write_tiles(job):
    out, z, keys, lx, ly, c16, sub, pur = job
    side = TILE + 2
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    for s, e in zip(starts, ends):
        k = int(keys[s]); tx, ty = k & 0xFFFFF, k >> 20
        a = np.zeros((side, side), np.uint16); b = np.zeros((side, side), np.uint8); c = np.zeros((side, side), np.uint8)
        a[ly[s:e], lx[s:e]] = c16[s:e]; b[ly[s:e], lx[s:e]] = sub[s:e]; c[ly[s:e], lx[s:e]] = pur[s:e]
        d = out / "d" / str(z) / str(tx)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{ty}.bin").write_bytes(gzip.compress(a.astype("<u2").tobytes() + b.tobytes() + c.tobytes(), 6))
    return len(starts)


def density(p: np.ndarray, sub: np.ndarray, field_of: np.ndarray, out: Path, procs: int) -> dict:
    W = TILE << DMAX
    px = np.minimum((p[:, 0] * W).astype(np.int64), W - 1)
    py = np.minimum((p[:, 1] * W).astype(np.int64), W - 1)
    key = ((py * W + px) << 8) | sub.astype(np.int64)
    key.sort()
    s = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    cnt = np.diff(np.r_[s, len(key)]).astype(np.int64)
    key = key[s]
    del px, py
    tiles: dict[int, list] = {}
    stats = []
    for z in range(DMAX, -1, -1):
        Wz = TILE << z
        if z < DMAX:   # parent pixels: halve both coordinates, merge equal (pixel, sub)
            pix = key >> 8
            x, y = (pix % (2 * Wz)) >> 1, (pix // (2 * Wz)) >> 1
            key = ((y * Wz + x) << 8) | (key & 0xFF)
            o = np.argsort(key, kind="stable")
            key, cnt = key[o], cnt[o]
            s = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
            cnt = np.add.reduceat(cnt, s); key = key[s]
        upix, total, dsub, pur = level_pixels(key >> 8, (key & 0xFF).astype(np.int64), cnt, field_of)
        c16 = np.minimum(np.round(np.log2(1.0 + total) * LOGQ), 65535).astype(np.uint16)
        x, y = upix % Wz, upix // Wz
        # every pixel goes to its own tile, and to the apron of the neighbours it touches
        parts = []
        lx0, ly0 = x % TILE, y % TILE
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                m = np.ones(len(x), bool)
                if dx == -1: m &= lx0 == 0
                if dx == 1: m &= lx0 == TILE - 1
                if dy == -1: m &= ly0 == 0
                if dy == 1: m &= ly0 == TILE - 1
                tx, ty = x[m] // TILE + dx, y[m] // TILE + dy
                ok = (tx >= 0) & (ty >= 0) & (tx < (Wz // TILE)) & (ty < (Wz // TILE))
                i = np.flatnonzero(m)[ok]
                parts.append(((ty[ok] << 20) | tx[ok], lx0[i] + 1 - dx * TILE, ly0[i] + 1 - dy * TILE, i, dx == 0 and dy == 0))
        own = set(np.unique(parts[4][0]).tolist())   # (0, 0) is the 5th combination
        keys = np.concatenate([q[0] for q in parts])
        lx = np.concatenate([q[1] for q in parts]).astype(np.int64)
        ly = np.concatenate([q[2] for q in parts]).astype(np.int64)
        idx = np.concatenate([q[3] for q in parts])
        keep = np.isin(keys, np.fromiter(own, np.int64))   # no tile made of apron only
        keys, lx, ly, idx = keys[keep], lx[keep], ly[keep], idx[keep]
        o = np.argsort(keys, kind="stable")
        keys, lx, ly, idx = keys[o], lx[o], ly[o], idx[o]
        # split into jobs at tile boundaries
        bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        cuts = bounds[np.linspace(0, len(bounds), min(len(bounds), procs * 4) + 1).astype(int)[1:-1]] if len(bounds) > 1 else []
        jobs, a = [], 0
        for b in list(cuts) + [len(keys)]:
            jobs.append((out, z, keys[a:b], lx[a:b], ly[a:b], c16[idx[a:b]], dsub[idx[a:b]], pur[idx[a:b]]))
            a = b
        with ProcessPoolExecutor(procs) as ex:
            n = sum(ex.map(write_tiles, jobs))
        tiles[z] = sorted(own)
        stats.append({"level": z, "pixels": int(len(upix)), "tiles": n, "max_papers_per_pixel": int(total.max())})
        log(f"density level {z}: {len(upix):,} pixels, {n:,} tiles, max {int(total.max()):,} per pixel")
    return {"levels": DMAX + 1, "tile": TILE, "apron": 1, "log_scale": LOGQ, "stats": stats[::-1],
            "tiles": {str(z): [[k & 0xFFFFF, k >> 20] for k in ks] for z, ks in sorted(tiles.items())}}


# ------------------------------------------------------------------------------------- points
def write_points(job):
    out, items = job
    for (z, x, y), pts, ids in items:
        for kind, data in (("p", gzip.compress(pts.tobytes(), 6)), ("i", ids.tobytes())):
            d = out / kind / str(z) / str(x)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{y}.bin").write_bytes(data)
    return len(items)


def points(p, work, year, smr, sub, cites, out: Path, procs: int) -> dict:
    # importance: citations, ties broken by a fixed hash; rank 0 = most important
    o = np.lexsort((splitmix_u(work), -cites))
    rank = np.empty(len(o), np.float64); rank[o] = np.arange(len(o))
    q = p.clip(0, np.nextafter(np.float32(1), np.float32(0)))
    level = assign_levels(q, rank, CAP).astype(np.int64)
    side = np.left_shift(np.int64(1), level)
    tx, ty = (q[:, 0] * side).astype(np.int64), (q[:, 1] * side).astype(np.int64)
    lx = np.round((q[:, 0] * side - tx) * 65535).astype(np.uint16)
    ly = np.round((q[:, 1] * side - ty) * 65535).astype(np.uint16)
    cclass = np.zeros(len(cites), np.uint16)
    for k, lo in enumerate([1, 4, 16, 64, 256, 1024, 4096], 1):
        cclass[cites >= lo] = k
    info = year.clip(0, 4095).astype(np.uint16) | (cclass << 12) | (smr.astype(np.uint16) << 15)
    key = (level << 40) | (ty << 20) | tx
    o = np.lexsort((rank, key))
    key, lx, ly, info, s16, ids = key[o], lx[o], ly[o], info[o], sub[o].astype(np.uint16), work[o]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]); ends = np.r_[starts[1:], len(key)]
    items, counts = [], {}
    for s, e in zip(starts, ends):
        k = int(key[s]); z, y, x = k >> 40, (k >> 20) & 0xFFFFF, k & 0xFFFFF
        items.append(((z, x, y), np.stack([lx[s:e], ly[s:e], info[s:e], s16[s:e]], 1).astype("<u2"),
                      ids[s:e].astype("<u8")))
        counts[f"{z}/{x}/{y}"] = int(e - s)
    with ProcessPoolExecutor(procs) as ex:
        n = sum(ex.map(write_points, [(out, items[i::procs * 4]) for i in range(procs * 4)]))
    levels = int(level.max()) + 1
    log(f"points: {n:,} tiles, {levels} levels")
    return {"capacity": CAP, "levels": levels, "tiles": counts,
            "per_level": [int((level == z).sum()) for z in range(levels)],
            "info": "year (12 bits) | citation class << 12 (0: none, 1: 1-3, 2: 4-15, 3: 16-63, 4: 64-255, "
                    "5: 256-1023, 6: 1024-4095, 7: >= 4096) | readable << 15"}


# ------------------------------------------------------------------------------------- labels
def box(text: str, kind: str) -> tuple[float, float]:
    f = FONT[kind]
    w, h = len(text) * f * CHAR + 10, f * 1.3 + 6
    return (w, 2 * h + 8) if kind == "paper" else (w, h)     # a paper's title hangs under its dot


def anchor(q: np.ndarray, r: float, rng) -> tuple[float, float, float, float]:
    """Densest point of a set (radius r), its share of the set within r, and the cluster's radius."""
    s = q if len(q) <= 20000 else q[rng.choice(len(q), 20000, replace=False)]
    t = cKDTree(s)
    n = t.query_ball_point(s, r, return_length=True)
    best = int(np.argmax(n))
    near = s[t.query_ball_point(s[best], r)]
    c = np.median(near, 0)
    d = np.linalg.norm(s - c, axis=1)
    within = d < 3 * r
    r50 = float(np.median(d[within])) if within.any() else r
    return float(c[0]), float(c[1]), float(within.mean()), max(r50, r / 4)


def place_labels(cands: list[dict]) -> list[dict]:
    """cands sorted by priority; each: x, y (unit square), w, h (px), s0, s1 (zoom range it may show in).

    Zooming in only moves names apart (boxes keep their size in pixels), so two names that do not
    overlap at zoom s never overlap beyond it. Walk zoom steps s_k = S_MIN * 2^(k/4) upwards; at each
    step the names that are on or about to come on (s0 < s_k+1) claim screen space in priority order,
    checked at s_k, the tightest zoom of the step (a grid of 128 px cells). A name that is free comes on
    at max(s0, s_k); a name that is on and now blocked by a more important one goes off for good (no
    flicker). Names that would show for less than a factor 1.6 of zoom are dropped.
    """
    n = len(cands)
    start = np.full(n, np.nan); end = np.array([c["s1"] for c in cands], float)
    s0 = np.array([c["s0"] for c in cands]); dead = np.zeros(n, bool)
    X = np.array([c["x"] for c in cands]); Y = np.array([c["y"] for c in cands])
    Wd = np.array([c["w"] for c in cands]); Ht = np.array([c["h"] for c in cands])
    k, CELL = 0, 128.0
    while True:
        s = S_MIN * 2 ** (k / 4)
        if s > 4096:
            break
        nxt = s * 2 ** 0.25
        act = np.flatnonzero(~dead & (s0 < nxt) & (end > s))
        k += 1
        if not len(act):
            continue
        grid: dict[tuple[int, int], list[int]] = {}
        px, py = X[act] * 1024 * s, Y[act] * 1024 * s
        for j, i in enumerate(act):          # act is in priority order (cands are sorted)
            x0, x1 = px[j] - Wd[i] / 2, px[j] + Wd[i] / 2
            y0, y1 = py[j] - Ht[i] / 2, py[j] + Ht[i] / 2
            cells = [(cx, cy) for cx in range(int(x0 // CELL), int(x1 // CELL) + 1)
                     for cy in range(int(y0 // CELL), int(y1 // CELL) + 1)]
            hit = False
            for cell in cells:
                for (a0, a1, b0, b1) in grid.get(cell, ()):
                    if a0 < x1 and x0 < a1 and b0 < y1 and y0 < b1:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                if not np.isnan(start[i]):      # was on: off for good; not on yet: wait for room
                    end[i] = min(end[i], s)
                    dead[i] = True
                continue
            if np.isnan(start[i]):
                start[i] = max(s0[i], s)
            box_ = (x0, x1, y0, y1)
            for cell in cells:
                grid.setdefault(cell, []).append(box_)
    out = []
    for i, c in enumerate(cands):
        if np.isnan(start[i]) or end[i] / start[i] < 1.6:
            continue
        out.append(dict(c, s0=float(start[i]), s1=float(end[i])))
    return out


def labels(p, sub, field_of, topic, topic_names, cites, work, atlas, corpus: Path, cache: Path) -> list[dict]:
    if cache.exists():
        cands = json.loads(cache.read_text())
        log(f"labels: {len(cands):,} candidates from {cache.name}")
        return finish_labels(cands)
    rng = np.random.default_rng(0)
    fields = [f["name"] for f in atlas["fields"]]
    subs = atlas["subfields"]
    cands = []
    log("labels: anchors")
    groups = [("field", field_of[sub], range(1, len(fields)), lambda i: fields[i], 0.02),
              ("subfield", sub.astype(np.int64), range(1, len(subs)), lambda i: subs[i]["name"], 0.008),
              ("topic", topic.astype(np.int64), range(1, len(topic_names)), lambda i: topic_names[i], 0.0025)]
    for kind, code, ids, name, r in groups:
        o = np.argsort(code, kind="stable"); cs = code[o]
        st = np.searchsorted(cs, np.arange(code.max() + 2))
        for i in ids:
            if i >= len(st) - 1 or st[i + 1] - st[i] < (2000 if kind == "topic" else 50):
                continue
            q = p[o[st[i]:st[i + 1]]]
            x, y, share, r50 = anchor(q, r, rng)
            mass = share * len(q)
            w, h = box(name(i), kind)
            # shows once its main cluster is ~35 px across, until it fills the screen
            s0, s1 = 35 / (2 * r50 * 1024), 900 / (2 * r50 * 1024)
            fcode = int(field_of[i]) if kind == "subfield" else \
                (i if kind == "field" else int(np.bincount(field_of[sub[o[st[i]:st[i + 1]]]]).argmax()))
            cands.append({"t": name(i), "k": kind, "x": x, "y": y, "w": w, "h": h, "s0": max(s0, 0.25),
                          "s1": max(s1, 2.0) if kind != "field" else max(s1, 3.0), "m": int(mass), "f": fcode})
        log(f"labels: {kind} anchors done ({sum(c['k'] == kind for c in cands)})")
    # papers: the most cited of all science, and the most cited of each topic
    top = np.argsort(-cites)[:40000]
    o = np.lexsort((-cites, topic)); tt = topic[o]
    firsts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    per_topic = np.concatenate([o[f:f + 3] for f in firsts])
    pick = np.unique(np.r_[top, per_topic[cites[per_topic] >= 20]])
    db = sqlite3.connect(f"file:{corpus / 'works.sqlite'}?mode=ro", uri=True)
    titles = {}
    ws = [int(w) for w in work[pick]]
    for a in range(0, len(ws), 900):
        chunk = ws[a:a + 900]
        for w, t in db.execute(f"select id, title from works where id in ({','.join('?' * len(chunk))})", chunk):
            titles[w] = t
    log(f"labels: {len(pick):,} paper candidates")
    for i in pick:
        t = " ".join(re.sub(r"<[^>]+>", " ", titles.get(int(work[i])) or "").split())
        if not t:
            continue
        short = t if len(t) <= 52 else t[:50].rsplit(" ", 1)[0].rstrip(",:;-") + "…"
        w, h = box(short, "paper")
        cands.append({"t": short, "full": t if short != t else None, "k": "paper", "id": f"W{int(work[i])}",
                      "x": float(p[i, 0]), "y": float(p[i, 1]), "w": w, "h": h,
                      "s0": 6.0, "s1": 1e6, "m": int(cites[i]), "f": int(field_of[sub[i]])})
    cache.write_text(json.dumps(cands))
    return finish_labels(cands)


def finish_labels(cands: list[dict]) -> list[dict]:
    order = {"field": 0, "subfield": 1, "topic": 2, "paper": 3}
    cands.sort(key=lambda c: (order[c["k"]], -c["m"]))
    log(f"labels: placing {len(cands):,}")
    placed = place_labels(cands)
    out = []
    for c in placed:
        d = {"t": c["t"], "k": c["k"], "x": round(c["x"], 6), "y": round(c["y"], 6), "s0": round(c["s0"], 3),
             "s1": round(min(c["s1"], 1e5), 3), "m": c["m"], "f": c["f"]}
        if c.get("id"):
            d["id"] = c["id"]
        if c.get("full"):
            d["full"] = c["full"]
        out.append(d)
    by = {k: sum(c["k"] == k for c in out) for k in order}
    log(f"labels: placed {by}")
    return out


# --------------------------------------------------------------------------------------- main
TITLE_LEVEL = 4      # study titles in a 16 x 16 grid of files


def split_labels(out: Path, lab: list[dict]) -> None:
    """What the explorer downloads: names.json (fields, subfields, topics: ~0.5 MB, at once) and the titles of
    famous studies by place, t/4/x/y.json, fetched for the cells on screen once zoomed in far enough (8 MB in one
    file took ~10 s through the public tunnel and delayed every tap). labels.json stays for older pages."""
    names = [{k: l[k] for k in ("t", "k", "x", "y", "s0", "s1")} for l in lab if l["k"] != "paper"]
    side = 1 << TITLE_LEVEL
    cells: dict[tuple[int, int], list] = {}
    for l in lab:
        if l["k"] != "paper":
            continue
        cx, cy = min(side - 1, int(l["x"] * side)), min(side - 1, int(l["y"] * side))
        cells.setdefault((cx, cy), []).append({"t": l["t"], "x": l["x"], "y": l["y"], "s0": l["s0"], "s1": l["s1"], "id": l["id"]})
    for (cx, cy), items in cells.items():
        d = out / "t" / str(TITLE_LEVEL) / str(cx)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cy}.json").write_text(json.dumps(items, ensure_ascii=False, separators=(",", ":")))
    s_min = min((l["s0"] for l in lab if l["k"] == "paper"), default=1e9)
    (out / "names.json").write_text(json.dumps(
        {"names": names, "titles": {"level": TITLE_LEVEL, "s_min": s_min, "cells": sorted(cells)}},
        ensure_ascii=False, separators=(",", ":")))


def compact(doc: dict) -> dict:
    """Tile lists as one bit per tile and level (base64, row-major, bit i of byte i>>3 = tile i): map.json
    shrinks from ~630 KB to ~30 KB, and it is the first thing the explorer must download."""
    import base64

    def mask(z: int, keys) -> str:
        side = 1 << z
        bits = np.zeros(side * side, np.uint8)
        for x, y in keys:
            bits[y * side + x] = 1
        return base64.b64encode(np.packbits(bits, bitorder="little").tobytes()).decode()
    d = doc.get("density")
    if d and "tiles" in d:
        d["mask"] = {z: mask(int(z), xy) for z, xy in d.pop("tiles").items()}
    q = doc.get("points")
    if q and "tiles" in q:
        by: dict[int, list] = {}
        for k in q.pop("tiles"):
            z, x, y = map(int, k.split("/"))
            by.setdefault(z, []).append((x, y))
        q["mask"] = {str(z): mask(z, xy) for z, xy in sorted(by.items())}
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--xy", type=Path, required=True, help="the release's position table (tiles.py's xy.npy)")
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True, action="append",
                    help="a details corpus; repeat it for the base and the daily updates corpus (pipeline/updates)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--only", choices=["density", "points", "labels", "compact", "split"], help="rebuild one part (development)")
    a = ap.parse_args()
    if (a.out / "map.json").exists() and not a.only:
        raise SystemExit(f"{a.out} is finished")
    out = a.out if a.only else a.out.with_name(a.out.name + ".tmp")
    out.mkdir(parents=True, exist_ok=True)
    atlas = json.loads((a.atlas / "meta.json").read_text())
    fields = [f["name"] for f in atlas["fields"]]
    field_of = np.array([fields.index(s["field"]) if s["field"] in fields else 0 for s in atlas["subfields"]], np.int64)
    keep = np.load(a.rows / "keep.npy")
    xy = np.load(a.xy, mmap_mode="r")
    rows = np.flatnonzero(keep & ~np.isnan(np.asarray(xy[:, 0])))
    p = np.asarray(xy[rows], np.float64)
    sub = np.load(a.rows / "subfield.npy")[rows].astype(np.uint8)
    work = np.load(a.rows / "work.npy")[rows].astype(np.int64)
    log(f"{len(rows):,} papers on the map")
    # per-paper facts from each corpus' map-meta; topic codes are per corpus: the first corpus' names, then the
    # names only later corpora use, appended
    found = np.zeros(len(work), bool)
    cites = np.zeros(len(work), np.int64)
    topic = np.zeros(len(work), np.uint16)
    topic_names: list[str] = []
    for c in a.corpus:
        mm = c / "map-meta"
        cid = np.load(mm / "id.npy", mmap_mode="r")
        names = json.loads((mm / "topics.json").read_text())
        if not topic_names:
            topic_names = list(names)
        known = {n: i for i, n in enumerate(topic_names)}
        code = []
        for n in names:
            if n not in known:
                known[n] = len(topic_names)
                topic_names.append(n)
            code.append(known[n])
        code = np.array(code, np.uint16)
        j = np.searchsorted(cid, work).clip(0, len(cid) - 1)
        f = (np.asarray(cid[j]) == work) & ~found
        cites[f] = np.load(mm / "citations.npy", mmap_mode="r")[j[f]]
        topic[f] = code[np.load(mm / "topic.npy", mmap_mode="r")[j[f]]]
        found |= f
    log(f"corpus facts for {found.mean():.4%} of them")
    doc = json.loads((out / "map.json").read_text()) if a.only and (out / "map.json").exists() else {}
    doc.update({"format": 2, "atlas": atlas["id"], "papers": int(len(rows)),
                "fields": [{"name": f["name"], "colour": f["colour"]} for f in atlas["fields"]],
                "subfields": [{"name": s["name"], "field": int(field_of[i])} for i, s in enumerate(atlas["subfields"])],
                "fonts": FONT, "label_scale": "s = screen pixels per unit square / 1024"})
    if a.only == "split":
        split_labels(out, json.loads((out / "labels.json").read_text()))
        return log("names.json and t/ written")
    if a.only == "compact":
        (out / "map.json").write_text(json.dumps(compact(json.loads((out / "map.json").read_text())), separators=(",", ":")))
        return log("map.json compacted")
    if a.only in (None, "density"):
        doc["density"] = density(p, sub, field_of, out, a.procs)
    if a.only in (None, "points"):
        doc["points"] = points(p.astype(np.float32), work, np.load(a.rows / "year.npy")[rows],
                               np.load(a.rows / "smr.npy")[rows], sub, cites, out, a.procs)
    if a.only in (None, "labels"):
        lab = labels(p, sub, field_of, topic, topic_names, cites, work, atlas, a.corpus[0], out / "label-candidates.json")
        (out / "labels.json").write_text(json.dumps(lab, ensure_ascii=False, separators=(",", ":")))
        split_labels(out, lab)
        doc["labels"] = len(lab)
    doc["seconds"] = round(time.time() - T0)
    (out / "map.json").write_text(json.dumps(compact(doc), separators=(",", ":")))
    if not a.only:
        os.replace(out, a.out)
    log("done")


if __name__ == "__main__":
    main()
