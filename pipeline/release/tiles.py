"""Map tiles: every kept paper of a release, in nested levels of detail, as static files.

    cd search && uv run python ../pipeline/release/tiles.py --rows ROWS --positions POS --atlas A --out OUT

A quadtree with a fixed capacity per tile (CAP points). Every paper has a fixed number
u = splitmix64(W-number) in [0, 1) (the atlas sample's hash). Level z is a 2^z x 2^z grid over the
unit square. Going down from level 0, each tile keeps the CAP papers with the smallest u among
the papers in its square not taken by a coarser level; the rest go to its four children. The
deepest level (MAX_LEVEL) keeps everything left. So:
  - every tile holds at most CAP points (except possibly at MAX_LEVEL), dense regions go deeper;
  - to draw a view, load the tiles that intersect it at every level down to the one matching the
    zoom: zooming in only adds papers, a paper never moves or disappears;
  - within a tile points are sorted by u, so any prefix of a tile is a uniform sample of it.
A tile that is absent from index.json has no papers (nothing below it either).

Per tile, two little-endian files:
  z/x/y.pts   8 bytes per point: x, y (uint16, position inside the tile, 0..65535),
              year (uint16; bit 15 = the site can open it), subfield (uint16, atlas code)
  z/x/y.ids   uint64 W-number per point (fetched only to identify a clicked point)
index.json    capacity, levels, per-tile counts, code lists (field colours from the atlas build)
xy.npy        float32 [rows, 2] position of every row of the release (NaN = shard not placed yet):
              the lookup table the API uses for any embedded paper, publication or not
Papers whose shard has no position yet are left out and counted in index.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "citewalk" / "science"))
from sample import splitmix_u  # noqa: E402

CAP, MAX_LEVEL = 32768, 14


def assign_levels(p: np.ndarray, u: np.ndarray, cap: int = CAP, max_level: int = MAX_LEVEL) -> np.ndarray:
    """Quadtree level of each point: a tile keeps its `cap` smallest-u points not taken above."""
    level = np.full(len(u), -1, np.int8)
    left = np.arange(len(u))
    for z in range(max_level + 1):
        if not len(left):
            break
        if z == max_level:
            level[left] = z
            break
        side = 1 << z
        cell = (p[left, 1] * side).astype(np.int64) * side + (p[left, 0] * side).astype(np.int64)
        o = np.lexsort((u[left], cell))
        c = cell[o]
        first = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1]
        rank = np.arange(len(c)) - np.repeat(first, np.diff(np.r_[first, len(c)]))
        take = o[rank < cap]
        level[left[take]] = z
        left = np.sort(left[o[rank >= cap]])
    return level


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--positions", type=Path, required=True, help="positions/<atlas id> directory")
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=CAP)
    a = ap.parse_args()
    if (a.out / "index.json").exists():
        raise SystemExit(f"{a.out} is finished")
    t0 = time.time()
    meta = json.loads((a.rows / "meta.json").read_text())
    shards = json.loads((a.rows / "shards.json").read_text())
    keep = np.load(a.rows / "keep.npy")
    N = meta["rows"]
    xy = np.full((N, 2), np.nan, np.float32)
    missing_shards = 0
    for s in shards:
        f = a.positions / s["dir"] / "xy.npy"
        if f.exists():
            xy[s["first"]:s["first"] + s["rows"]] = np.load(f)
        else:
            missing_shards += 1
    # references keep their fitted coordinates (exactly the atlas's own dots)
    ref = np.load(a.atlas / "reference.npz")
    work = np.load(a.rows / "work.npy")
    by_work = np.load(a.rows / "by_work.npy")
    j = np.searchsorted(work[by_work], ref["work"])
    j = j.clip(0, len(by_work) - 1)
    hit = work[by_work[j]] == ref["work"]
    xy[by_work[j[hit]]] = ref["coords"][hit]
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "xy.npy", xy)        # the release's position table, aligned with its rows (NaN = not placed)
    have = keep & ~np.isnan(xy[:, 0])
    rows = np.flatnonzero(have)
    print(f"{len(rows):,} kept papers with a position ({int(keep.sum()) - len(rows):,} kept without; "
          f"{missing_shards} shards not placed yet) {time.time() - t0:.0f}s", flush=True)

    u = splitmix_u(work[rows])
    p = xy[rows].clip(0, np.nextafter(np.float32(1), np.float32(0)))
    level = assign_levels(p, u, a.cap).astype(np.int64)
    side = np.left_shift(np.int64(1), level)
    tx, ty = (p[:, 0] * side).astype(np.int64), (p[:, 1] * side).astype(np.int64)
    lx = np.round((p[:, 0] * side - tx) * 65535).astype(np.uint16)
    ly = np.round((p[:, 1] * side - ty) * 65535).astype(np.uint16)
    year = np.load(a.rows / "year.npy")[rows].clip(0, 4095).astype(np.uint16)
    year |= (np.load(a.rows / "smr.npy")[rows].astype(np.uint16) << 15)
    sub = np.load(a.rows / "subfield.npy")[rows].astype(np.uint16)
    key = (level.astype(np.int64) << 40) | (ty << 20) | tx
    o = np.lexsort((u, key))
    key, lx, ly, year, sub, ids = key[o], lx[o], ly[o], year[o], sub[o], work[rows][o]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    tiles = {}
    for s, e in zip(starts, ends):
        k = int(key[s]); z, y, x = k >> 40, (k >> 20) & 0xFFFFF, k & 0xFFFFF
        d = tmp / str(z) / str(x)
        d.mkdir(parents=True, exist_ok=True)
        np.stack([lx[s:e], ly[s:e], year[s:e], sub[s:e]], 1).astype("<u2").tofile(d / f"{y}.pts")
        ids[s:e].astype("<u8").tofile(d / f"{y}.ids")
        tiles[f"{z}/{x}/{y}"] = int(e - s)
    atlas = json.loads((a.atlas / "meta.json").read_text())
    levels = int(level.max()) + 1
    if sum(tiles.values()) != len(rows) or len(tiles) != len(starts):
        raise SystemExit("tile bookkeeping does not add up")
    per_level = [int(sum(n for t, n in tiles.items() if int(t.split("/")[0]) == z)) for z in range(levels)]
    doc = {"format": "pts: <u2 x,y (in tile, /65535), year (bit15 = readable on the site), subfield; ids: <u8 W-number",
           "scheme": "quadtree, at most `capacity` smallest-u points per tile; load all levels down to the zoom",
           "capacity": a.cap, "levels": levels, "papers": int(len(rows)), "per_level": per_level,
           "kept_without_position": int(keep.sum()) - len(rows), "shards_not_placed": missing_shards,
           "largest_tile": max(tiles.values()), "tiles": tiles, "atlas": atlas["id"],
           "subfields": [{"name": s["name"], "field": s["field"]} for s in atlas["subfields"]],
           "fields": [{"name": f["name"], "colour": f["colour"], "domain": f.get("domain", "")} for f in atlas["fields"]],
           "seconds": round(time.time() - t0)}
    (tmp / "index.json").write_text(json.dumps(doc))
    os.replace(tmp, a.out)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("tiles", "subfields", "fields")}), flush=True)


if __name__ == "__main__":
    main()
