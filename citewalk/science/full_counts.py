"""Exact counts of every publication on the map, per year, place and field: what the pilot's growth views use.

    cd pipeline/embeddings && .venv/bin/python ../../citewalk/science/full_counts.py [--release ID] [--atlas DIR]

The viewer draws a 1 % sample as dots (a browser cannot draw 125 million), but its territories, growth map
and bar race were also computed from those dots, so small fields and early years were mostly sampling noise.
This counts every publication of a data release instead: the release's rows (publications, one per distinct
text: `keep`), each paper's position under this atlas build (pipeline/release/positions.py, one file per
embedding shard), its publication year (1800-2026) and its OpenAlex field and subfield (atlas codes).

Writes /mnt/fast/science-atlas/counts/<atlas id>--<release id>/:
  terr.npy    [years, fields, 256*256]  papers per year, field and cell of the territory grid (layers.js TG)
  heat.u32    [years, 160*160]          papers per year and cell of the growth grid (layers.js HG), all fields
  sub.u32     [years, subfields]        papers per year and subfield (the bar race)
  meta.json   years, grids, totals per year, sources, what was left out
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

RELEASES = Path("/mnt/fast/scholarsreadinglist/release")
POSITIONS = RELEASES / "positions"
OUT = Path("/mnt/fast/science-atlas/counts")
TG, HG = 256, 160
Y0, Y1 = 1800, 2026


def log(t0, msg):
    print(f"{time.time() - t0:6.0f}s {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", help="release id (default: current)")
    ap.add_argument("--atlas", type=Path, default=Path("/mnt/fast/science-atlas/build/20261008T131030-tsne"))
    a = ap.parse_args()
    t0 = time.time()
    rel = json.loads(((RELEASES / "releases" / a.release) if a.release else (RELEASES / "current")).joinpath("manifest.json").read_text())
    rows = Path(rel["paths"]["rows"])
    atlas = json.loads((a.atlas / "meta.json").read_text())
    rmeta = json.loads((rows / "meta.json").read_text())
    if rmeta["fields"] != [f["name"] for f in atlas["fields"]] or rmeta["subfields"] != [s["name"] for s in atlas["subfields"]]:
        raise SystemExit("the release's field codes differ from this atlas build's")
    out = OUT / f"{atlas['id']}--{rel['id']}"
    if (out / "meta.json").exists():
        raise SystemExit(f"{out} is finished")
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    shards = json.loads((rows / "shards.json").read_text())
    N = rmeta["rows"]
    xy = np.full((N, 2), np.nan, np.float32)
    missing = 0
    for s in shards:
        f = POSITIONS / atlas["id"] / s["dir"] / "xy.npy"
        if not f.exists():
            missing += s["rows"]
            continue
        p = np.load(f)
        if len(p) != s["rows"]:
            raise SystemExit(f"{f}: {len(p)} positions for {s['rows']} rows")
        xy[s["first"]:s["first"] + s["rows"]] = p
    log(t0, f"{len(shards):,} shards, {N:,} rows; {missing:,} rows have no position under {atlas['id']}")

    keep = np.load(rows / "keep.npy")
    year = np.load(rows / "year.npy").astype(np.int64)
    field = np.load(rows / "field.npy").astype(np.int64)
    sub = np.load(rows / "subfield.npy").astype(np.int64)
    placed = ~np.isnan(xy[:, 0])
    inyear = (year >= Y0) & (year <= Y1)
    use = keep & placed & inyear
    left = {"publications": int(keep.sum()), "without_position": int((keep & ~placed).sum()),
            "outside_1800_2026": int((keep & placed & ~inyear).sum()), "counted": int(use.sum())}
    log(t0, json.dumps(left))
    idx = np.flatnonzero(use)
    x, y = xy[idx, 0], xy[idx, 1]
    yi, fi, si = year[idx] - Y0, field[idx], sub[idx]
    del xy, keep, year, field, sub
    NY, NF, NS = Y1 - Y0 + 1, len(atlas["fields"]), len(atlas["subfields"])

    def cell(g):
        gx = np.clip((x * g).astype(np.int64), 0, g - 1)   # layers.js drops points outside; positions are inside
        gy = np.clip((y * g).astype(np.int64), 0, g - 1)
        return gy * g + gx

    heat = np.bincount(yi * HG * HG + cell(HG), minlength=NY * HG * HG).astype(np.uint32)
    heat.tofile(tmp / "heat.u32")
    log(t0, "growth grid")
    subc = np.bincount(yi * NS + si, minlength=NY * NS).astype(np.uint32)
    subc.tofile(tmp / "sub.u32")
    # territory: per year (bounded memory), into one memory-mapped file
    terr = np.lib.format.open_memmap(tmp / "terr.npy", mode="w+", dtype=np.uint32, shape=(NY, NF, TG * TG))
    c = cell(TG)
    order = np.argsort(yi, kind="stable")
    bounds = np.searchsorted(yi[order], np.arange(NY + 1))
    for k in range(NY):
        j = order[bounds[k]:bounds[k + 1]]
        if len(j):
            terr[k] = np.bincount(fi[j] * TG * TG + c[j], minlength=NF * TG * TG).reshape(NF, TG * TG)
    terr.flush()
    del terr
    log(t0, "territory grid")
    totals = np.bincount(yi, minlength=NY)
    meta = {"atlas": atlas["id"], "release": rel["id"], "rows": str(rows), "positions": str(POSITIONS / atlas["id"]),
            "years": [Y0, Y1], "territory_grid": TG, "growth_grid": HG, "fields": NF, "subfields": NS,
            "papers_per_year": totals.tolist(), "papers": int(totals.sum()), "left_out": left,
            "what": "publications (one per distinct title + abstract) of the release, by publication year, at "
                    "their position under this atlas build",
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0)}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    os.replace(tmp, out)
    log(t0, f"done: {meta['papers']:,} papers -> {out}")


if __name__ == "__main__":
    main()
