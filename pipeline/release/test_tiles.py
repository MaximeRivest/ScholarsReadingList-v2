"""cd search && uv run python -m pytest -q ../pipeline/release/test_tiles.py"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "citewalk" / "science"))
from sample import splitmix_u  # noqa: E402


def make(tmp: Path, n=20000, shard_rows=1000):
    rng = np.random.default_rng(0)
    rows, pos, atlas = tmp / "rows", tmp / "pos", tmp / "atlas"
    for d in (rows, pos, atlas):
        d.mkdir()
    work = rng.choice(10**9, n, replace=False).astype(np.uint64)
    xy = rng.random((n, 2)).astype(np.float32)
    keep = rng.random(n) < .9
    shards = []
    for k, a in enumerate(range(0, n, shard_rows)):
        d = f"year=2020/rg-000000/part-{k:04d}"
        shards.append({"dir": d, "rows": shard_rows, "first": a})
        if k != 3:                                      # one shard not placed yet
            (pos / d).mkdir(parents=True)
            np.save(pos / d / "xy.npy", xy[a:a + shard_rows])
    np.save(rows / "work.npy", work); np.save(rows / "keep.npy", keep)
    np.save(rows / "by_work.npy", np.argsort(work).astype(np.uint32))
    np.save(rows / "year.npy", rng.integers(1900, 2027, n).astype(np.int16))
    np.save(rows / "smr.npy", rng.random(n) < .1)
    np.save(rows / "subfield.npy", rng.integers(0, 250, n).astype(np.uint16))
    (rows / "shards.json").write_text(json.dumps(shards))
    (rows / "meta.json").write_text(json.dumps({"rows": n}))
    ref = np.array([0, 1, 2]) + 3 * shard_rows        # references inside the unplaced shard keep their coords
    np.savez(atlas / "reference.npz", work=work[ref], coords=xy[ref])
    (atlas / "meta.json").write_text(json.dumps({"id": "test", "subfields": [], "fields": []}))
    return rows, pos, atlas, work, xy, keep, ref


def test_assign_levels_capacity_and_order():
    sys.path.insert(0, str(HERE))
    import tiles
    rng = np.random.default_rng(1)
    p = np.r_[rng.random((3000, 2)) * .05, rng.random((2000, 2))].astype(np.float32)   # one dense corner
    u = rng.random(len(p))
    lv = tiles.assign_levels(p, u, cap=100, max_level=8)
    for z in range(int(lv.max()) + 1):
        side = 1 << z
        m = lv == z
        cell = (p[m, 1] * side).astype(int) * side + (p[m, 0] * side).astype(int)
        assert np.bincount(cell).max() <= 100 or z == 8
        if z:   # a point went deeper only because its parent tile was full of smaller-u points
            for i in np.flatnonzero(m)[:50]:
                ps = 1 << (z - 1)
                par = lv == z - 1
                same = par & ((p[:, 0] * ps).astype(int) == int(p[i, 0] * ps)) & ((p[:, 1] * ps).astype(int) == int(p[i, 1] * ps))
                assert same.sum() == 100 and u[same].max() < u[i]
    assert lv.max() > 3      # the dense corner went deep


def test_tiles_roundtrip_and_nesting(tmp_path):
    rows, pos, atlas, work, xy, keep, ref = make(tmp_path)
    out = tmp_path / "tiles"
    subprocess.run([sys.executable, str(HERE / "tiles.py"), "--rows", str(rows), "--positions", str(pos),
                    "--atlas", str(atlas), "--out", str(out), "--cap", "1"], check=True, capture_output=True)
    idx = json.loads((out / "index.json").read_text())
    placed = keep.copy(); placed[3000:4000] = False; placed[ref] = keep[ref]
    assert idx["papers"] == int(placed.sum())
    seen = {}
    for t, cnt in idx["tiles"].items():
        z, x, y = map(int, t.split("/"))
        p = np.fromfile(out / f"{t}.pts", "<u2").reshape(-1, 4)
        ids = np.fromfile(out / f"{t}.ids", "<u8")
        assert len(p) == len(ids) == cnt
        side = 2 ** z
        gx, gy = (x + p[:, 0] / 65535) / side, (y + p[:, 1] / 65535) / side
        u = splitmix_u(ids)
        assert np.all(np.diff(u) >= 0)                               # random order inside a tile
        assert cnt <= idx["capacity"]
        for w, a, b in zip(ids, gx, gy):
            seen[int(w)] = (a, b)
    assert len(seen) == idx["papers"]                                  # each paper exactly once
    assert idx["levels"] >= 9 and sum(idx["per_level"]) == idx["papers"]   # deep levels exercised
    row = {int(w): i for i, w in enumerate(work)}
    err = max(max(abs(a - xy[row[w], 0]), abs(b - xy[row[w], 1])) for w, (a, b) in seen.items())
    assert err < 1 / 65535 / 1 + 1e-6
    table = np.load(out / "xy.npy")
    assert np.isnan(table[3500, 0]) and np.allclose(table[ref], xy[ref])
