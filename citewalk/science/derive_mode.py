"""A new atlas build that keeps an existing build's geography (references and their fitted
coordinates) and switches its placement rule to "mode" (placement.mode_place).

    cd citewalk/science && ../../search/.venv/bin/python derive_mode.py SRC_BUILD

Writes /mnt/fast/science-atlas/build/<UTC time>-<method>-mode/: reference.npz with rule, k,
tau, h, jitter and every reference's spacing; meta.json with the new id and its placement;
points.f32 (the display sample) re-placed with the new rule (references keep their fitted
coordinates); the other files are hard links to the source build (they do not depend on the
rule). A new id because positions of an atlas never change silently (DATA.md).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

import placement

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--jitter", type=float, default=0.25)
    a = ap.parse_args()
    t0 = time.time()
    meta = json.loads((a.src / "meta.json").read_text())
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    new_id = f"{stamp}-{meta['method']}-mode"
    out = a.src.parent / new_id
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir()
    ref = dict(np.load(a.src / "reference.npz"))
    Y = ref["coords"].astype(np.float64)
    spacing = cKDTree(Y).query(Y, k=2)[0][:, 1].astype(np.float32)
    h = 2.0 * float(np.median(spacing))
    ref.update(k=np.int64(placement.MODE_K), tau=np.float64(placement.MODE_TAU), rule=np.str_("mode"),
               h=np.float64(h), jitter=np.float64(a.jitter), spacing=spacing)
    np.savez(tmp / "reference.npz", **ref)
    print(f"references {len(Y):,}: median spacing {np.median(spacing):.3e}, h {h:.3e}", flush=True)

    # the display sample, re-placed (references keep their fitted coordinates); exact search on the CPU
    import faiss
    faiss.omp_set_num_threads(int(os.environ.get("THREADS", "32")))
    index = faiss.IndexFlatIP(ref["vectors"].shape[1])
    index.add(np.ascontiguousarray(ref["vectors"], dtype=np.float32))
    rule = placement.Rule(np.load(tmp / "reference.npz"))
    work = pq.read_table(a.src / "info.parquet", columns=["work"])["work"].to_numpy()
    X = np.load(a.src / "vectors.npy", mmap_mode="r")
    pts = np.fromfile(a.src / "points.f32", np.float32).reshape(-1, 2).copy()
    is_ref = np.isin(work, ref["work"])
    todo = np.flatnonzero(~is_ref)
    for s in range(0, len(todo), 200_000):
        i = todo[s:s + 200_000]
        q = np.ascontiguousarray(X[i], dtype=np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        sims, nn = index.search(q, rule.k)
        pts[i] = rule(sims, nn, work[i])
        print(f"display sample {min(s + 200_000, len(todo)):,}/{len(todo):,}", flush=True)
    pts.tofile(tmp / "points.f32")

    for f in a.src.iterdir():
        if f.name not in ("reference.npz", "points.f32", "meta.json") and f.is_file() and not f.name.endswith(".faiss"):
            os.link(f, tmp / f.name)
    meta.update(id=new_id, derived_from=meta["id"], built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                placement={"rule": "mode", "k": placement.MODE_K, "tau": placement.MODE_TAU, "h": h,
                           "jitter": a.jitter, "iters": placement.MODE_ITERS})
    (tmp / "meta.json").write_text(json.dumps(meta))
    os.replace(tmp, out)
    print(f"{out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
