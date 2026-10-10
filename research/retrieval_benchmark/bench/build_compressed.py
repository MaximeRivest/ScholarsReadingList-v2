"""Build the indexes for the compression test over the 1.2M-paper collection.

WARP: LateOn at pool factor 1/2/4 x residual bits 1/2/4  -> DATA/index2/warp-nb{b}/{model}/
Faiss (DenseOn, CPU): HNSW32 float32 (RAM-only) and IVF4096-SQ8 (mmap-able)
                                                          -> DATA/index2/faiss/denseon/
DATA/index2 has btrfs compression disabled (chattr +m) so disk reads measure the drive.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

import numpy as np

from . import encode as enc, engines
from .common import DATA, write_json
from .run_scale import NAME, _union, collection

ROOT = DATA / "index2"


def prepare_root():
    ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chattr", "+m", str(ROOT)], check=True)


def build_warp(models, nbits_list, device):
    parts, _ = collection()
    for m in models:
        docs, doclens, ids = _union(m, parts)
        for nb in nbits_list:
            path = ROOT / f"warp-nb{nb}" / m / NAME
            if (path.parent / f"{NAME}.build.json").exists():
                continue
            info = engines.build("warp", path, docs, doclens, device, nbits=nb)
            write_json(path.parent / f"{NAME}.ids.json", ids)
            write_json(path.parent / f"{NAME}.build.json", info | {
                "docs": len(ids), "vectors": int(doclens.sum()), "model": m})
            print(f"built warp nb{nb} {m}: {info}", flush=True)
        del docs


def dense_matrix():
    parts, _ = collection()
    vecs, ids = [], []
    for name, pids, keep in parts:
        d, *_ = enc.load_dense("denseon", name)
        sel = np.arange(len(pids)) if keep is None else np.flatnonzero(keep)
        vecs.append(d[sel].astype(np.float32))
        ids += [pids[i] for i in sel]
    return np.concatenate(vecs), ids


def build_faiss(threads):
    import faiss
    faiss.omp_set_num_threads(threads)
    X, ids = dense_matrix()
    out = ROOT / "faiss" / "denseon"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{NAME}.ids.json", ids)
    d = X.shape[1]
    specs = {
        "hnsw32": lambda: faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT),
        "ivf4096_sq8": lambda: faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatIP(d), d, 4096, faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT),
    }
    for name, make in specs.items():
        path = out / f"{NAME}.{name}.index"
        if path.exists():
            continue
        t = time.perf_counter()
        index = make()
        if name.startswith("hnsw"):
            index.hnsw.efConstruction = 200
        else:
            rng = np.random.default_rng(0)
            index.train(X[rng.choice(len(X), 262_144, replace=False)])
        index.add(X)
        faiss.write_index(index, str(path))
        info = {"build_seconds": time.perf_counter() - t, "index_bytes": os.path.getsize(path),
                "build_threads": threads, "docs": len(ids)}
        write_json(out / f"{NAME}.{name}.build.json", info)
        print(f"built faiss {name}: {info}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["warp", "faiss"])
    p.add_argument("--models", nargs="+", default=["lateon", "lateon_pool2", "lateon_pool4"])
    p.add_argument("--nbits", type=int, nargs="+", default=[4, 2, 1])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--threads", type=int, default=32)
    a = p.parse_args()
    prepare_root()
    if a.what == "warp":
        build_warp(a.models, a.nbits, a.device)
    else:
        build_faiss(a.threads)
