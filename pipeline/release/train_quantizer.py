"""Train the semantic index's lists once, on a uniform sample of every year (not on the backfill).

    cd search && uv run python ../pipeline/release/train_quantizer.py

Training sample: the science atlas's display sample (citewalk/science/sample.py): a fixed 1 % of
the embedding plan's works in every publication year 1800-2026, restricted to the publication
types the release indexes (citewalk/science/scope.py). It is uniform over the whole corpus, so it
is the right training distribution, and all of it is already embedded (the atlas encoded the
years the backfill has not reached). The trained index is reused by every release; each release
only adds its rows (pipeline/release/build_index.py). Retrain only when the corpus has grown a
lot (see DATA.md).

Writes /mnt/fast/scholarsreadinglist/release/quantizer/<id>/{trained.faiss, manifest.json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "citewalk" / "science"))
import scope  # noqa: E402

ATLAS = Path("/mnt/fast/science-atlas")
OUT = Path("/mnt/fast/scholarsreadinglist/release/quantizer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", type=int, default=32768)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    t0 = time.time()
    work = np.load(ATLAS / "vectors" / "work.npy")
    X16 = np.load(ATLAS / "vectors" / "vectors.npy", mmap_mode="r")
    meta = pq.read_table(sorted((ATLAS / "sample").glob("year=*.parquet")), columns=["work", "u", "type"])
    meta = meta.take(np.argsort(meta["work"].to_numpy()))
    if not np.array_equal(meta["work"].to_numpy(), work):
        raise SystemExit("atlas sample and its vectors disagree")
    rate = json.loads((ATLAS / "sample" / "summary.json").read_text())["display_rate"]
    types = np.asarray(meta["type"].fill_null("").to_pylist(), dtype=object)
    rows = np.flatnonzero((meta["u"].to_numpy() < rate) & np.isin(types, list(scope.SCHOLARLY_TYPES)))
    train = np.ascontiguousarray(X16[rows], dtype=np.float32)
    per_list = len(train) / a.lists
    print(f"{len(train):,} training vectors ({per_list:.0f} per list) in {time.time() - t0:.0f}s", flush=True)
    if per_list < 39:
        raise SystemExit("fewer than 39 training vectors per list (faiss's minimum); use fewer lists")

    factory = f"IVF{a.lists}_HNSW32,RaBitQ"
    ident = f"ivf{a.lists}-rabitq-{hashlib.sha256(work[rows].tobytes()).hexdigest()[:10]}"
    out = OUT / ident
    if (out / "manifest.json").exists():
        print(f"already trained: {out}"); return
    out.mkdir(parents=True, exist_ok=True)
    index = faiss.index_factory(768, factory, faiss.METRIC_INNER_PRODUCT)
    faiss.extract_index_ivf(index).cp.niter = a.iterations
    t = time.time()
    index.train(train)
    secs = time.time() - t
    faiss.write_index(index, str(out / "trained.faiss.tmp"))
    os.replace(out / "trained.faiss.tmp", out / "trained.faiss")
    doc = {"id": ident, "factory": factory, "metric": "inner_product", "lists": a.lists,
           "kmeans_iterations": a.iterations, "train_rows": int(len(train)),
           "train_source": "science-atlas display sample (u < 1%, every year), scholarly types",
           "types": sorted(scope.SCHOLARLY_TYPES), "train_seconds": round(secs),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "manifest.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1), flush=True)


if __name__ == "__main__":
    main()
