"""DenseOn on CPU over the 1.2M-paper collection: exact (flat) vs HNSW, one query at a time.

Build uses many cores; search is pinned to BENCH_THREADS cores (run under taskset).
Run: taskset -c 8-11 uv run python -m probes.dense_cpu
"""
import os
import time

import faiss
import numpy as np

from bench import encode as enc
from bench.common import DATA, write_json
from bench.metrics import evaluate
from bench.run_beir import latency_stats
from bench.run_scale import collection

THREADS = int(os.environ.get("BENCH_THREADS", "4"))
parts, per_ds = collection()
vecs, ids = [], []
for name, pids, keep in parts:
    d, *_ = enc.load_dense("denseon", name)
    sel = np.arange(len(pids)) if keep is None else np.flatnonzero(keep)
    vecs.append(d[sel].astype(np.float32))
    ids += [pids[i] for i in sel]
X = np.concatenate(vecs)
print("docs", X.shape, flush=True)

def run(index, label, extra):
    faiss.omp_set_num_threads(THREADS)
    out = {}
    for ds, (qids, _, qrels) in per_ds.items():
        _, Q, *_ = enc.load_dense("denseon", ds)
        Q = Q.astype(np.float32)
        for i in range(min(10, len(Q))):
            index.search(Q[i:i + 1], 100)
        run_, lat = {}, []
        for i, q in enumerate(qids):
            t = time.perf_counter()
            s, I = index.search(Q[i:i + 1], 100)
            lat.append(time.perf_counter() - t)
            run_[q] = {ids[j]: float(v) for j, v in zip(I[0], s[0]) if j >= 0}
        r = evaluate(run_, qrels) | {"search_latency": latency_stats(lat), "threads": THREADS} | extra
        out[ds] = r
        print(f"{label:22s} {ds:10s} ndcg@10={r['ndcg@10']:.4f} p50={r['search_latency']['p50_ms']:.2f}ms "
              f"p95={r['search_latency']['p95_ms']:.2f}ms", flush=True)
    return out

results = {}
flat = faiss.IndexFlatIP(X.shape[1]); flat.add(X)
results["flat"] = run(flat, "denseon-flat", {"index_bytes": X.nbytes})
faiss.omp_set_num_threads(32)
t = time.perf_counter()
hnsw = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 200
hnsw.add(X)
build = time.perf_counter() - t
print(f"hnsw build {build:.0f}s (32 threads)", flush=True)
for ef in (64, 128, 256):
    hnsw.hnsw.efSearch = ef
    results[f"hnsw_ef{ef}"] = run(hnsw, f"denseon-hnsw-ef{ef}", {"build_seconds": build, "M": 32, "efSearch": ef})
write_json(DATA / "results" / "beir4+openalex1m" / "denseon_cpu.json", results)
