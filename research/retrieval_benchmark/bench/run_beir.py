"""Benchmark on BEIR scientific sets; documents are title + abstract.

Steps (each caches its output, so they can be rerun):
  encode   documents and queries -> vectors (GPU)
  exact    BM25 and exhaustive model scoring -> quality ceiling per model
  build    FastPLAID / WARP indexes
  search   query each index one request at a time on CPU -> quality + latency
  qenc     CPU query-encoding latency per model
Run `search` and `qenc` under `taskset -c 0-3` with BENCH_THREADS=4 to emulate a small server.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

THREADS = int(os.environ.get("BENCH_THREADS", "4"))
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(var, str(THREADS))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import beir_data, bm25, encode as enc, engines, exact  # noqa: E402
from .common import DATA, DATASETS, write_json  # noqa: E402
from .metrics import evaluate  # noqa: E402
from .models import MODELS  # noqa: E402

RESULTS = DATA / "results"
K = 1000          # exhaustive runs keep 1000 results (for recall@1000)
K_ENGINE = 100    # served search returns 100 results


def latency_stats(seconds: list[float]) -> dict:
    ms = sorted(s * 1000 for s in seconds)
    return {"n": len(ms), "p50_ms": statistics.median(ms),
            "p95_ms": ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))],
            "mean_ms": statistics.fmean(ms)}


def save(dataset: str, method: str, record: dict) -> None:
    record |= {"dataset": dataset, "method": method, "host": platform.node(),
               "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(RESULTS / dataset / f"{method}.json", record)
    shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()
             if k in ("ndcg@10", "recall@100", "recall@1000", "index_bytes")}
    lat = record.get("search_latency", {})
    print(f"{dataset:10s} {method:22s} {shown} p50={lat.get('p50_ms', 0):.1f}ms "
          f"p95={lat.get('p95_ms', 0):.1f}ms", flush=True)


def corpus(dataset):
    docs, queries, qrels = beir_data.load(dataset)
    return ([d[0] for d in docs], [d[2] for d in docs], list(queries), list(queries.values()),
            qrels)


def step_encode(datasets, models, device):
    for ds in datasets:
        doc_ids, doc_texts, qids, qtexts, _ = corpus(ds)
        for m in models:
            meta = enc.encode(m, ds, doc_ids, doc_texts, qids, qtexts, device=device)
            print(f"{ds} {m}: {meta['docs_per_second_gpu']:.0f} docs/s, truncated "
                  f"{meta['truncated_fraction']}", flush=True)


def step_exact(datasets, models, device, binary=False):
    for ds in datasets:
        doc_ids, doc_texts, qids, qtexts, qrels = corpus(ds)
        if "bm25" in models and not binary:
            t = time.perf_counter()
            retr = bm25.build(doc_texts)
            build_s = time.perf_counter() - t
            ids, scores, lat = bm25.search(retr, qtexts, K)
            run = exact.to_run(ids, scores, doc_ids, qids)
            save(ds, "bm25", evaluate(run, qrels) | {"build_seconds": build_s,
                 "search_latency": latency_stats(lat), "search_threads": 1})
        for m in models:
            if m == "bm25":
                continue
            if MODELS[m].kind == "late":
                docs, doclens, queries, _, d_ids, q_ids, meta = enc.load_late(m, ds)
                t = time.perf_counter()
                ids, scores = exact.maxsim_topk(docs, doclens, queries, K, device=device,
                                                binarize=binary)
            else:
                docs, queries, d_ids, q_ids, meta = enc.load_dense(m, ds)
                t = time.perf_counter()
                ids, scores = exact.dense_topk(docs, queries, K, device=device, binarize=binary)
            assert d_ids == doc_ids and q_ids == qids
            run = exact.to_run(ids, scores, doc_ids, qids)
            bits = 1 if binary else 16
            bytes_per_doc = meta.get("mean_vectors_per_doc", 1) * meta["dim"] * bits / 8
            save(ds, f"{m}{'-1bit' if binary else ''}-exact", evaluate(run, qrels) | {
                "exhaustive_gpu_seconds": time.perf_counter() - t,
                "vector_bytes": int(bytes_per_doc * len(doc_ids)), "bits_per_value": bits,
                "encode": meta})


def step_build(datasets, models, device):
    for ds in datasets:
        for m in models:
            if MODELS[m].kind != "late":
                continue
            docs, doclens, *_ = enc.load_late(m, ds)
            for e in engines.ENGINES:
                path = DATA / "index" / e / m / ds
                try:
                    info = engines.build(e, path, docs, doclens, device)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  {e}/{m}/{ds}: GPU out of memory, building on CPU", flush=True)
                    info = engines.build(e, path, docs, doclens, "cpu")
                write_json(path.parent / f"{ds}.build.json", info)
                print(f"built {e}/{m}/{ds}: {info}", flush=True)


def step_search(datasets, models):
    torch.set_num_threads(THREADS)
    for ds in datasets:
        doc_ids, _, qids, _, qrels = corpus(ds)
        for m in models:
            if MODELS[m].kind != "late":
                continue
            _, _, queries, qlens, _, q_ids, _ = enc.load_late(m, ds)
            assert q_ids == qids
            for e in engines.ENGINES:
                path = DATA / "index" / e / m / ds
                build = json.loads((path.parent / f"{ds}.build.json").read_text())
                s = engines.Searcher(e, path, THREADS)
                for i in range(min(10, len(qids))):           # warm-up
                    s.search(queries[i, : qlens[i]], K_ENGINE)
                run, lat = {}, []
                for i, qid in enumerate(qids):
                    t = time.perf_counter()
                    hits = s.search(queries[i, : qlens[i]], K_ENGINE)
                    lat.append(time.perf_counter() - t)
                    run[qid] = {doc_ids[int(d)]: float(sc) for d, sc in hits}
                save(ds, f"{m}-{e}", evaluate(run, qrels) | build | {
                    "search_latency": latency_stats(lat), "search_threads": THREADS,
                    "top_k": K_ENGINE})


def step_qenc(models, n=100):
    torch.set_num_threads(THREADS)
    _, queries, _ = beir_data.load("scifact")
    texts = list(queries.values())[:n]
    out = {}
    for m in models:
        if m == "bm25":
            continue
        model = enc.load_model(MODELS[m], "cpu")
        f = enc._encode_late if MODELS[m].kind == "late" else enc._encode_dense
        for t in texts[:10]:
            f(model, [t], True, 1)
        lat = []
        for t in texts:
            s = time.perf_counter()
            f(model, [t], True, 1)
            lat.append(time.perf_counter() - s)
        out[m] = latency_stats(lat) | {"threads": THREADS, "dtype": "float32", "runtime": "pytorch"}
        print(m, out[m], flush=True)
    write_json(RESULTS / "query_encoding.json", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["encode", "exact", "build", "search", "qenc"])
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--models", nargs="+", default=["bm25", *MODELS])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--binary", action="store_true", help="exact step: 1-bit documents")
    a = p.parse_args()
    models = [m for m in a.models if m == "bm25" or m in MODELS]
    if a.step == "encode":
        step_encode(a.datasets, [m for m in models if m != "bm25"], a.device)
    elif a.step == "exact":
        step_exact(a.datasets, models, a.device, a.binary)
    elif a.step == "build":
        step_build(a.datasets, models, a.device)
    elif a.step == "search":
        step_search(a.datasets, models)
    else:
        step_qenc(models)


if __name__ == "__main__":
    main()
