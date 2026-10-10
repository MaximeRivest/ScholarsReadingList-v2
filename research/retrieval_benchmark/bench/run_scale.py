"""Scale test: every BEIR query searched against one ~1.2M-paper collection.

Collection = the four BEIR scientific corpora + a random sample of 1M OpenAlex works
(title + abstract), minus OpenAlex works whose normalised title matches any BEIR doc.
Doc ids are prefixed with their source ("scifact:123", "openalex:W..."). Each query is
scored against its own dataset's judgments; OpenAlex papers are unjudged, so any relevant
one counts as wrong. Quality drops here are therefore an upper bound on the true drop.

Steps: encode | exact | build | search   (run search under taskset, see run_beir).
"""
from __future__ import annotations

import argparse
import json
import os
import time

THREADS = int(os.environ.get("BENCH_THREADS", "4"))
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(var, str(THREADS))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from . import beir_data, bm25, encode as enc, engines, exact  # noqa: E402
from .common import DATA, DATASETS, normalize_title, write_json  # noqa: E402
from .metrics import evaluate  # noqa: E402
from .models import MODELS  # noqa: E402
from .run_beir import K, K_ENGINE, latency_stats  # noqa: E402

# FastPLAID is excluded at this scale: on 4 CPU cores it already takes ~0.35-1.2 s per
# query on 5k documents (probes/fastplaid_cpu.py), so it is not a CPU-serving candidate.
SCALE_ENGINES = ("warp",)

OA = "openalex1m"
NAME = "beir4+openalex1m"
RESULTS = DATA / "results" / NAME


def openalex_part():
    t = pq.read_table(DATA / "openalex" / "sample_1000000.parquet", columns=["id", "title", "text"])
    return t.column("id").to_pylist(), t.column("title").to_pylist(), t.column("text").to_pylist()


def collection():
    """Ordered parts: (name, prefixed ids, keep mask). Queries/qrels per dataset."""
    parts, beir_titles, per_ds = [], set(), {}
    for ds in DATASETS:
        docs, queries, qrels = beir_data.load(ds)
        beir_titles |= {normalize_title(d[1]) for d in docs if normalize_title(d[1])}
        parts.append((ds, [f"{ds}:{d[0]}" for d in docs], None))
        per_ds[ds] = (list(queries), list(queries.values()),
                      {q: {f"{ds}:{d}": g for d, g in j.items()} for q, j in qrels.items()})
    ids, titles, _ = openalex_part()
    keep = np.array([normalize_title(t) not in beir_titles for t in titles])
    parts.append((OA, [f"openalex:{i.rsplit('/', 1)[-1]}" for i in ids], keep))
    return parts, per_ds


def step_encode(models, device):
    ids, _, texts = openalex_part()
    dummy_q = ["placeholder"]
    for m in models:
        meta = enc.encode(m, OA, ids, texts, ["q"], dummy_q, device=device)
        print(f"{OA} {m}: {meta['docs_per_second_gpu']:.0f} docs/s", flush=True)


def _late_parts(m, parts):
    for name, ids, keep in parts:
        docs, doclens, *_ = enc.load_late(m, name)
        yield name, ids, keep, docs, doclens


def step_exact(models, device, binary=False, datasets=None):
    parts, per_ds = collection()
    n_docs = sum(len(i) if k is None else int(k.sum()) for _, i, k in parts)
    for ds, (qids, qtexts, qrels) in per_ds.items():
        if datasets is not None and ds not in datasets:
            continue
        if "bm25" in models:
            pass  # BM25 is run once over the whole collection below
        for m in models:
            if m == "bm25":
                continue
            merged_s, merged_i = None, None
            t = time.perf_counter()
            if MODELS[m].kind == "late":
                _, _, queries, *_ = enc.load_late(m, ds)
                for name, ids, keep, docs, doclens in _late_parts(m, parts):
                    extra = 0 if keep is None else int((~keep).sum())
                    ti, ts = exact.maxsim_topk(docs, doclens, queries, K + extra, device=device,
                                               binarize=binary)
                    merged_s, merged_i = _merge(merged_s, merged_i, ti, ts, ids, keep)
            else:
                _, queries, *_ = enc.load_dense(m, ds)
                for name, ids, keep in parts:
                    docs, *_ = enc.load_dense(m, name)
                    extra = 0 if keep is None else int((~keep).sum())
                    ti, ts = exact.dense_topk(docs, queries, K + extra, device=device,
                                              binarize=binary)
                    merged_s, merged_i = _merge(merged_s, merged_i, ti, ts, ids, keep)
            run = {q: dict(sorted(zip(merged_i[r], merged_s[r]), key=lambda x: -x[1])[:K])
                   for r, q in enumerate(qids)}
            rec = evaluate(run, qrels) | {"docs": n_docs, "exhaustive_gpu_seconds":
                                          time.perf_counter() - t}
            rec |= _openalex_share(run)
            _save(ds, f"{m}{'-1bit' if binary else ''}-exact", rec)
    if "bm25" in models and not binary:
        _bm25(parts, per_ds, n_docs)


def _merge(ms, mi, ti, ts, ids, keep):
    rows_i, rows_s = [], []
    for r in range(len(ti)):
        pairs = [(ids[int(d)], float(s)) for d, s in zip(ti[r], ts[r])
                 if keep is None or keep[int(d)]]
        if ms is not None:
            pairs += list(zip(mi[r], ms[r]))
        pairs.sort(key=lambda x: -x[1])
        pairs = pairs[:K]
        rows_i.append([p[0] for p in pairs])
        rows_s.append([p[1] for p in pairs])
    return rows_s, rows_i


def _openalex_share(run):
    top10 = [d for scores in run.values()
             for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:10]]
    return {"openalex_share_of_top10": sum(d.startswith("openalex:") for d in top10) / max(1, len(top10))}


def _bm25(parts, per_ds, n_docs):
    texts, ids = [], []
    for name, pids, keep in parts:
        if name == OA:
            t = openalex_part()[2]
        else:
            t = [d[2] for d in beir_data.load(name)[0]]
        for i, (pid, tx) in enumerate(zip(pids, t)):
            if keep is None or keep[i]:
                ids.append(pid)
                texts.append(tx)
    t = time.perf_counter()
    retr = bm25.build(texts)
    build_s = time.perf_counter() - t
    for ds, (qids, qtexts, qrels) in per_ds.items():
        ti, ts, lat = bm25.search(retr, qtexts, K)
        run = exact.to_run(ti, ts, ids, qids)
        _save(ds, "bm25", evaluate(run, qrels) | {"docs": n_docs, "build_seconds": build_s,
              "search_latency": latency_stats(lat), "search_threads": 1} | _openalex_share(run))


def _union(m, parts):
    """Concatenated vectors + lengths of the kept documents, and their prefixed ids."""
    vecs, lens, ids = [], [], []
    for name, pids, keep, docs, doclens in _late_parts(m, parts):
        if keep is None:
            vecs.append(docs)
            lens.append(doclens)
            ids += pids
            continue
        offsets = np.concatenate([[0], np.cumsum(doclens, dtype=np.int64)])
        sel = np.flatnonzero(keep)
        vecs.append(np.concatenate([docs[offsets[i]:offsets[i + 1]] for i in sel]))
        lens.append(doclens[sel])
        ids += [pids[i] for i in sel]
    return np.concatenate(vecs), np.concatenate(lens), ids


def step_build(models, device):
    parts, _ = collection()
    for m in models:
        if m == "bm25" or MODELS[m].kind != "late":
            continue
        docs, doclens, ids = _union(m, parts)
        for e in SCALE_ENGINES:
            path = DATA / "index" / e / m / NAME
            info = engines.build(e, path, docs, doclens, device)
            write_json(path.parent / f"{NAME}.ids.json", ids)
            write_json(path.parent / f"{NAME}.build.json", info | {"docs": len(ids),
                       "vectors": int(doclens.sum())})
            print(f"built {e}/{m}: {info}", flush=True)
        del docs


def step_search(models):
    torch.set_num_threads(THREADS)
    _, per_ds = collection()
    for m in models:
        if m == "bm25" or MODELS[m].kind != "late":
            continue
        for e in SCALE_ENGINES:
            path = DATA / "index" / e / m / NAME
            ids = json.loads((path.parent / f"{NAME}.ids.json").read_text())
            build = json.loads((path.parent / f"{NAME}.build.json").read_text())
            s = engines.Searcher(e, path, THREADS)
            for ds, (qids, _, qrels) in per_ds.items():
                _, _, queries, qlens, *_ = enc.load_late(m, ds)
                for i in range(min(10, len(qids))):
                    s.search(queries[i, : qlens[i]], K_ENGINE)
                run, lat = {}, []
                for i, q in enumerate(qids):
                    t = time.perf_counter()
                    hits = s.search(queries[i, : qlens[i]], K_ENGINE)
                    lat.append(time.perf_counter() - t)
                    run[q] = {ids[int(d)]: float(sc) for d, sc in hits}
                _save(ds, f"{m}-{e}", evaluate(run, qrels) | build | _openalex_share(run) | {
                      "search_latency": latency_stats(lat), "search_threads": THREADS,
                      "top_k": K_ENGINE})


def _save(ds, method, rec):
    rec |= {"dataset": ds, "method": method, "collection": NAME,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(RESULTS / ds / f"{method}.json", rec)
    lat = rec.get("search_latency", {})
    print(f"{ds:10s} {method:22s} ndcg@10={rec['ndcg@10']:.4f} r@100={rec['recall@100']:.4f} "
          f"oa-top10={rec.get('openalex_share_of_top10', 0):.2f} "
          f"p50={lat.get('p50_ms', 0):.1f}ms p95={lat.get('p95_ms', 0):.1f}ms", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["encode", "exact", "build", "search"])
    p.add_argument("--models", nargs="+", default=["bm25", "lateon", "mxbai32m", "denseon"])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--binary", action="store_true", help="exact step: 1-bit documents")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, help="exact step: evaluate selected query sets against the unchanged full collection")
    a = p.parse_args()
    models = [m for m in a.models if m == "bm25" or m in MODELS]
    {"encode": lambda: step_encode([m for m in models if m != "bm25"], a.device),
     "exact": lambda: step_exact(models, a.device, a.binary, a.datasets),
     "build": lambda: step_build(models, a.device),
     "search": lambda: step_search(models)}[a.step]()


if __name__ == "__main__":
    main()
