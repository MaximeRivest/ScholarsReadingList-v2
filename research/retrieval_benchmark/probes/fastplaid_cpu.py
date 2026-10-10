"""FastPLAID 1.7 CPU latency on scifact: one query at a time vs all queries in one call.

Run: OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 taskset -c 0-3 uv run --no-project \
  --with fast-plaid==1.7.0.2110 --with torch==2.11.0 --with numpy --with pytrec-eval-terrier \
  python -m probes.fastplaid_cpu
"""
import json, os, shutil, sys, time
import numpy as np, torch
from bench import encode as enc
from bench.common import DATA
from bench.metrics import evaluate
from bench import beir_data
import importlib.metadata as m
from fast_plaid import search
print("fast-plaid", m.version("fast-plaid"))
docs, dl, Q, ql, dids, qids, _ = enc.load_late("lateon", "scifact")
path = DATA / "index/fastplaid17/lateon/scifact"
if not path.exists():
    off = np.concatenate([[0], np.cumsum(dl)]); flat = torch.from_numpy(np.array(docs))
    search.FastPlaid(index=str(path), device="cuda:1").create(
        documents_embeddings=[flat[off[i]:off[i+1]] for i in range(len(dl))], nbits=4, seed=42)
idx = search.FastPlaid(index=str(path), device="cpu")
_, _, qrels = beir_data.load("scifact")
for kw in ({}, {"n_full_scores": 1024}):
    run, lat = {}, []
    for i, q in enumerate(qids):
        x = torch.from_numpy(Q[i:i+1, :ql[i]]).float()
        t = time.perf_counter(); r = idx.search(queries_embeddings=x, top_k=100, show_progress=False, **kw)[0]
        lat.append(time.perf_counter() - t)
        run[q] = {dids[int(d)]: float(s) for d, s in r}
    lat.sort(); print(kw, "ndcg@10 %.4f" % evaluate(run, qrels)["ndcg@10"], "p50 %.1f ms p95 %.1f ms" % (lat[len(lat)//2]*1e3, lat[int(.95*(len(lat)-1))]*1e3), flush=True)
qmax = int(ql.max()); X = torch.zeros(len(qids), qmax, Q.shape[2])
for i in range(len(qids)): X[i, :ql[i]] = torch.from_numpy(Q[i, :ql[i]]).float()
t = time.perf_counter(); idx.search(queries_embeddings=X, top_k=100, show_progress=False)
dt = time.perf_counter() - t
print("all %d queries in one call: %.1f s -> %.1f queries/s, %.1f ms average each" % (len(qids), dt, len(qids)/dt, dt*1e3/len(qids)))
