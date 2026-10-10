"""Token pooling of cached late-interaction vectors (no re-encoding).

Same algorithm as PyLate's ColBERT.pool_embeddings_hierarchical (Clavié et al. 2024):
keep the first token, Ward-cluster the rest on cosine distance into n // factor
clusters, replace each cluster by its mean. Computed in float32, stored as float16.
Output layout matches bench.encode (emb/<model>_pool<factor>/<corpus>/).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from multiprocessing import Pool

import numpy as np
from scipy.cluster import hierarchy

from .common import write_json
from .encode import cache_dir

CHUNK = 4000


def pool_one(doc: np.ndarray, factor: int, protected: int = 1) -> np.ndarray:
    doc = doc.astype(np.float32)
    keep, rest = doc[:protected], doc[protected:]
    n = len(rest)
    k = max(n // factor, 1)
    if k >= n:
        return doc
    # Rounding makes 1 - cos slightly negative for (near-)duplicate tokens, which scipy
    # rejects; PyLate would raise here. Clamp to the true minimum distance, 0.
    dist = np.maximum(1.0 - rest @ rest.T, 0.0)
    condensed = dist[np.triu_indices(n, k=1)]
    labels = hierarchy.fcluster(hierarchy.linkage(condensed, method="ward"), t=k,
                                criterion="maxclust") - 1
    sums = np.zeros((k, doc.shape[1]), dtype=np.float32)
    np.add.at(sums, labels, rest)
    counts = np.bincount(labels, minlength=k)
    mask = counts > 0
    return np.concatenate([keep, sums[mask] / counts[mask, None]])


_SRC = {}


def _work(args):
    start, stop, factor = args
    docs, offsets = _SRC["docs"], _SRC["offsets"]
    out = [pool_one(np.asarray(docs[offsets[i]:offsets[i + 1]]), factor).astype(np.float16)
           for i in range(start, stop)]
    return out


def _init(path, dim, doclens):
    _SRC["docs"] = np.memmap(path, dtype=np.float16, mode="r").reshape(-1, dim)
    _SRC["offsets"] = np.concatenate([[0], np.cumsum(doclens, dtype=np.int64)])


def pool_corpus(model: str, corpus: str, factor: int, workers: int = 32) -> dict:
    src, dst = cache_dir(model, corpus), cache_dir(f"{model}_pool{factor}", corpus)
    if (dst / "meta.json").exists():
        return json.loads((dst / "meta.json").read_text())
    meta = json.loads((src / "meta.json").read_text())
    doclens = np.load(src / "doclens.npy")
    dst.mkdir(parents=True, exist_ok=True)
    tasks = [(s, min(s + CHUNK, len(doclens)), factor) for s in range(0, len(doclens), CHUNK)]
    lens = []
    with open(dst / "docs.f16.part", "wb") as fh, Pool(
            workers, initializer=_init, initargs=(src / "docs.f16", meta["dim"], doclens)) as pool:
        for chunk in pool.imap(_work, tasks):
            for e in chunk:
                fh.write(e.tobytes())
                lens.append(len(e))
    (dst / "docs.f16.part").replace(dst / "docs.f16")
    np.save(dst / "doclens.npy", np.asarray(lens, dtype=np.int32))
    for f in ("queries.npy", "qlens.npy", "doc_ids.json", "query_ids.json"):
        shutil.copy(src / f, dst / f)
    meta = meta | {"pool_factor": factor, "pooled_from": model, "n_vectors": int(sum(lens)),
                   "mean_vectors_per_doc": float(np.mean(lens))}
    write_json(dst / "meta.json", meta)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lateon")
    p.add_argument("--factors", type=int, nargs="+", default=[2, 4])
    p.add_argument("--corpora", nargs="+",
                   default=["scifact", "nfcorpus", "scidocs", "trec-covid", "openalex1m"])
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    for f in a.factors:
        for c in a.corpora:
            m = pool_corpus(a.model, c, f, a.workers)
            print(f"{a.model} pool{f} {c}: {m['mean_vectors_per_doc']:.1f} vectors/doc", flush=True)
