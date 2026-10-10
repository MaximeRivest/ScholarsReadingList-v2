"""BM25 keyword baseline (bm25s, Lucene variant, k1=1.5, b=0.75, English stemming)."""
from __future__ import annotations

import time

import bm25s
import numpy as np
import Stemmer

_STEMMER = Stemmer.Stemmer("english")


def _tok(texts):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, show_progress=False)


def build(texts: list[str]) -> bm25s.BM25:
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(_tok(texts), show_progress=False)
    return retriever


def search(retriever: bm25s.BM25, queries: list[str], k: int):
    """One query at a time, single thread, so the timing is per-request latency."""
    ids, scores, lat = [], [], []
    for q in queries:
        t = time.perf_counter()
        r, s = retriever.retrieve(_tok([q]), k=k, n_threads=1, show_progress=False)
        lat.append(time.perf_counter() - t)
        ids.append(r[0])
        scores.append(s[0])
    return np.asarray(ids), np.asarray(scores), lat
