"""Exhaustive (exact) scoring: the quality ceiling an index can at best match."""
from __future__ import annotations

import os

import numpy as np
import torch

NEG = -1e4  # padding bias; fp16-safe and far below any cosine similarity


def maxsim_topk(docs: np.ndarray, doclens: np.ndarray, queries: np.ndarray, k: int,
                device: str = "cuda:1", docs_per_chunk: int = 4096,
                budget_bytes: int = 768 * 2**20, binarize: bool = False
                ) -> tuple[np.ndarray, np.ndarray]:
    """Exact ColBERT MaxSim for every (query, document) pair; returns top-k ids, scores.

    docs: (total_vectors, dim) concatenation of every document's token vectors.
    queries: (n_queries, q_max, dim), zero-padded (a zero query row adds exactly 0).
    Documents are processed shortest-first in padded chunks; padded document positions
    get a large negative bias so they can never be the maximum.
    binarize: score against 1-bit documents (each value replaced by its sign, +1/-1)
    with full-precision queries ("asymmetric" binary quantization).
    """
    docs_per_chunk = min(docs_per_chunk, int(os.environ.get("BENCH_MAXSIM_DOC_CHUNK", "4096")))
    budget_bytes = min(budget_bytes, int(os.environ.get("BENCH_MAXSIM_BUDGET_MIB", "768")) * 2**20)
    offsets = np.concatenate([[0], np.cumsum(doclens, dtype=np.int64)])
    order = np.argsort(doclens, kind="stable")
    nq, qmax, dim = queries.shape
    q = torch.from_numpy(np.ascontiguousarray(queries)).to(device).reshape(nq * qmax, dim)
    best_s = torch.full((nq, 0), float("-inf"), device=device)
    best_i = torch.zeros((nq, 0), dtype=torch.long, device=device)
    for start in range(0, len(order), docs_per_chunk):
        ids = order[start:start + docs_per_chunk]
        lmax = int(doclens[ids].max())
        block = np.zeros((len(ids), lmax, dim), dtype=np.float16)
        mask = np.full((len(ids), lmax), NEG, dtype=np.float16)
        for row, d in enumerate(ids):
            n = doclens[d]
            block[row, :n] = docs[offsets[d]:offsets[d] + n]
            mask[row, :n] = 0
        d_t = torch.from_numpy(block).to(device).reshape(-1, dim)
        if binarize:
            d_t = _sign(d_t)
        m_t = torch.from_numpy(mask).to(device)
        qc = max(1, budget_bytes // (2 * qmax * len(ids) * lmax))
        chunk_scores = []
        for qs in range(0, nq, qc):
            sim = (q[qs * qmax:(qs + qc) * qmax] @ d_t.T).view(-1, len(ids), lmax)
            sim = (sim + m_t).amax(dim=2).float()                       # best doc token per query token
            chunk_scores.append(sim.view(-1, qmax, len(ids)).sum(dim=1))  # sum over query tokens
        s = torch.cat(chunk_scores)
        idx = torch.from_numpy(ids).to(device).expand(nq, -1)
        all_s, all_i = torch.cat([best_s, s], 1), torch.cat([best_i, idx], 1)
        top = all_s.topk(min(k, all_s.shape[1]), dim=1)
        best_s, best_i = top.values, all_i.gather(1, top.indices)
    return best_i.cpu().numpy(), best_s.cpu().numpy()


def _sign(x: torch.Tensor) -> torch.Tensor:
    """1-bit quantisation: >= 0 -> +1, < 0 -> -1 (a stored bit; zero maps to +1)."""
    return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)


def dense_topk(docs: np.ndarray, queries: np.ndarray, k: int, device: str = "cuda:1",
               chunk: int = 500_000, binarize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product search (embeddings are L2-normalised, so this is cosine).
    binarize: 1-bit documents, full-precision queries."""
    # Bound both dimensions. In particular, never materialize an int64 document-id
    # matrix for every query/document pair: keep only each tile's local top-k.
    chunk = min(chunk, 32768)
    all_ids, all_scores = [], []
    for qs in range(0, len(queries), 64):
        q = torch.from_numpy(np.ascontiguousarray(queries[qs:qs + 64])).to(device)
        best_s = torch.empty((len(q), 0), device=device)
        best_i = torch.empty((len(q), 0), dtype=torch.long, device=device)
        for start in range(0, len(docs), chunk):
            d = torch.from_numpy(np.ascontiguousarray(docs[start:start + chunk])).to(device)
            if binarize:
                d = _sign(d)
            local = (q @ d.T).float().topk(min(k, len(d)), dim=1)
            merged_s = torch.cat([best_s, local.values], dim=1)
            merged_i = torch.cat([best_i, local.indices + start], dim=1)
            top = merged_s.topk(min(k, merged_s.shape[1]), dim=1)
            best_s, best_i = top.values, merged_i.gather(1, top.indices)
        all_ids.append(best_i.cpu().numpy())
        all_scores.append(best_s.cpu().numpy())
    return np.concatenate(all_ids), np.concatenate(all_scores)


def to_run(top_ids: np.ndarray, top_scores: np.ndarray, doc_ids: list[str],
           query_ids: list[str]) -> dict[str, dict[str, float]]:
    return {qid: {doc_ids[int(i)]: float(s) for i, s in zip(top_ids[r], top_scores[r])}
            for r, qid in enumerate(query_ids)}
