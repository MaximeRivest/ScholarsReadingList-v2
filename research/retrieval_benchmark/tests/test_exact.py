import numpy as np
import pytest
import torch

from bench.exact import dense_topk, maxsim_topk

DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"


def naive(docs, doclens, queries, qlens):
    offsets = np.concatenate([[0], np.cumsum(doclens)])
    out = np.zeros((len(queries), len(doclens)))
    for qi, q in enumerate(queries):
        q = q[: qlens[qi]].astype(np.float64)
        for di in range(len(doclens)):
            d = docs[offsets[di]:offsets[di + 1]].astype(np.float64)
            out[qi, di] = (q @ d.T).max(axis=1).sum()
    return out


def test_maxsim_matches_naive_with_chunking_and_padding():
    rng = np.random.default_rng(0)
    doclens = rng.integers(1, 40, size=230).astype(np.int32)
    docs = rng.normal(size=(doclens.sum(), 16)).astype(np.float16)
    docs /= np.linalg.norm(docs.astype(np.float32), axis=1, keepdims=True).astype(np.float16)
    qlens = rng.integers(3, 9, size=7)
    queries = np.zeros((7, 8, 16), dtype=np.float16)
    for i, n in enumerate(qlens):
        v = rng.normal(size=(n, 16))
        queries[i, :n] = v / np.linalg.norm(v, axis=1, keepdims=True)
    ref = naive(docs, doclens, queries, qlens)
    ids, scores = maxsim_topk(docs, doclens, queries, k=20, device=DEVICE,
                              docs_per_chunk=37, budget_bytes=4096)
    for qi in range(7):
        expected = np.sort(ref[qi])[::-1][:20]
        assert scores[qi] == pytest.approx(expected, abs=2e-2)
        assert ref[qi, ids[qi]] == pytest.approx(scores[qi], abs=2e-2)


def test_dense_matches_numpy():
    rng = np.random.default_rng(1)
    docs = rng.normal(size=(1000, 8)).astype(np.float16)
    queries = rng.normal(size=(5, 8)).astype(np.float16)
    ids, _ = dense_topk(docs, queries, k=10, device=DEVICE, chunk=137)
    ref = (queries.astype(np.float32) @ docs.astype(np.float32).T)
    for r in range(5):
        assert set(ids[r]) == set(np.argsort(-ref[r])[:10])


def test_binarized_maxsim_matches_naive_sign():
    rng = np.random.default_rng(2)
    doclens = rng.integers(1, 30, size=120).astype(np.int32)
    docs = rng.normal(size=(doclens.sum(), 16)).astype(np.float16)
    queries = np.zeros((4, 6, 16), dtype=np.float16)
    queries[:, :5] = rng.normal(size=(4, 5, 16))
    ref = naive(np.where(docs >= 0, 1, -1).astype(np.float16), doclens, queries, [5] * 4)
    ids, scores = maxsim_topk(docs, doclens, queries, k=15, device=DEVICE,
                              docs_per_chunk=29, budget_bytes=4096, binarize=True)
    for qi in range(4):
        assert scores[qi] == pytest.approx(np.sort(ref[qi])[::-1][:15], abs=5e-2)


def test_binarized_dense():
    rng = np.random.default_rng(3)
    docs = rng.normal(size=(500, 8)).astype(np.float16)
    q = rng.normal(size=(3, 8)).astype(np.float16)
    ids, scores = dense_topk(docs, q, k=5, device=DEVICE, chunk=77, binarize=True)
    ref = q.astype(np.float32) @ np.where(docs >= 0, 1.0, -1.0).T
    for r in range(3):
        # With only 8 bits many documents are identical: top-k may choose any tied
        # document. Validate scores and the selected documents, not arbitrary tie IDs.
        assert scores[r] == pytest.approx(np.sort(ref[r])[::-1][:5], abs=1e-2)
        assert ref[r, ids[r]] == pytest.approx(scores[r], abs=1e-2)
