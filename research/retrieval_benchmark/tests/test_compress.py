import numpy as np
import torch

from bench.compress import pool_one


def test_matches_pylate_pooling():
    from pylate.models.colbert import ColBERT
    rng = np.random.default_rng(0)
    for n, factor in ((50, 2), (181, 4), (7, 4), (2, 2), (1, 4)):
        x = rng.normal(size=(n, 32)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        ref = ColBERT.pool_embeddings_hierarchical(None, [torch.from_numpy(x)], factor, 1)[0]
        ours = pool_one(x, factor)
        assert ours.shape == tuple(ref.shape)
        np.testing.assert_allclose(ours, ref.numpy(), atol=1e-5)


def test_duplicate_tokens_do_not_crash():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(1, 64)).astype(np.float16)
    v = (v / np.linalg.norm(v.astype(np.float32))).astype(np.float16)
    doc = np.repeat(v, 40, axis=0)
    out = pool_one(doc, 4)
    assert len(out) == 1 + 39 // 4 or len(out) <= 1 + 39 // 4
