"""Walk-on-atlas positions: exact for dots and references, frozen placement otherwise, never guessed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_overlay import ScienceAtlas, ScienceAtlases  # noqa: E402
import placement  # noqa: E402  (science/placement.py, put on the path by atlas_overlay)

D = 16


def unit(rng, n):
    v = rng.normal(size=(n, D)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeEmb:
    plan_id = 'plan-A'
    def __init__(self, vecs): self.vecs = vecs
    def vectors(self, numbers): return [self.vecs.get(w) for w in numbers]


def make_build(root: Path, rng):
    ref_work = np.arange(1000, 1040, dtype=np.uint64)
    ref_xy = rng.random((40, 2)).astype(np.float32)
    ref_X = unit(rng, 40)
    dot_work = np.array([5, 9, 1001], np.uint64)          # 1001 is both a dot and a reference
    dot_xy = np.array([[.1, .2], [.3, .4], [.55, .66]], np.float32)
    b = root / 'build-1'
    b.mkdir()
    pl.DataFrame({'work': dot_work}).write_parquet(b / 'info.parquet')
    dot_xy.tofile(b / 'points.f32')
    np.savez(b / 'reference.npz', work=ref_work, coords=ref_xy, vectors=ref_X.astype(np.float16),
             k=placement.K, tau=placement.TAU, method='tsne')
    (b / 'meta.json').write_text(json.dumps({'id': 'build-1', 'method': 'tsne', 'embedding_plan': 'plan-A'}))
    s = root / 'vectors'
    s.mkdir()
    sample_work = np.array([7, 2000], np.uint64)
    sample_vec = unit(rng, 2).astype(np.float16)
    np.save(s / 'work.npy', sample_work)
    np.save(s / 'vectors.npy', sample_vec)
    return b, s, ref_X, ref_xy, dict(zip(sample_work.tolist(), sample_vec))


def node(i, *records):
    return {'id': f'W{i}', 'records': [f'W{r}' for r in (i,) + records]}


class Overlay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.rng = np.random.default_rng(3)
        self.b, self.s, self.ref_X, self.ref_xy, self.sample = make_build(Path(self.tmp.name), self.rng)
        self.atlas = ScienceAtlas(self.b, self.s)

    def placed(self, v):
        X = np.asarray(self.ref_X.astype(np.float16), np.float32)        # what the build stores
        v = np.asarray(v, np.float64); v = (v / np.linalg.norm(v)).astype(np.float32)
        return placement.place_numpy(X, self.ref_xy, v[None])[0]

    def test_sources_in_order_and_exact(self):
        v300 = unit(self.rng, 1)[0]
        nodes = {'W5': node(5), 'W1001': node(1001), 'W1003': node(1003), 'W300': node(300),
                 'W7': node(7), 'W400': node(400)}
        out = self.atlas.locate(nodes, FakeEmb({300: v300}))
        self.assertEqual((nodes['W5']['ax'], nodes['W5']['ay'], nodes['W5']['place']), (np.float32(.1).item(), np.float32(.2).item(), 'dot'))
        self.assertEqual(nodes['W1001']['place'], 'dot')                 # a dot wins over a reference
        self.assertEqual(nodes['W1003']['place'], 'reference')
        self.assertEqual([nodes['W1003']['ax'], nodes['W1003']['ay']], self.ref_xy[3].tolist())
        np.testing.assert_allclose([nodes['W300']['ax'], nodes['W300']['ay']], self.placed(v300), atol=1e-6)
        np.testing.assert_allclose([nodes['W7']['ax'], nodes['W7']['ay']], self.placed(self.sample[7]), atol=1e-6)
        self.assertIsNone(nodes['W400']['ax']); self.assertIsNone(nodes['W400']['place'])
        self.assertEqual(out['unplaced'], ['W400'])
        self.assertEqual(out['by'], {'dot': 2, 'reference': 1, 'placed': 2})

    def test_position_does_not_depend_on_other_papers(self):
        vs = {300 + i: unit(self.rng, 1)[0] for i in range(6)}
        a = {f'W{w}': node(w) for w in vs}
        b = {'W303': node(303)}
        self.atlas.locate(a, FakeEmb(vs)); self.atlas.locate(b, FakeEmb(vs))
        self.assertEqual((a['W303']['ax'], a['W303']['ay']), (b['W303']['ax'], b['W303']['ay']))

    def test_other_records_of_the_publication_are_used(self):
        nodes = {'W400': node(400, 9)}                                      # W9 is a dot
        self.atlas.locate(nodes, FakeEmb({}))
        self.assertEqual((nodes['W400']['place'], nodes['W400']['place_record']), ('dot', 'W9'))

    def test_different_embedding_plan_is_never_mixed_in(self):
        emb = FakeEmb({300: unit(self.rng, 1)[0]}); emb.plan_id = 'plan-B'
        nodes = {'W300': node(300), 'W5': node(5)}
        out = self.atlas.locate(nodes, emb)
        self.assertIsNone(nodes['W300']['ax'])
        self.assertEqual(nodes['W5']['place'], 'dot')
        self.assertFalse(out['plan_matches'])

    def test_bad_vectors_are_not_placed(self):
        nodes = {'W300': node(300), 'W301': node(301)}
        self.atlas.locate(nodes, FakeEmb({300: np.zeros(D), 301: np.full(D, np.nan)}))
        self.assertIsNone(nodes['W300']['ax']); self.assertIsNone(nodes['W301']['ax'])

    def test_build_ids_are_validated(self):
        reg = ScienceAtlases(Path(self.tmp.name), self.s)
        self.assertEqual(reg.get('build-1').id, 'build-1')
        for bad in ('../x', '', None, 'missing'):
            with self.assertRaises(ValueError):
                reg.get(bad)


if __name__ == '__main__':
    unittest.main()
