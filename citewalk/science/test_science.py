"""cd citewalk/science && uv run python -m unittest test_science -v"""
import unittest

import numpy as np

import placement
from sample import splitmix_u, rates


def unit(rng, n, d=32):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class Sampling(unittest.TestCase):
    def test_hash_is_fixed_uniform_and_order_free(self):
        ids = np.arange(1, 200_001, dtype=np.uint64)
        u = splitmix_u(ids)
        self.assertTrue(np.all((u >= 0) & (u < 1)))
        self.assertAlmostEqual(float((u < .01).mean()), .01, delta=.002)
        np.testing.assert_array_equal(splitmix_u(ids[::-1])[::-1], u)
        # pinned values: changing the hash would silently change the sample
        np.testing.assert_allclose(splitmix_u(np.array([1, 4210000000], np.uint64)), [0.5665615751722809, 0.2437196721069561], rtol=0, atol=1e-15)

    def test_display_rate_everywhere_and_balance_only_raises_old_years(self):
        counts = {1900: 30_000, 1975: 400_000, 2020: 7_000_000}
        r = rates(counts)
        at = lambda y: r[y - 1800]
        self.assertAlmostEqual(at(1900), .1)          # 3000 / 30000
        self.assertAlmostEqual(at(1975), .01)         # 1% already exceeds 3000
        self.assertAlmostEqual(at(2020), .01)         # outside balance years
        self.assertTrue(np.all(r >= .01))             # display sample = u < 1% in every year


class Placement(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.Xr, self.Yr = unit(rng, 500), rng.normal(size=(500, 2)).astype(np.float32)
        self.Q = unit(rng, 50)

    def test_position_does_not_depend_on_other_papers(self):
        alone = placement.place_numpy(self.Xr, self.Yr, self.Q[:1])
        batch = placement.place_numpy(self.Xr, self.Yr, self.Q)
        np.testing.assert_allclose(alone[0], batch[0], atol=1e-6)

    def test_numpy_and_faiss_agree(self):
        a = placement.place_numpy(self.Xr, self.Yr, self.Q)
        b = placement.place_faiss(self.Xr, self.Yr, self.Q, log=lambda m: None)
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_reference_excluding_itself_is_not_placed_on_itself(self):
        rows = np.arange(10)
        placed = placement.place_faiss(self.Xr, self.Yr, self.Xr[rows], exclude=rows, log=lambda m: None)
        own = placement.place_faiss(self.Xr, self.Yr, self.Xr[rows], log=lambda m: None)
        np.testing.assert_allclose(own, self.Yr[rows], atol=1e-3)   # self dominates the softmax
        self.assertTrue(np.all(np.abs(placed - self.Yr[rows]).sum(1) > 1e-3))

    def test_position_inside_neighbour_hull(self):
        p = placement.place_numpy(self.Xr, self.Yr, self.Q)
        self.assertTrue(np.all(p.min(0) >= self.Yr.min(0)) and np.all(p.max(0) <= self.Yr.max(0)))


if __name__ == '__main__':
    unittest.main()
