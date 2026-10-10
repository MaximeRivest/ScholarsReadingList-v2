"""Check Graph.analyze against a brute-force computation on random small graphs.

    uv run --with-requirements citewalk/requirements.txt python -m unittest citewalk/test_citewalk.py
"""
import json
from math import sqrt
from pathlib import Path
import random
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import Graph  # noqa: E402


def write_index(d: Path, n, edges, cites, titles=None, nrefs=None):
    np.save(d / 'wn.npy', np.arange(100, 100 + n, dtype=np.int64))  # W100.. ; node id = position
    np.save(d / 'year.npy', np.full(n, 2020, np.int16))
    np.save(d / 'cites.npy', np.asarray(cites, np.int32))
    np.save(d / 'nrefs.npy', np.asarray(nrefs if nrefs is not None else [-1] * n, np.int32))
    np.save(d / 'kind.npy', np.zeros(n, np.uint8))
    np.save(d / 'complete.npy', np.ones(n, bool))
    enc = [t.encode() for t in (titles or [f'paper {i}' for i in range(n)])]
    np.save(d / 'title_row.npy', np.arange(n, dtype=np.uint32))
    np.save(d / 'title_off.npy', np.r_[0, np.cumsum([len(t) for t in enc])].astype(np.uint64))
    (d / 'titles.bin').write_bytes(b''.join(enc))
    np.save(d / 'doi_hash.npy', np.zeros(0, np.uint64)); np.save(d / 'doi_node.npy', np.zeros(0, np.uint32))
    for name, a, b in (('fwd', 0, 1), ('bwd', 1, 0)):
        lists = [[] for _ in range(n)]
        for e in edges:
            lists[e[a]].append(e[b])
        np.save(d / f'{name}_off.npy', np.r_[0, np.cumsum([len(l) for l in lists])].astype(np.uint64))
        np.save(d / f'{name}.npy', np.asarray([v for l in lists for v in l], np.uint32))
    (d / 'meta.json').write_text(json.dumps(dict(nodes=n, edges=len(edges), types=['article'])))


def norm(t):
    return re.sub(r'[\W_]+', ' ', t.casefold()).strip()


def brute(n, edges, cites, titles, x, k, normalize, nrefs, max_refs, max_cites):
    """Independent restatement of the rules. Returns (similar publication keys in order, answer keys)."""
    refs = {i: set() for i in range(n)}
    for s, t in edges:
        refs[s].add(t)
    key = lambda i: norm(titles[i]) if len(norm(titles[i])) >= 20 else ('#', i)
    count = lambda i: nrefs[i] if nrefs[i] >= 0 else len(refs[i])
    rec_out = lambda i: bool((max_cites and cites[i] > max_cites) or (max_refs and count(i) > max_refs))
    grp_out = lambda m: bool((max_cites and sum(max(cites[i], 0) for i in m) > max_cites)
                             or (max_refs and max(count(i) for i in m) > max_refs))
    R = refs[x]
    groups = {}
    for r in R:
        groups.setdefault(key(r), set()).add(r)
    use = {K for K, m in groups.items() if not grp_out(m)}
    if not use:
        return [], []
    rows = []
    for y in range(n):
        if y == x:
            continue
        sh = len({key(r) for r in refs[y] & R if key(r) in use})
        if sh:
            cos = sh / sqrt(max(len(refs[y]), 1) * len(use))
            rows.append(((-(cos if normalize else sh), -cos, y), y))
    tx = norm(titles[x])
    chosen, pubs, dups = [], {}, []
    for _, y in sorted(rows):
        if len(chosen) == k:
            break
        if rec_out(y):
            continue
        ty = norm(titles[y])
        if len(tx) >= 20 and (tx in ty or (len(ty) >= 20 and ty in tx)):
            dups.append(y)
            continue
        if key(y) in pubs:
            pubs[key(y)].add(y)
            continue
        pubs[key(y)] = {y}
        chosen.append(key(y))
    excluded = set(groups) | {key(x)} | {key(d) for d in dups}
    support, members = {}, {}
    for K in chosen:
        cited = set().union(*(refs[r] for r in pubs[K]))
        for K2 in {key(c) for c in cited}:
            if K2 not in excluded:
                support[K2] = support.get(K2, 0) + 1
        for c in cited:
            members.setdefault(key(c), set()).add(c)
    ok = {K: s for K, s in support.items() if not grp_out(members[K])}
    if not ok:
        return chosen, []
    best = max(ok.values())
    return chosen, sorted((K for K, s in ok.items() if s == best), key=str)


class AnalyzeMatchesBruteForce(unittest.TestCase):
    def test_random_graphs(self):
        rng = random.Random(11)
        pool = ['Shared long publication title number one', 'Another long shared title, version two',
                'Third long repeated publication title']
        compared = 0
        for trial in range(150):
            n = rng.randint(8, 40)
            edges = sorted({(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(10, 160))})
            edges = [e for e in edges if e[0] != e[1]]
            cites = [rng.randint(0, 50) for _ in range(n)]
            nrefs = [rng.choice([-1, -1, rng.randint(0, 12)]) for _ in range(n)]
            titles = [rng.choice(pool) if rng.random() < .3 else rng.choice(['Editorial', f'unique study number {i} on things'])
                      for i in range(n)]
            key = lambda i: norm(titles[i]) if len(norm(titles[i])) >= 20 else ('#', i)
            with tempfile.TemporaryDirectory() as d:
                write_index(Path(d), n, edges, cites, titles, nrefs)
                g = Graph(Path(d))
                for x in rng.sample(range(n), 3):
                    k, nm = rng.choice([3, 5, 30]), rng.random() < .5
                    mr, mc = rng.choice([0, 4, 8]), rng.choice([0, 20, 40, 90])
                    out = g.analyze(x, k=k, normalize=nm, max_refs=mr, max_cites=mc)
                    Y, answers = brute(n, edges, cites, titles, x, k, nm, nrefs, mr, mc)
                    ctx = (trial, x, k, nm, mr, mc)
                    if 'error' in out:
                        self.assertFalse(answers, (ctx, out['error']))
                        continue
                    idx = lambda wid: int(wid[1:]) - 100
                    self.assertEqual([key(idx(i)) for i in out['similar']], Y, ctx)
                    self.assertEqual(sorted((key(idx(i)) for i in out['answers']), key=str), answers, ctx)
                    ids = [nd['id'] for nd in out['nodes'].values()]
                    self.assertEqual(len({key(idx(i)) for i in ids}), len(ids), ctx)  # one card per publication
                    compared += 1
        self.assertGreater(compared, 200)

    def test_duplicate_records_are_one_publication(self):
        T = ['X paper about something long enough', 'Shared reference study long title',
             'Shared reference number two long title', 'Similar paper three with a long title',
             'Similar paper four with a long title', 'Similar paper five with a long title',
             'Shared reference study long title',            # 6: another record of 1, which X cites
             'Candidate publication with two records',        # 7
             'Candidate publication with two records',        # 8: same publication as 7
             'Another candidate, cited once only',            # 9
             'Similar paper three with a long title']         # 10: another record of 3
        edges = [(0, 1), (0, 2), (3, 1), (3, 2), (3, 6), (3, 7), (4, 1), (4, 2), (4, 8), (4, 6),
                 (5, 1), (5, 9), (10, 1), (10, 2)]
        cites = [0, 10, 10, 0, 0, 0, 0, 1, 5, 0, 0]
        with tempfile.TemporaryDirectory() as d:
            write_index(Path(d), 11, edges, cites, T)
            out = Graph(Path(d)).analyze(0, k=5, max_refs=0, max_cites=0)
            self.assertEqual(out['similar'], ['W110', 'W104', 'W105'])     # 3 and 10 count once
            self.assertEqual(out['nodes']['W110']['records'], ['W103', 'W110'])
            self.assertEqual(out['answers'], ['W108'])                    # 7 + 8 together: 2 of 3
            self.assertEqual(out['nodes']['W108']['support'], 2)
            self.assertNotIn('W106', out['nodes'])                        # X cites it (via record 1)
            self.assertNotIn('W106', [i for c in out['columns'].values() for i in c])

    def test_outliers_are_drawn_but_never_chosen(self):
        edges = [(0, 1), (0, 2), (3, 1), (3, 2), (3, 5), (3, 6), (4, 1), (4, 2), (4, 6), (7, 2), (7, 6)]
        cites = [0, 900, 3, 0, 0, 2, 9000, 0]
        nrefs = [2, 0, 0, 4, 400, 0, 0, 2]
        with tempfile.TemporaryDirectory() as d:
            write_index(Path(d), 8, edges, cites, nrefs=nrefs)
            g = Graph(Path(d))
            out = g.analyze(0, k=5, max_refs=300, max_cites=500)
            self.assertEqual(out['stats']['refs_outliers'], 1)
            self.assertEqual(sorted(out['similar']), ['W103', 'W107'])
            self.assertEqual(out['answers'], ['W105'])
            col = {i: nd['column'] for i, nd in out['nodes'].items()}
            self.assertEqual((col['W101'], col['W104'], col['W106']), ('refs_out', 'similar_out', 'candidates_out'))
            self.assertEqual(g.analyze(0, k=5, max_refs=0, max_cites=0)['answers'], ['W106'])

    def test_budget_skips_most_cited_references_and_reports_them(self):
        edges = [(0, 1), (0, 2)] + [(i, 1) for i in range(3, 30)] + [(3, 2), (3, 31), (4, 2), (4, 31)]
        with tempfile.TemporaryDirectory() as d:
            write_index(Path(d), 32, edges, [0] * 32)
            out = Graph(Path(d)).analyze(0, k=5, budget=5, max_refs=0, max_cites=0)
            self.assertEqual(out['stats']['refs_skipped'], 1)
            self.assertEqual(out['stats']['skipped_examples'][0]['id'], 'W101')
            self.assertEqual(out['answers'], ['W131'])

    def test_copies_of_x_are_left_out_by_default(self):
        t = 'A long and specific title about protein folding'
        titles = [t, 'Faculty Opinions recommendation of ' + t, 'Another study', 'r1', 'r2', 'missing', 'copy only']
        edges = [(0, 3), (0, 4), (1, 3), (1, 4), (1, 6), (2, 3), (2, 5)]
        with tempfile.TemporaryDirectory() as d:
            write_index(Path(d), 7, edges, [0] * 7, titles)
            g = Graph(Path(d))
            out = g.analyze(0, k=5, max_refs=0, max_cites=0)
            self.assertEqual(out['similar'], ['W102'])
            self.assertEqual(out['answers'], ['W105'])
            self.assertEqual([p['id'] for p in out['stats']['duplicates_skipped']], ['W101'])
            self.assertEqual(g.analyze(0, k=5, skip_duplicates=False, max_refs=0, max_cites=0)['similar'], ['W101', 'W102'])


if __name__ == '__main__':
    unittest.main()
