"""The authorship index the other way round: each author's papers, plus what a profile and a name search need.

    uv run --with-requirements citewalk/requirements.txt python citewalk/build_author_works.py

Reads /mnt/fast/citewalk/<generation>/authors/ (build_authors.py) and writes beside it, in authors/:

  by_off.npy    uint64[M+1]  CSR offsets, one row per author (row = position in author_an.npy)
  by_node.npy   uint32[E]    the author's papers (citation-index nodes), oldest first; ties by node
  by_pos.npy    uint16[E]    the author's position on each (1 = first)
  a_works.npy   uint32[M]    publications (search scope types) of the author
  a_cites.npy   uint32[M]    citations to those publications, summed (capped at 2^32-1)
  a_h.npy       uint16[M]    h-index over them
  a_year0.npy   uint16[M]    first publication year (0 = none)
  a_year1.npy   uint16[M]    latest publication year
  a_topic.npy   int16[M]     the author's commonest primary topic (index into ../topics.json; -1 = none)
  meta_works.json

Positions and years come from the citation index (year.npy, cites.npy, kind.npy, topic.npy).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from citewalk.science.scope import SCHOLARLY_TYPES  # noqa: E402


def log(msg):
    print(time.strftime('%H:%M:%S ') + msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--citations', type=Path, help='citation index directory (default: /mnt/fast/citewalk/<generation>)')
    args = ap.parse_args()
    if args.citations:
        cit = args.citations
    else:
        from pipeline.tidy.dataset import manifest
        cit = Path('/mnt/fast/citewalk') / manifest()['generation']
    A = cit / 'authors'
    if (A / 'meta_works.json').exists():
        log(f'already built at {A}')
        return
    t0 = time.time()
    meta = json.loads((cit / 'meta.json').read_text())
    types = meta['types']
    is_pub = np.array([t in SCHOLARLY_TYPES for t in types], bool)
    off = np.load(A / 'auth_off.npy', mmap_mode='r')
    auth = np.load(A / 'auth.npy')                      # uint32[E]
    pos = np.load(A / 'auth_pos.npy')                   # uint16[E]
    M = int(np.load(A / 'author_an.npy', mmap_mode='r').shape[0])
    N = int(off.shape[0]) - 1
    E = len(auth)
    log(f'{E:,} authorships, {M:,} authors, {N:,} nodes')

    if (A / 'by_pos.npy').exists():                      # a run stopped after the sort: pick up from its files
        log('sorted files found, reusing them')
        by_node = np.load(A / 'by_node.npy')
        by_off = np.load(A / 'by_off.npy')
        cnt = np.diff(by_off)
        by_auth = np.repeat(np.arange(M, dtype=np.uint32), cnt.astype(np.int64))
        del auth, pos
        year = np.load(cit / 'year.npy')
    else:
        log('node of each authorship')
        counts = np.diff(np.asarray(off)).astype(np.int64)
        node = np.repeat(np.arange(N, dtype=np.uint32), counts)
        del counts
        assert len(node) == E

        log('sort by author, then year, then node')
        year = np.load(cit / 'year.npy')                    # uint16[N]
        # two stable sorts (author, year, node won't fit one int64 key): the input is in node order already
        o1 = np.argsort(year[node], kind='stable')
        o2 = np.argsort(auth[o1], kind='stable')
        order = o1[o2]
        del o1, o2
        by_node = node[order]
        by_pos = pos[order]
        by_auth = auth[order]
        del order, node, pos, auth
        cnt = np.bincount(by_auth, minlength=M).astype(np.uint64)
        by_off = np.zeros(M + 1, np.uint64)
        np.cumsum(cnt, out=by_off[1:])
        assert int(by_off[-1]) == E
        np.save(A / 'by_off.npy', by_off)
        np.save(A / 'by_node.npy', by_node)
        np.save(A / 'by_pos.npy', by_pos)
        del by_pos

    log('per-author stats')
    kind = np.load(cit / 'kind.npy')
    cites = np.load(cit / 'cites.npy')                  # uint32[N]
    topic = np.load(cit / 'topic.npy') if (cit / 'topic.npy').exists() else None
    pub = is_pub[kind[by_node]]                         # bool[E]: the authorship is on a publication
    del kind
    a_works = np.bincount(by_auth, weights=pub, minlength=M).astype(np.uint32)
    c = cites[by_node].astype(np.float64) * pub
    a_cites = np.minimum(np.bincount(by_auth, weights=c, minlength=M), 2**32 - 1).astype(np.uint32)
    y = year[by_node].astype(np.int64)
    starts = by_off[:-1].astype(np.int64)             # every author has at least one authorship
    a_year1 = np.maximum.reduceat(np.where(pub, y, 0), starts)
    a_year0 = np.minimum.reduceat(np.where(pub & (y > 0), y, 1 << 20), starts)   # unknown years (0) don't count
    a_year0[a_year0 == 1 << 20] = 0
    np.save(A / 'a_works.npy', a_works)
    np.save(A / 'a_cites.npy', a_cites)
    np.save(A / 'a_year0.npy', a_year0.astype(np.uint16))
    np.save(A / 'a_year1.npy', a_year1.astype(np.uint16))
    del y, c

    log('h-index')
    # within each author's block, citations of their publications sorted descending; h = #{i: c_i >= i+1}
    cp = np.where(pub, cites[by_node].astype(np.int64), -1)
    del cites
    # sort within author: key = author << 34 | (2^33 - 1 - cites) keeps authors together, citations descending
    k2 = (by_auth.astype(np.int64) << 34) | ((1 << 33) - 1 - np.minimum(cp, (1 << 33) - 2))
    o2 = np.argsort(k2, kind='stable')
    del k2
    cs = cp[o2]
    del cp, o2
    rank = np.arange(E, dtype=np.int64) - np.repeat(by_off[:-1].astype(np.int64), cnt.astype(np.int64))
    ok = (cs >= rank + 1) & (cs >= 0)
    del cs, rank
    a_h = np.bincount(by_auth, weights=ok, minlength=M)
    del ok
    np.save(A / 'a_h.npy', np.minimum(a_h, 65535).astype(np.uint16))

    log('commonest topic')
    a_topic = np.full(M, -1, np.int16)
    if topic is not None:
        tp = topic[by_node].astype(np.int64)
        keep = pub & (tp >= 0)
        pair = by_auth[keep].astype(np.int64) * 8192 + tp[keep]
        del tp
        u, cu = np.unique(pair, return_counts=True)
        del pair
        ua = u // 8192
        # per author, the pair with the highest count: sort by (author, -count), take first of each author
        o3 = np.lexsort((-cu, ua))
        ua, u = ua[o3], u[o3]
        first = np.r_[0, np.flatnonzero(np.diff(ua)) + 1]
        a_topic[ua[first]] = (u[first] % 8192).astype(np.int16)
    np.save(A / 'a_topic.npy', a_topic)

    (A / 'meta_works.json').write_text(json.dumps(dict(
        generation=meta['generation'], authors=M, authorships=E,
        authors_with_publications=int((a_works > 0).sum()), built_at=time.time(),
        scope='citewalk/science/scope.py SCHOLARLY_TYPES'), indent=1))
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
