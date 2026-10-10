"""Citation-graph walker: a tiny HTTP server over the memory-mapped citation index.

    uv run --with-requirements citewalk/requirements.txt python citewalk/server.py [--port 8798]

Endpoints (JSON):
  /api/resolve?q=W123 | https://openalex.org/W123 | 10.xxxx/... | https://doi.org/10...
  /api/analyze?id=W123&k=30&normalize=0&max_refs=300&max_cites=500
      Outlier = lists more than max_refs references or has more than max_cites citations (0 = no limit).
      R  = papers X cites; R' = R without outliers
      Y  = the k non-outlier papers (other than X) citing the most papers of R'
      C  = non-outlier papers cited by Y, not in R and not X, ranked by how many of Y cite them
      The top of C (all ties) is the answer. Outliers are still returned for drawing.
  /api/neighbors?id=W123&dir=refs|citers&limit=40   (most-cited first)
"""
import argparse
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse, parse_qs

import numpy as np
import polars as pl
import polars_hash  # noqa: F401

if __package__:
    from .atlas import AtlasStore, AtlasError
    from .atlas_overlay import ScienceAtlases
else:
    from atlas import AtlasStore, AtlasError
    from atlas_overlay import ScienceAtlases

HERE = Path(__file__).resolve().parent
DOI_PREFIX = r'^(https?://(dx\.)?doi\.org/|doi:)'
MIN_TITLE_KEY = 20
WORK = re.compile(r'^(?:https?://(?:api\.)?openalex\.org/(?:works/)?|openalex:)?[Ww]([0-9]+)$')


class Embeddings:
    """Read-only lookup of the DenseOn vectors written by pipeline/embeddings (committed shards only).

    Index: sorted OpenAlex work numbers -> (shard, row). New shards are picked up by refresh();
    vectors are read from memory-mapped shard files on demand.
    """
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.RLock()
        self.refresh_lock = threading.Lock()
        summary = json.loads((root / 'plan-summary.json').read_text())
        self.plan_id = summary['plan_id']
        self.dimensions = int(summary['dimensions'])
        self.shards, self.ids, self.loc = [], np.zeros(0, np.uint64), np.zeros(0, np.uint64)
        self.cache = OrderedDict()
        self.refreshed_at = 0.0
        self.refresh()

    def refresh(self):
        with self.refresh_lock:
            return self._refresh()

    def _refresh(self):
        t = time.time()
        summary = json.loads((self.root / 'plan-summary.json').read_text())
        if summary['plan_id'] != self.plan_id or int(summary['dimensions']) != self.dimensions:
            raise ValueError('Embedding plan changed; refusing to mix vector spaces')
        # Writer temporary directories start with .part-. Only atomically published
        # part-* directories count, even if a temporary directory has commit.json.
        commits = sorted(self.root.glob('shards/year=*/rg-*/part-*/commit.json'))
        known = set(self.shards)
        new = [str(c.parent) for c in commits if str(c.parent) not in known]
        if not new:
            self.refreshed_at = time.time()
            return 0
        base = len(self.shards)
        ids, loc = [self.ids], [self.loc]
        for k, d in enumerate(new):
            path = Path(d)
            commit = json.loads((path / 'commit.json').read_text())
            if commit['plan_id'] != self.plan_id:
                raise ValueError(f'Wrong embedding plan in {path}')
            w = np.load(path / 'work_ids.npy', allow_pickle=False)
            v = np.load(path / 'vectors.npy', mmap_mode='r', allow_pickle=False)
            if w.ndim != 1 or w.dtype != np.uint64 or v.shape != (len(w), self.dimensions) or v.dtype.kind != 'f':
                raise ValueError(f'Invalid committed embedding dimensions/IDs in {path}')
            if len(w) != commit['task']['rows'] or len(w) >= 2**32:
                raise ValueError(f'Invalid committed row count in {path}')
            del v
            ids.append(w)
            loc.append((np.uint64(base + k) << np.uint64(32)) | np.arange(len(w), dtype=np.uint64))
        ids, loc = np.concatenate(ids), np.concatenate(loc)
        o = np.argsort(ids, kind='stable')
        new_ids, new_loc = ids[o], loc[o]
        if len(new_ids) > 1 and np.any(new_ids[1:] == new_ids[:-1]):
            raise ValueError('Duplicate work IDs in committed embedding shards')
        with self.lock:
            self.shards = self.shards + new
            self.ids, self.loc = new_ids, new_loc
            self.refreshed_at = time.time()
        print(f'embeddings: +{len(new)} shards, {len(self.ids):,} vectors indexed ({time.time() - t:.1f}s)', flush=True)
        return len(new)

    def vectors(self, work_numbers):
        """float32 vectors for the given work numbers; rows for missing works are None."""
        with self.lock:
            ids, loc, shards = self.ids, self.loc, self.shards
        q = np.asarray(work_numbers, np.uint64)
        i = np.searchsorted(ids, q)
        out = [None] * len(q)
        for k, (w, j) in enumerate(zip(q, i)):
            if j < len(ids) and ids[j] == w:
                shard, row = int(loc[j] >> np.uint64(32)), int(loc[j] & np.uint64(0xFFFFFFFF))
                path = shards[shard]
                with self.lock:
                    if path not in self.cache:
                        self.cache[path] = np.load(Path(path) / 'vectors.npy', mmap_mode='r', allow_pickle=False)
                    self.cache.move_to_end(path)
                    out[k] = np.array(self.cache[path][row], dtype=np.float32, copy=True)
                    while len(self.cache) > 128:
                        self.cache.popitem(last=False)
        return out

    def stats(self):
        with self.lock:
            return dict(vectors=int(len(self.ids)), shards=len(self.shards), refreshed_at=self.refreshed_at,
                        plan_id=self.plan_id, dimensions=self.dimensions)


def _mds2d(D, iters=300):
    """2-D positions whose distances follow D, weighting near pairs most (Sammon-like stress, SMACOF)."""
    n = len(D)
    if n == 1:
        return np.zeros((1, 2))
    J = np.eye(n) - 1.0 / n
    w, v = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    top = np.argsort(w)[::-1][:2]
    X = v[:, top] * np.sqrt(np.maximum(w[top], 1e-9))
    W = 1.0 / (D ** 2 + 1e-4)
    np.fill_diagonal(W, 0)
    V = -W.copy()
    np.fill_diagonal(V, W.sum(1))
    Vp = np.linalg.pinv(V)
    for _ in range(iters):
        d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) + 1e-9
        B = -W * D / d
        np.fill_diagonal(B, 0)
        np.fill_diagonal(B, -B.sum(1))
        X = Vp @ B @ X
    return X


def _agglomerate(D, k):
    """Ward clustering into k groups. D = cosine distance; squared Euclidean on unit vectors = 2 D."""
    n = len(D)
    groups = {i: [i] for i in range(n)}
    M = 2.0 * D.astype(float)
    np.fill_diagonal(M, np.inf)
    size = np.ones(n)
    while len(groups) > k:
        i, j = np.unravel_index(np.argmin(M), M.shape)
        i, j = min(i, j), max(i, j)
        ni, nj, dij = size[i], size[j], M[i, j]
        alive = np.array(sorted(groups))
        nk = size[alive]
        row = ((ni + nk) * M[i, alive] + (nj + nk) * M[j, alive] - nk * dij) / (ni + nj + nk)   # Lance–Williams (Ward)
        M[i, alive] = row; M[alive, i] = row; M[i, i] = np.inf
        M[j, :] = np.inf; M[:, j] = np.inf
        size[i] = ni + nj
        groups[i] += groups.pop(j)
    return list(groups.values())


STOP = set('a an and are as at by for from in into is of on or the to with without via using its their this that than '
           'new study analysis review based between among during after under over about de la et des le les en art state'.split())


def legacy_semantic_map(graph, emb, result):
    """Retired per-query map experiment; the HTTP API now uses AtlasStore instead."""
    if emb is None or 'nodes' not in result:
        return
    nodes = result['nodes']
    vec = {}
    for nid, n in nodes.items():
        recs = [n['id']] + [r for r in n['records'] if r != n['id']]
        for r, v in zip(recs, emb.vectors([int(r[1:]) for r in recs])):
            if v is not None:
                vec[nid] = v
                break
        node = graph.node_of_work(int(nid[1:]))
        t = graph.topic_of(node) if node is not None else None
        n['topic'] = t['name'] if t else None
        n['subfield'] = t['subfield'] if t else None
    ids = list(vec)
    ref, method = vec.get(result['focus']), 'distance to X'
    if ref is None and vec:
        c = np.mean([vec[i] for i in ids], axis=0)
        ref, method = c / max(np.linalg.norm(c), 1e-12), 'distance to the centroid (X has no vector)'
    for nid, n in nodes.items():
        n['sem'] = None if nid not in vec or ref is None else round(float(1 - np.dot(vec[nid], ref)), 4)
        n['mx'] = n['my'] = None
        n['region'] = None
    regions = []
    if ids:
        V = np.stack([vec[i] for i in ids])
        D = np.clip(1 - V @ V.T, 0, 2)
        np.fill_diagonal(D, 0)
        XY = _mds2d(D)
        span = max(np.ptp(XY[:, 0]), np.ptp(XY[:, 1]), 1e-9)
        XY = (XY - XY.mean(0)) / span
        for i, nid in enumerate(ids):
            nodes[nid]['mx'], nodes[nid]['my'] = round(float(XY[i, 0]), 4), round(float(XY[i, 1]), 4)
        k = int(min(len(ids), max(2, min(6, round(len(ids) / 8)))))
        groups = sorted(_agglomerate(D, k), key=len, reverse=True)
        from collections import Counter
        used = set()
        for r, g in enumerate(groups):
            members = [ids[i] for i in g]
            for m in members:
                nodes[m]['region'] = r
            topics = Counter(nodes[m]['topic'] for m in members if nodes[m]['topic'])
            subs = Counter(nodes[m]['subfield'] for m in members if nodes[m]['subfield'])
            name, kind, share = None, 'words', None
            for t, c in topics.most_common():          # most common topic not already naming a bigger area
                if t not in used and (c / len(members) >= 0.25 or len(members) <= 3):
                    name, kind, share = t, 'topic', c / len(members)
                    break
            if not name:
                for t, c in subs.most_common():
                    if t not in used and c / len(members) >= 0.25:
                        name, kind, share = t, 'subfield', c / len(members)
                        break
            if not name:                                # distinctive title words
                inside = Counter(w for m in members for w in set(re.findall(r'[a-z][a-z0-9-]{2,}', nodes[m]['title'].lower())) if w not in STOP)
                overall = Counter(w for m in ids for w in set(re.findall(r'[a-z][a-z0-9-]{2,}', nodes[m]['title'].lower())))
                ranked = sorted(inside, key=lambda w: (-(inside[w] ** 2) / overall[w], w))
                name = ' · '.join(ranked[:3]) or 'other papers'
            used.add(name)
            mean = V[g].mean(0)
            spread = float(np.mean(1 - V[g] @ (mean / max(np.linalg.norm(mean), 1e-12))))
            regions.append(dict(id=r, name=name, name_kind=kind, share=round(share, 2) if share else None,
                                size=len(members), spread=round(spread, 3)))
    result['regions'] = regions
    result['stats']['semantic'] = dict(method=method, with_vector=len(vec), nodes=len(nodes),
                                       model='DenseOn (title + abstract, English works with an abstract)',
                                       indexed=emb.stats()['vectors'], topics_loaded=graph.topics is not None)


class Graph:
    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / 'meta.json').read_text())
        L = lambda n: np.load(path / f'{n}.npy', mmap_mode='r')
        self.wn, self.year, self.cites, self.nrefs = L('wn'), L('year'), L('cites'), L('nrefs')
        self.kind, self.complete = L('kind'), L('complete')
        self.title_row, self.title_off = L('title_row'), L('title_off')
        self.titles = np.memmap(path / 'titles.bin', dtype=np.uint8, mode='r')
        self.doi_hash, self.doi_node = L('doi_hash'), L('doi_node')
        self.fwd_off, self.fwd, self.bwd_off, self.bwd = L('fwd_off'), L('fwd'), L('bwd_off'), L('bwd')
        self.types = self.meta['types']
        self.topic, self.topics = None, None
        self.load_topics()

    def load_topics(self):
        if self.topics is None and (self.path / 'topic.npy').exists() and (self.path / 'topics.json').exists():
            self.topic = np.load(self.path / 'topic.npy', mmap_mode='r')
            self.topics = json.loads((self.path / 'topics.json').read_text())

    def topic_of(self, i):
        self.load_topics()   # picks the file up once build_topics.py has finished
        if self.topics is None:
            return None
        t = int(self.topic[i])
        return self.topics[t] if t >= 0 else None

    # ── lookups ──
    def node_of_work(self, w):
        i = int(np.searchsorted(self.wn, w))
        return i if i < len(self.wn) and int(self.wn[i]) == w else None

    def resolve(self, q):
        q = (q or '').strip()
        m = WORK.match(q)
        if m:
            i = self.node_of_work(int(m.group(1)))
            return [i] if i is not None else []
        if '10.' in q:
            h = pl.DataFrame({'d': [q]}).select(
                pl.col('d').str.strip_chars().str.to_lowercase().str.replace(DOI_PREFIX, '').nchash.wyhash())['d'][0]
            lo = int(np.searchsorted(self.doi_hash, np.uint64(h), 'left'))
            hi = int(np.searchsorted(self.doi_hash, np.uint64(h), 'right'))
            return [int(x) for x in self.doi_node[lo:hi]]
        return []

    def title(self, i):
        r = int(self.title_row[i])
        return bytes(self.titles[int(self.title_off[r]):int(self.title_off[r + 1])]).decode('utf-8', 'replace')

    def refs(self, i):
        return np.asarray(self.fwd[int(self.fwd_off[i]):int(self.fwd_off[i + 1])])

    def citers(self, i):
        return np.asarray(self.bwd[int(self.bwd_off[i]):int(self.bwd_off[i + 1])])

    def out_degree(self, ids):
        ids = np.asarray(ids, np.int64)
        return (self.fwd_off[ids + 1] - self.fwd_off[ids]).astype(np.int64)

    def in_degree(self, ids):
        ids = np.asarray(ids, np.int64)
        return (self.bwd_off[ids + 1] - self.bwd_off[ids]).astype(np.int64)

    def info(self, i):
        i = int(i)
        k = int(self.kind[i])
        return dict(id=f'W{int(self.wn[i])}', title=self.title(i), year=int(self.year[i]) or None,
                    cited_by=int(self.cites[i]), refs_reported=int(self.nrefs[i]),
                    refs_indexed=int(self.fwd_off[i + 1] - self.fwd_off[i]),
                    citers_indexed=int(self.bwd_off[i + 1] - self.bwd_off[i]),
                    type=self.types[k] if k < len(self.types) else None, complete=bool(self.complete[i]))

    # ── the analysis ──
    @staticmethod
    def _norm_title(t):
        return re.sub(r'[\W_]+', ' ', (t or '').casefold()).strip()

    def pubkey(self, i):
        """Publication identity: records with the same normalized title (>= 20 characters) are one publication."""
        t = self._norm_title(self.title(i))
        return t if len(t) >= MIN_TITLE_KEY else f'#{int(i)}'

    def ref_count(self, ids):
        """References a paper lists: OpenAlex's reported count, or the indexed count when unknown."""
        ids = np.asarray(ids, np.int64)
        reported = self.nrefs[ids].astype(np.int64)
        return np.where(reported >= 0, reported, self.out_degree(ids))

    def outliers(self, ids, max_refs, max_cites):
        """Per record: more than max_refs references or more than max_cites citations (0 = no limit)."""
        ids = np.asarray(ids, np.int64)
        out = np.zeros(len(ids), bool)
        if max_cites:
            out |= self.cites[ids].astype(np.int64) > max_cites
        if max_refs:
            out |= self.ref_count(ids) > max_refs
        return out

    def analyze(self, x, k=30, normalize=False, max_refs=300, max_cites=500, budget=150_000_000,
                n_candidates=12, n_refs_shown=40, skip_duplicates=True, n_outliers_shown=8):
        t0 = time.time()
        keys = {}

        def key(i):
            i = int(i)
            if i not in keys:
                keys[i] = self.pubkey(i)
            return keys[i]

        def group_outlier(members):
            """Reason a publication (all its records together) is an outlier, or None."""
            m = np.asarray(sorted(members), np.int64)
            c = int(self.cites[m].astype(np.int64).clip(min=0).sum())
            r = int(self.ref_count(m).max())
            why = []
            if max_cites and c > max_cites:
                why.append(f'{c:,} citations' + (f' over {len(m)} records' if len(m) > 1 else ''))
            if max_refs and r > max_refs:
                why.append(f'{r:,} references')
            return ' and '.join(why) or None

        def most_cited(members):
            return max(members, key=lambda i: (int(self.cites[i]), -i))

        # 1. X's references, grouped into publications; outlier publications are not used
        R = np.unique(self.refs(x)).astype(np.int64)
        if len(R) == 0:
            return dict(error='This paper has no references inside the snapshot, so there is nothing to compare.',
                        focus=self.info(x))
        xkey = key(x)
        Rgroups = {}
        for r in R:
            Rgroups.setdefault(key(r), []).append(int(r))
        Rout = {K: group_outlier(m) for K, m in Rgroups.items()}
        use_keys = [K for K in Rgroups if not Rout[K]]
        if not use_keys:
            return dict(error=f'All {len(Rgroups)} publications this paper cites are outliers (over the limits). '
                              'Raise or remove the limits.', focus=self.info(x))
        used_recs = np.asarray([r for K in use_keys for r in Rgroups[K]], np.int64)
        kidx = {K: n for n, K in enumerate(use_keys)}
        rec_key = np.asarray([kidx[key(r)] for r in used_recs], np.int64)
        deg = self.in_degree(used_recs)
        take, skipped, total = [], [], 0
        for j in np.argsort(deg, kind='stable'):  # cost guard; with outliers removed it practically never triggers
            if total + deg[j] > budget and take:
                skipped.append(int(used_recs[j]))
            else:
                take.append(int(j)); total += int(deg[j])
        take = np.asarray(sorted(take), np.int64)
        n_used = len(set(rec_key[take].tolist()))
        # 2. everyone citing them; count distinct shared publications per citing paper (bibliographic coupling)
        citing = np.concatenate([self.citers(used_recs[j]) for j in take]).astype(np.int64)
        pairs = pl.DataFrame({'y': citing, 'g': np.repeat(rec_key[take], deg[take])}).unique()
        vc = pairs.filter(pl.col('y') != x).group_by('y').len(name='shared')
        if vc.height == 0:
            return dict(error='No other paper shares a (non-outlier) reference with this one.', focus=self.info(x))
        y = vc['y'].to_numpy().astype(np.int64)
        shared = vc['shared'].to_numpy().astype(np.int64)
        cosine = shared / np.sqrt(np.maximum(self.out_degree(y), 1) * n_used)
        primary = cosine if normalize else shared
        rank = np.lexsort((y, -cosine, -primary))      # score desc, then cosine desc, then id
        y_out = self.outliers(y, max_refs, max_cites)
        # 3. the k most coupled publications; outliers and copies of X are set aside
        tx = self._norm_title(self.title(x)) if skip_duplicates else ''
        pubs, pub_of, dups, sim_out, n_sim_out = [], {}, [], [], 0
        for j in rank:
            if len(pubs) == k:
                break
            yi = int(y[j])
            if y_out[j]:
                n_sim_out += 1
                if len(sim_out) < n_outliers_shown:
                    sim_out.append((yi, int(shared[j])))
                continue
            if tx:
                ty = self._norm_title(self.title(yi))
                if len(tx) >= MIN_TITLE_KEY and (tx in ty or (len(ty) >= MIN_TITLE_KEY and ty in tx)):
                    dups.append(yi)
                    continue
            K = key(yi)
            if K in pub_of:                       # another record of a publication already chosen
                pub_of[K]['records'].append(yi)
                continue
            p = dict(key=K, rep=yi, records=[yi], shared=int(shared[j]), cosine=float(cosine[j]))
            pubs.append(p); pub_of[K] = p
        # 4. tally, per publication, how many of the k cite it (any of its records)
        members, support = {}, {}
        for p in pubs:
            refs = np.unique(np.concatenate([self.refs(r) for r in p['records']]))
            p['ref_keys'] = set()
            for c in refs:
                K = key(c)
                p['ref_keys'].add(K)
                members.setdefault(K, set()).add(int(c))
            for K in p['ref_keys']:
                support[K] = support.get(K, 0) + 1
        excluded = set(Rgroups) | {xkey} | {key(d) for d in dups}
        cand = []
        for K, sup in support.items():
            if K in excluded:
                continue
            rep = pub_of[K]['rep'] if K in pub_of else most_cited(members[K])
            cand.append((K, sup, rep, group_outlier(members[K])))
        cand.sort(key=lambda c: (-c[1], -int(self.cites[c[2]]), c[2]))
        inside = [c for c in cand if not c[3]]
        if not inside:
            return dict(error='The similar papers cite nothing (except outliers) that this paper does not already cite.',
                        focus=self.info(x))
        best = inside[0][1]
        n_best = sum(1 for c in inside if c[1] == best)
        top = inside[:max(n_candidates, n_best)]
        above = [c for c in cand if c[3] and c[1] >= best]
        # ── one card per publication ──
        nodes, links = {}, []

        def node(K, rep, column, **extra):
            recs = sorted(members.get(K, set()) | set(Rgroups.get(K, [])) | set(pub_of[K]['records'] if K in pub_of else []) | {rep})
            n = dict(self.info(rep), column=column, records=[f'W{int(self.wn[i])}' for i in recs],
                     x_cites=K in Rgroups, also_similar=K in pub_of and column.startswith('candidates'), **extra)
            nodes[n['id']] = n
            placed[K] = n['id']
            return n['id']

        placed = {}
        X = node(xkey, x, 'x', outlier=group_outlier([x]))
        cand_cols = [(c, 'candidates') for c in top] + [(c, 'candidates_out') for c in above[:n_outliers_shown]]
        for (K, sup, rep, why), col in cand_cols:
            node(K, rep, col, support=sup, outlier=why, answer=(sup == best and not why),
                 shared=pub_of[K]['shared'] if K in pub_of else None)
        for p in pubs:
            if p['key'] not in placed:
                node(p['key'], p['rep'], 'similar', shared=p['shared'])
        ref_support = {K: sum(1 for p in pubs if K in p['ref_keys']) for K in Rgroups}
        def ref_order(K):
            return (-ref_support[K], -int(self.cites[most_cited(Rgroups[K])]), K)
        shared_refs = sorted((K for K in Rgroups if ref_support[K] and K not in placed and not Rout[K]), key=ref_order)
        out_refs = sorted((K for K in Rgroups if ref_support[K] and K not in placed and Rout[K]), key=ref_order)
        for K in shared_refs[:n_refs_shown]:
            node(K, most_cited(Rgroups[K]), 'refs', ref_support=ref_support[K])
        for K in out_refs[:n_outliers_shown]:
            node(K, most_cited(Rgroups[K]), 'refs_out', ref_support=ref_support[K], outlier=Rout[K])
        for yi, sh in sim_out:
            if key(yi) not in placed:
                node(key(yi), yi, 'similar_out', shared=sh, outlier=group_outlier([yi]))
        # lines between cards
        for K, nid in placed.items():
            if K in Rgroups and nid != X:
                links.append(dict(s=X, t=nid, kind='x-ref'))
        for p in pubs:
            src = placed[p['key']]
            for K in p['ref_keys']:
                if K in placed and placed[K] != src and K != xkey:
                    col = nodes[placed[K]]['column']
                    kind = 'y-ref' if col.startswith('refs') else 'y-cand' if col.startswith('candidates') else 'y-sim'
                    links.append(dict(s=src, t=placed[K], kind=kind))
        for yi, _ in sim_out:
            src = placed.get(key(yi))
            if src and nodes[src]['column'] == 'similar_out':
                for c in np.intersect1d(self.refs(yi), R):
                    if key(c) in placed and nodes[placed[key(c)]]['column'].startswith('refs'):
                        links.append(dict(s=src, t=placed[key(c)], kind='yo-ref'))
        cols = {c: [n['id'] for n in nodes.values() if n['column'] == c]
                for c in ('refs', 'refs_out', 'similar', 'similar_out', 'candidates', 'candidates_out')}
        merged = sum(len(n['records']) - 1 for n in nodes.values())
        return dict(focus=X, nodes=nodes, columns=cols, links=links,
                    answers=[placed[c[0]] for c in inside if c[1] == best],
                    similar=[placed[p['key']] for p in pubs],
                    stats=dict(refs_indexed=len(R), ref_publications=len(Rgroups),
                               refs_outliers=sum(1 for v in Rout.values() if v), refs_used=n_used,
                               refs_shared=len(shared_refs) + len(out_refs), refs_shared_hidden=max(0, len(shared_refs) - n_refs_shown),
                               refs_not_shared=sum(1 for K in Rgroups if not ref_support[K]),
                               refs_skipped=len(skipped),
                               skipped_examples=[self.info(r) for r in sorted(skipped, key=lambda r: -int(self.in_degree([r])[0]))[:3]],
                               citing_links_scanned=int(total), papers_sharing_a_reference=int(vc.height),
                               similar_outliers_skipped=n_sim_out, candidate_outliers_above=len(above),
                               max_refs=max_refs, max_cites=max_cites, k=len(pubs), normalize=bool(normalize),
                               best_support=best, records_merged=merged,
                               duplicates_skipped=[self.info(j) for j in dups],
                               seconds=round(time.time() - t0, 2)))

    def neighbors(self, x, direction, limit=40):
        ids = self.refs(x) if direction == 'refs' else self.citers(x)
        ids = np.unique(ids).astype(np.int64)
        total = len(ids)
        if total > limit:
            c = self.cites[ids]
            ids = ids[np.argsort(-c, kind='stable')[:limit]]
        return dict(focus=self.info(x), direction=direction, total=total,
                    items=[self.info(i) for i in ids])


def make_handler(graph, emb=None, atlas_root=None, science=None):
    static = HERE / 'static'
    atlases = AtlasStore(atlas_root or (graph.path / 'local-maps'))
    science = science or ScienceAtlases(Path('/mnt/fast/science-atlas/build'), Path('/mnt/fast/science-atlas/vectors'))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(time.strftime('%H:%M:%S ') + (fmt % args), flush=True)

        def send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == '/api/resolve':
                    found = graph.resolve(q.get('q'))
                    return self.send(200, dict(matches=[graph.info(i) for i in found]))
                if u.path == '/api/walk':          # the walk, positioned on a science-atlas build
                    found = graph.resolve(q.get('id'))
                    if not found:
                        return self.send(404, dict(error=f"Not found in the citation snapshot: {q.get('id')!r}"))
                    try:
                        atlas = science.get(q.get('build'))
                    except ValueError as e:
                        return self.send(400, dict(error=str(e)))
                    k = max(1, min(200, int(q.get('k', 30))))
                    limit = lambda name, default: max(0, int(q.get(name, default)))
                    started = time.perf_counter()
                    result = graph.analyze(found[0], k=k, normalize=q.get('normalize') == '1',
                                           max_refs=limit('max_refs', 300), max_cites=limit('max_cites', 500),
                                           skip_duplicates=q.get('dups', '0') != '1')
                    if 'nodes' in result:
                        for n in result['nodes'].values():
                            i = graph.node_of_work(int(n['id'][1:]))
                            t = graph.topic_of(i) if i is not None else None
                            n['topic'] = t['name'] if t else None
                            n['field'] = t['field'] if t else None
                        result['atlas_overlay'] = atlas.locate(result['nodes'], emb)
                        result['stats']['request_seconds'] = round(time.perf_counter() - started, 3)
                    return self.send(200, result)
                if u.path in ('/api/analyze', '/api/neighbors'):
                    found = graph.resolve(q.get('id'))
                    if not found:
                        return self.send(404, dict(error=f"Not found in the snapshot: {q.get('id')!r}"))
                    if u.path == '/api/analyze':
                        k = max(1, min(200, int(q.get('k', 30))))
                        limit = lambda name, default: max(0, int(q.get(name, default)))
                        started = time.perf_counter()
                        result = graph.analyze(found[0], k=k, normalize=q.get('normalize') == '1',
                                               max_refs=limit('max_refs', 300), max_cites=limit('max_cites', 500),
                                               skip_duplicates=q.get('dups', '0') != '1')
                        atlases.annotate(graph, emb, result, identifier=q.get('atlas') or None)
                        if 'stats' in result:
                            result['stats']['request_seconds'] = round(time.perf_counter() - started, 3)
                        return self.send(200, result)
                    return self.send(200, graph.neighbors(found[0], q.get('dir', 'refs'),
                                                          max(1, min(500, int(q.get('limit', 40))))))
                if u.path == '/api/meta':
                    return self.send(200, graph.meta)
                name = 'atlas.html' if u.path in ('/', '') else u.path.lstrip('/')
                f = (static / name).resolve()
                if static in f.parents and f.is_file():
                    ctype = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css'}.get(f.suffix, 'application/octet-stream')
                    return self.send(200, f.read_bytes(), ctype)
                return self.send(404, dict(error='not found'))
            except AtlasError as e:
                return self.send(409, dict(error=str(e), code='atlas_unavailable'))
            except Exception as e:  # report, keep serving
                import traceback; traceback.print_exc()
                return self.send(500, dict(error=f'{type(e).__name__}: {e}'))

    return Handler


def latest_index(root: Path):
    done = sorted(p for p in root.glob('*') if (p / 'meta.json').exists() and '-first' not in p.name)
    if not done:
        raise SystemExit(f'No built index under {root}; run citewalk/build_index.py first')
    return done[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', type=Path)
    ap.add_argument('--root', type=Path, default=Path('/mnt/fast/citewalk'))
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8798)
    ap.add_argument('--embeddings', type=Path, default=Path('/mnt/fast/openalex-embeddings/denseon-english-abstracts'))
    ap.add_argument('--atlas-root', type=Path, help='saved frozen local maps (default: <index>/local-maps)')
    args = ap.parse_args()
    graph = Graph(args.index or latest_index(args.root))
    print(f"index {graph.path} · {graph.meta['nodes']:,} papers · {graph.meta['edges']:,} citations", flush=True)
    emb = None
    if args.embeddings and args.embeddings.exists():
        emb = Embeddings(args.embeddings)
        def keep_fresh():   # the backfill keeps writing shards; pick them up
            while True:
                time.sleep(600)
                try:
                    emb.refresh()
                except Exception as e:
                    print('embeddings refresh failed:', e, flush=True)
        threading.Thread(target=keep_fresh, daemon=True).start()
    ThreadingHTTPServer((args.host, args.port), make_handler(graph, emb, args.atlas_root)).serve_forever()


if __name__ == '__main__':
    main()
