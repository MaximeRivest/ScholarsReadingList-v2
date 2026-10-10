"""Positions of citation-walk papers on the frozen science-atlas build (citewalk/science).

Order of evidence for each publication (its records are tried in order, representative first):
  1. 'dot'        the paper is one of the atlas's displayed dots: its exact stored position
  2. 'reference'  the paper is one of the atlas's fitted reference papers: its fitted position
  3. 'placed'     the paper has a DenseOn vector (backfill, or the atlas's own sample vectors):
                  the atlas's frozen map function (science/placement.py), unchanged
  4. None         no vector: listed as unplaced; no position is guessed
No position depends on which other papers are in the walk, so the walk never moves the map.
"""
from pathlib import Path
import json
import re
import sys
import threading

import numpy as np
import polars as pl

SCIENCE = Path(__file__).resolve().parent / 'science'
if str(SCIENCE) not in sys.path:
    sys.path.insert(0, str(SCIENCE))
import placement  # noqa: E402  (the atlas's own frozen map function)

BUILD_ID = re.compile(r'^[0-9A-Za-z_-]+$')


class ScienceAtlas:
    def __init__(self, build: Path, sample_vectors: Path | None = None):
        self.build = Path(build)
        self.meta = json.loads((self.build / 'meta.json').read_text())
        work = pl.read_parquet(self.build / 'info.parquet', columns=['work'])['work'].to_numpy().astype(np.uint64)
        xy = np.fromfile(self.build / 'points.f32', np.float32).reshape(-1, 2)
        if len(work) != len(xy):
            raise ValueError('atlas build: info.parquet and points.f32 disagree')
        o = np.argsort(work, kind='stable')
        self.dot_work, self.dot_xy, self.dot_row = work[o], xy[o], o
        if len(self.dot_work) > 1 and np.any(self.dot_work[1:] == self.dot_work[:-1]):
            raise ValueError('atlas build: duplicate displayed works')
        with np.load(self.build / 'reference.npz', allow_pickle=False) as r:
            if int(r['k']) != placement.K or float(r['tau']) != placement.TAU:
                raise ValueError('atlas build was made with different placement parameters')
            rw = r['work'].astype(np.uint64)
            ro = np.argsort(rw, kind='stable')
            self.ref_work, self.ref_xy_sorted = rw[ro], r['coords'][ro].astype(np.float32)
            self.ref_X = np.ascontiguousarray(r['vectors'], dtype=np.float32)   # fitted order, for placement
            self.ref_xy = r['coords'].astype(np.float32)
        self.sample_work = self.sample_vec = None
        if sample_vectors and (sample_vectors / 'work.npy').exists():
            self.sample_work = np.load(sample_vectors / 'work.npy', mmap_mode='r')
            self.sample_vec = np.load(sample_vectors / 'vectors.npy', mmap_mode='r')
            if self.sample_vec.shape != (len(self.sample_work), self.ref_X.shape[1]):
                raise ValueError('atlas sample vectors have the wrong shape')
        self.lock = threading.Lock()

    @property
    def id(self):
        return self.meta['id']

    @staticmethod
    def _find(sorted_ids, w):
        i = int(np.searchsorted(sorted_ids, np.uint64(w)))
        return i if i < len(sorted_ids) and int(sorted_ids[i]) == w else None

    def _sample_vector(self, w):
        if self.sample_work is None:
            return None
        i = self._find(self.sample_work, w)
        return None if i is None else np.asarray(self.sample_vec[i], np.float32)

    def locate(self, nodes, emb):
        """nodes: {id: node with 'id' and 'records'}. Sets node['ax'], ['ay'], ['place'], ['place_record']."""
        same_plan = emb is not None and getattr(emb, 'plan_id', None) == self.meta.get('embedding_plan')
        todo = []                                   # (node id, record, vector) to place by the map function
        for nid, n in nodes.items():
            n.update(ax=None, ay=None, place=None, place_record=None)
            records = [n['id']] + [r for r in n.get('records', []) if r != n['id']]
            numbers = [int(r[1:]) for r in records]
            hit = None
            for r, w in zip(records, numbers):      # exact positions first, for any record
                i = self._find(self.dot_work, w)
                if i is not None:
                    hit = ('dot', r, self.dot_xy[i]); break
            if hit is None:
                for r, w in zip(records, numbers):
                    i = self._find(self.ref_work, w)
                    if i is not None:
                        hit = ('reference', r, self.ref_xy_sorted[i]); break
            if hit:
                n.update(ax=float(hit[2][0]), ay=float(hit[2][1]), place=hit[0], place_record=hit[1])
                continue
            vectors = emb.vectors(numbers) if same_plan else [None] * len(numbers)
            for r, w, v in zip(records, numbers, vectors):
                v = v if v is not None else self._sample_vector(w)
                if v is None:
                    continue
                v = np.asarray(v, np.float64)
                norm = np.linalg.norm(v)
                if v.shape == (self.ref_X.shape[1],) and np.isfinite(v).all() and norm > 1e-12:
                    todo.append((nid, r, (v / norm).astype(np.float32)))
                    break
        if todo:
            with self.lock:                          # one batched matrix product over the references
                xy = placement.place_numpy(self.ref_X, self.ref_xy, np.stack([v for _, _, v in todo]))
            for (nid, r, _), p in zip(todo, xy):
                nodes[nid].update(ax=float(p[0]), ay=float(p[1]), place='placed', place_record=r)
        counts = {k: sum(1 for n in nodes.values() if n['place'] == k) for k in ('dot', 'reference', 'placed')}
        return dict(build=self.id, method=self.meta.get('method'), plan_matches=same_plan,
                    placed=sum(counts.values()), total=len(nodes), by=counts,
                    unplaced=[nid for nid, n in nodes.items() if n['place'] is None])


class ScienceAtlases:
    """Loads the requested atlas build once (by id) and keeps it."""
    def __init__(self, root: Path, sample_vectors: Path):
        self.root, self.sample_vectors = Path(root), Path(sample_vectors)
        self.loaded, self.lock = {}, threading.Lock()

    def get(self, build_id):
        if not build_id or not BUILD_ID.match(build_id):
            raise ValueError('invalid atlas build id')
        with self.lock:
            if build_id not in self.loaded:
                path = self.root / build_id
                if not (path / 'meta.json').exists():
                    raise ValueError(f'atlas build {build_id} not found')
                self.loaded = {build_id: ScienceAtlas(path, self.sample_vectors)}   # keep one build in memory
            return self.loaded[build_id]
