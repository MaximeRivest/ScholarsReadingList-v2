"""Frozen local semantic maps, persisted separately from the embedding backfill.

A map fits PCA ONCE to genuine normalized vectors in its starting neighborhood.
All later papers use those same axes. No citation forces, clustering, jitter,
collision movement, or per-query normalization changes the coordinates.

This is a diagnostic local chart, NOT a global atlas. The two-dimensional loss
is reported using explained anchor variance and exact-neighbor preservation.
"""
from collections import OrderedDict
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import threading
import time

import numpy as np

VERSION = 'frozen-local-pca-v1'
MAX_ANCHORS = 256
MAX_QUALITY = 512


class AtlasError(ValueError):
    pass


def unit_vector(value, dimensions=None):
    if value is None:
        return None
    v = np.asarray(value, dtype=np.float64)
    if v.ndim != 1 or (dimensions is not None and len(v) != dimensions) or not np.isfinite(v).all():
        return None
    norm = np.linalg.norm(v)
    return v / norm if np.isfinite(norm) and norm > 1e-12 else None


def fit_projection(vectors):
    """Return a mean and two orthonormal axes; rank-one data stays a line, never jittered."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2 or len(v) < 3 or not np.isfinite(v).all():
        raise AtlasError('At least three valid vectors are needed to start a local map.')
    mean = v.mean(axis=0)
    centered = v - mean
    # The neighborhood is much smaller than 768 dimensions. Its Gram matrix
    # gives the same principal axes without a 768x768 eigendecomposition.
    eigenvalues, left = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, left = eigenvalues[order], left[:, order]
    total = float(np.square(centered).sum())
    if total <= 1e-12:
        raise AtlasError('The available vectors are identical; there is no semantic variation to map.')
    axes = np.zeros((v.shape[1], 2), dtype=np.float64)
    for i in range(min(2, len(eigenvalues))):
        if eigenvalues[i] <= total * 1e-12:
            continue
        axis = centered.T @ left[:, i] / np.sqrt(eigenvalues[i])
        # Fix arbitrary eigenvector signs; persisted axes are authoritative on reload.
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        axes[:, i] = axis
    captured = float(np.square(centered @ axes).sum() / total)
    return mean, axes, min(1.0, max(0.0, captured))


def publication_key(node):
    # Same local title grouping already used by Graph.analyze. This is not a
    # new identity resolver and does not prove two source records are identical.
    title = re.sub(r'[\W_]+', ' ', (node.get('title') or '').casefold()).strip()
    return 'title:' + title if len(title) >= 20 else 'record:' + node['id']


def _point_metadata(node, source):
    keep = ('id', 'title', 'year', 'cited_by', 'type', 'topic', 'subfield')
    return {**{k: node.get(k) for k in keep}, 'vector_source_id': source}


class LocalMap:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path / 'projection.npz', allow_pickle=False) as z:
            self.mean, self.axes = z['mean'].copy(), z['axes'].copy()
            self.meta = json.loads(str(z['metadata'].item()))

    def project(self, vectors):
        return (np.asarray(vectors, dtype=np.float64) - self.mean) @ self.axes

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path / 'points.sqlite', timeout=15)
        try:
            with db:
                db.execute('CREATE TABLE IF NOT EXISTS points (key TEXT PRIMARY KEY, x REAL NOT NULL, y REAL NOT NULL, vector BLOB NOT NULL, metadata TEXT NOT NULL)')
                yield db
        finally:
            db.close()

    def points(self, keys):
        out = {}
        with self.connect() as db:
            for start in range(0, len(keys), 400):
                chunk = keys[start:start + 400]
                if not chunk:
                    continue
                for key, x, y, vector, meta in db.execute(
                        'SELECT key,x,y,vector,metadata FROM points WHERE key IN (' + ','.join('?' for _ in chunk) + ')', chunk):
                    out[key] = dict(x=x, y=y, vector=np.frombuffer(vector, dtype='<f8').copy(), metadata=json.loads(meta))
        return out

    def add(self, observations):
        """First genuine vector fixes a title-group's position; later aliases never move it."""
        with self.connect() as db:
            for key, vector, meta in observations:
                xy = self.project(vector)
                db.execute('INSERT OR IGNORE INTO points VALUES (?,?,?,?,?)',
                           (key, float(xy[0]), float(xy[1]), vector.astype('<f8').tobytes(), json.dumps(meta, ensure_ascii=False)))


class AtlasStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cache = OrderedDict()

    def load(self, identifier):
        if not re.fullmatch(r'[0-9a-f]{64}', identifier or ''):
            raise AtlasError('Invalid saved-map ID. Start a new map.')
        if identifier in self.cache:
            self.cache.move_to_end(identifier)
            return self.cache[identifier]
        path = self.root / identifier
        if not (path / 'projection.npz').exists():
            raise AtlasError('This saved map is unavailable. Start a new map.')
        chart = LocalMap(path)
        self.cache[identifier] = chart
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        return chart

    def create(self, observed, context):
        # Stable bounded anchor sample for unusually large tie sets. Normal views
        # use every available vector. Sampling does not move or exclude later points.
        anchors = sorted(observed, key=lambda o: sha256(o[0].encode()).hexdigest())[:MAX_ANCHORS]
        anchors = sorted(anchors, key=lambda o: o[0])
        vectors = np.stack([v for _, v, _ in anchors])
        identity = sha256(json.dumps([VERSION, context, [k for k, _, _ in anchors]], sort_keys=True).encode()
                          + vectors.astype('<f8').tobytes()).hexdigest()
        path = self.root / identity
        path.mkdir(exist_ok=True)
        with (path / 'projection.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (path / 'projection.npz').exists():
                mean, axes, captured = fit_projection(vectors)
                meta = dict(context, id=identity, algorithm=VERSION, anchor_keys=[k for k, _, _ in anchors],
                            anchor_count=len(anchors), anchor_variance_captured=captured,
                            dimensions=vectors.shape[1], created_at=time.time(), numpy_version=np.__version__)
                tmp = path / 'projection.tmp.npz'
                np.savez(tmp, mean=mean, axes=axes, metadata=np.asarray(json.dumps(meta)))
                tmp.replace(path / 'projection.npz')
        chart = self.load(identity)
        chart.add(observed)
        return chart

    def annotate(self, graph, embeddings, result, identifier=None):
        if 'nodes' not in result:
            return
        with self.lock:
            self._annotate(graph, embeddings, result, identifier)

    def _annotate(self, graph, embeddings, result, identifier):
        start = time.perf_counter()
        nodes = result['nodes']
        for n in nodes.values():
            i = graph.node_of_work(int(n['id'][1:]))
            topic = graph.topic_of(i) if i is not None else None
            n['topic'] = topic['name'] if topic else None
            n['subfield'] = topic['subfield'] if topic else None
            n.update(atlas_key=publication_key(n), mx=None, my=None, sem=None, region=None,
                     map_status='missing_vector', semantic_neighbors=[])
        generation = graph.meta.get('generation', 'unknown')
        plan = getattr(embeddings, 'plan_id', None) if embeddings else None
        chart = self.load(identifier) if identifier else None
        if chart and (chart.meta['generation'] != generation or (plan and chart.meta['embedding_plan'] != plan)):
            raise AtlasError('This map belongs to a different citation or embedding snapshot. Start a new map.')
        unique = {n['atlas_key']: n for n in sorted(nodes.values(), key=lambda n: n['id'])}
        cached = chart.points(list(unique)) if chart else {}
        observed = []
        for key, n in unique.items():
            if key in cached:
                continue
            # Fetch only actual published embeddings. Missing vectors remain missing;
            # no centroid/citation-position fallback is used.
            records = [n['id']] + sorted(set(n.get('records', [])) - {n['id']})
            values = embeddings.vectors([int(r[1:]) for r in records]) if embeddings else [None] * len(records)
            for rid, raw in zip(records, values):
                vector = unit_vector(raw, chart.meta['dimensions'] if chart else getattr(embeddings, 'dimensions', None))
                if vector is not None:
                    observed.append((key, vector, _point_metadata(n, rid)))
                    break
                if raw is not None:
                    n['map_status'] = 'invalid_vector'
        context = dict(generation=generation, embedding_plan=plan, seed_id=result['focus'],
                       seed_title=nodes[result['focus']]['title'])
        reason = None
        if not chart:
            if len(observed) < 3:
                reason = 'At least three papers with vectors are needed to start a local map.'
            else:
                try:
                    chart = self.create(observed, context)
                except AtlasError as e:
                    reason = str(e)
        else:
            chart.add(observed)
        result['regions'] = []  # no invented territories in the clean test
        if not chart:
            valid_keys = {key for key, _, _ in observed}
            for n in nodes.values():
                if n['atlas_key'] in valid_keys:
                    n['map_status'] = 'no_projection'
            result['atlas'] = dict(id=None, status='unavailable', reason=reason,
                                   with_vector=len(observed), plotted=0, total=len(nodes), unplaced=list(nodes), landmarks=[])
            result['stats']['semantic'] = dict(method='No stable projection yet', with_vector=len(observed), nodes=len(nodes),
                                              model='DenseOn', indexed=embeddings.stats()['vectors'] if embeddings else 0)
            return
        points = chart.points(list(unique))
        plotted = []
        for nid, n in nodes.items():
            p = points.get(n['atlas_key'])
            if p:
                n.update(mx=p['x'], my=p['y'], map_status='placed', vector_source_id=p['metadata']['vector_source_id'])
                plotted.append(nid)
        x = points.get(nodes[result['focus']]['atlas_key'])
        if x:
            for nid in plotted:
                v = points[nodes[nid]['atlas_key']]['vector']
                nodes[nid]['sem'] = float(np.clip(1 - v @ x['vector'], 0, 2))
        landmarks = []
        for key, p in chart.points(chart.meta['anchor_keys']).items():
            landmarks.append(dict(p['metadata'], atlas_key=key, mx=p['x'], my=p['y'], map_status='placed', column='landmark'))
        # A diagnostic, not a promise that PCA preserves all neighborhoods. Exact
        # neighbors are computed in 768-D, separately from the drawing coordinates.
        measured = sorted(plotted, key=lambda nid: nodes[nid]['atlas_key'])[:MAX_QUALITY]
        quality = dict(evaluated=len(measured), population=len(plotted), k=0, neighbor_preservation=None)
        if len(measured) > 1:
            vectors = np.stack([points[nodes[i]['atlas_key']]['vector'] for i in measured])
            xy = np.asarray([[nodes[i]['mx'], nodes[i]['my']] for i in measured])
            exact = np.clip(1 - vectors @ vectors.T, 0, 2)
            projected = np.square(xy[:, None, :] - xy[None, :, :]).sum(axis=-1)
            np.fill_diagonal(exact, np.inf); np.fill_diagonal(projected, np.inf)
            k = min(5, len(measured) - 1)
            neighbors = np.argsort(exact, axis=1, kind='stable')[:, :k]
            displayed = np.argsort(projected, axis=1, kind='stable')[:, :k]
            preservation = np.mean([len(set(a) & set(b)) / k for a, b in zip(neighbors, displayed)])
            quality.update(k=k, neighbor_preservation=float(preservation))
            for j, nid in enumerate(measured):
                nodes[nid]['semantic_neighbors'] = [dict(id=measured[z], similarity=float(1-exact[j, z])) for z in neighbors[j]]
        result['atlas'] = dict(id=chart.meta['id'], status='ready', algorithm=VERSION,
                               seed_id=chart.meta['seed_id'], seed_title=chart.meta['seed_title'],
                               embedding_plan=chart.meta['embedding_plan'], anchor_count=chart.meta['anchor_count'],
                               anchor_variance_captured=chart.meta['anchor_variance_captured'],
                               with_vector=len(plotted), plotted=len(plotted), total=len(nodes),
                               unplaced=[i for i in nodes if nodes[i]['map_status'] != 'placed'],
                               landmarks=landmarks, quality=quality,
                               seconds=round(time.perf_counter()-start, 3))
        result['stats']['semantic'] = dict(method='Frozen local PCA; cosine distances use the original vectors',
                                           with_vector=len(plotted), nodes=len(nodes), model='DenseOn',
                                           indexed=embeddings.stats()['vectors'] if embeddings else 0)
