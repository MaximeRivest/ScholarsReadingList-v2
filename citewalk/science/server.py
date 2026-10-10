"""Science-atlas pilot viewer.

    cd pipeline/embeddings && uv run python ../../citewalk/science/server.py --port 8806

Serves the static viewer, the frozen build's binary arrays, and three read-only APIs:
  /api/paper?i=       details of display paper i
  /api/similar?i=     its 10 nearest papers in the original 768-D vectors (among the display
                      sample), and how many of them are also among its 10 nearest on the map
  /api/search?q=      a free-text query encoded with DenseOn's query prompt; its 20 most similar
                      display papers and its position under the frozen placement function
Run with the pipeline/embeddings environment (torch + sentence-transformers for /api/search).
"""
from __future__ import annotations

import argparse
import gzip
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import math
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import placement  # noqa: E402

PLAN = Path('/mnt/fast/openalex-embeddings/denseon-english-abstracts')
DATA_FILES = ('meta.json', 'points.f32', 'year.u16', 'field.u8', 'subfield.u16')
CITEWALK = 'http://127.0.0.1:8798'   # citewalk/server.py (citation index + embeddings lookup)


def clean(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


class Atlas:
    def __init__(self, build: Path):
        t = time.time()
        self.build = build
        self.meta = json.loads((build / 'meta.json').read_text())
        self.files = {}
        for name in DATA_FILES:
            raw = (build / name).read_bytes()
            self.files[name] = (raw, gzip.compress(raw, 5))
        self.info = pq.read_table(build / 'info.parquet')
        self.xy = np.fromfile(build / 'points.f32', np.float32).reshape(-1, 2)
        self.X = np.load(build / 'vectors.npy').astype(np.float32)
        ref = np.load(build / 'reference.npz')
        self.ref_X = ref['vectors'].astype(np.float32)
        self.ref_xy = ref['coords'].astype(np.float32)
        if int(ref['k']) != placement.K or float(ref['tau']) != placement.TAU:
            raise ValueError('Build was made with different placement parameters')
        self.encoder = None
        self.encoder_lock = threading.Lock()
        print(f'atlas {build.name}: {len(self.xy):,} papers loaded in {time.time() - t:.0f}s', flush=True)

    def paper(self, i):
        row = self.info.slice(i, 1).to_pylist()[0]
        row = {k: clean(v) for k, v in row.items()}
        row['openalex'] = f"https://openalex.org/W{row['work']}"
        row['work'] = str(row['work'])
        row['i'] = i
        row['x'], row['y'] = map(float, self.xy[i])
        return row

    def similar(self, i, k=10):
        sims = self.X @ self.X[i]
        sims[i] = -2
        hd = np.argpartition(-sims, k)[:k]
        hd = hd[np.argsort(-sims[hd])]
        d2 = ((self.xy - self.xy[i]) ** 2).sum(1)
        d2[i] = np.inf
        lo = np.argpartition(d2, k)[:k]
        # ties count: a true neighbour is kept if it is no farther on the map than the k-th nearest dot
        radius = d2[lo].max()
        kept = int((d2[hd] <= radius * (1 + 1e-6) + 1e-12).sum())
        return {'true': [{'i': int(j), 'sim': round(float(sims[j]), 4), 'title': self.info['title'][int(j)].as_py(),
                          'year': self.info['year'][int(j)].as_py()} for j in hd],
                'on_map': [int(j) for j in lo[np.argsort(d2[lo])]],
                'kept': kept, 'k': k}

    def search(self, text, k=20):
        with self.encoder_lock:
            if self.encoder is None:
                import torch
                from sentence_transformers import SentenceTransformer
                torch.set_num_threads(4)
                emb = json.loads((PLAN / 'manifest.json').read_text())['embedding']
                self.encoder = SentenceTransformer(emb['model_snapshot'], device='cpu', local_files_only=True)
                self.encoder.max_seq_length = emb['max_seq_length']
            q = self.encoder.encode([text], prompt_name='query', normalize_embeddings=True,
                                    convert_to_numpy=True).astype(np.float32)
        sims = self.X @ q[0]
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        x, y = placement.place_numpy(self.ref_X, self.ref_xy, q)[0]
        return {'query': text, 'x': float(x), 'y': float(y),
                'results': [{'i': int(j), 'sim': round(float(sims[j]), 4), 'title': self.info['title'][int(j)].as_py(),
                             'year': self.info['year'][int(j)].as_py()} for j in top]}


def make_handler(atlas: Atlas):
    static = HERE / 'static'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype='application/json', gz=None, cache='no-store'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            if gz is not None and 'gzip' in self.headers.get('Accept-Encoding', ''):
                data = gz
                self.send_header('Content-Encoding', 'gzip')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', cache)
            self.end_headers()
            self.wfile.write(data)

        def walk(self, q):
            from urllib.parse import urlencode
            from urllib.request import urlopen
            from urllib.error import HTTPError, URLError
            keep = {k: q[k] for k in ('id', 'k', 'normalize', 'dups', 'max_refs', 'max_cites') if k in q}
            keep['build'] = atlas.build.name                    # always this atlas's own geometry
            try:
                with urlopen(f'{CITEWALK}/api/walk?{urlencode(keep)}', timeout=120) as r:
                    return self.send(200, r.read())
            except HTTPError as e:
                return self.send(e.code, e.read())
            except (URLError, TimeoutError) as e:
                return self.send(503, {'error': f'The citation-walk service is not reachable ({e}).'})

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            n = len(atlas.xy)
            try:
                if u.path.startswith('/data/'):
                    name = u.path[6:]
                    if name not in atlas.files:
                        return self.send(404, {'error': 'not found'})
                    raw, gz = atlas.files[name]
                    ctype = 'application/json' if name.endswith('.json') else 'application/octet-stream'
                    # build id in the URL makes these immutable
                    return self.send(200, raw, ctype, gz, 'public, max-age=86400' if 'b' in q else 'no-store')
                if u.path in ('/api/paper', '/api/similar'):
                    i = int(q.get('i', -1))
                    if not 0 <= i < n:
                        return self.send(400, {'error': 'bad index'})
                    return self.send(200, atlas.paper(i) if u.path == '/api/paper' else atlas.similar(i))
                if u.path == '/api/walk':        # citation walk, computed by the citewalk server on this build
                    return self.walk(q)
                if u.path == '/api/search':
                    text = (q.get('q') or '').strip()[:500]
                    if not text:
                        return self.send(400, {'error': 'empty query'})
                    return self.send(200, atlas.search(text))
                name = 'index.html' if u.path in ('/', '') else u.path.lstrip('/')
                f = (static / name).resolve()
                if static in f.parents and f.is_file():
                    ctype = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript',
                             '.css': 'text/css'}.get(f.suffix, 'application/octet-stream')
                    return self.send(200, f.read_bytes(), ctype)
                return self.send(404, {'error': 'not found'})
            except Exception as e:  # report, keep serving
                import traceback; traceback.print_exc()
                return self.send(500, {'error': f'{type(e).__name__}: {e}'})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', type=Path)
    ap.add_argument('--root', type=Path, default=Path('/mnt/fast/science-atlas/build'))
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8806)
    args = ap.parse_args()
    # the newest finished build this viewer can place questions on (rule "mean"; "mode" builds are for the site)
    def readable(p):
        if p.name.startswith('.') or p.name.endswith('.tmp') or not (p / 'meta.json').exists():
            return False
        return json.loads((p / 'meta.json').read_text()).get('placement', {}).get('rule', 'mean') == 'mean'
    build = args.build or sorted(p for p in args.root.iterdir() if readable(p))[-1]
    atlas = Atlas(build)
    ThreadingHTTPServer((args.host, args.port), make_handler(atlas)).serve_forever()


if __name__ == '__main__':
    main()
