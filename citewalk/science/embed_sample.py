"""Step 2 of the science-atlas pilot: one DenseOn vector for every sampled work.

    cd pipeline/embeddings && uv run python ../../citewalk/science/embed_sample.py

Vectors already committed by the English backfill are copied from its shards. The rest
(mostly works published before the backfill's current date) are encoded here with the
backfill's own DenseEncoder and pinned manifest settings (model revision, prompt, 512-token
cap, BF16 inference, float32 normalization, float16 storage), so both sources share one
vector space. A check re-encodes works the backfill already has and compares them.

Resumable: encoded chunks are written atomically under vectors/chunks/.
Output (all aligned, sorted by work number):
  /mnt/fast/science-atlas/vectors/work.npy     uint64[N]
  /mnt/fast/science-atlas/vectors/vectors.npy  float16[N,768]
  /mnt/fast/science-atlas/vectors/source.npy   uint8[N]  0 = backfill shard, 1 = encoded here
  /mnt/fast/science-atlas/vectors/check.json   agreement of the two sources
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pipeline' / 'embeddings'))
from embed_content import DenseEncoder, read_manifest  # noqa: E402

PLAN = Path('/mnt/fast/openalex-embeddings/denseon-english-abstracts')
SAMPLE = Path('/mnt/fast/science-atlas/sample')
OUT = Path('/mnt/fast/science-atlas/vectors')
CHUNK = 20000
CHECK_ROWS = 2000


def text_of(title, abstract):
    return f"{title or ''} {abstract}".strip()   # identical to embed_content.records


def backfill_index(doc):
    """Sorted work IDs of committed shards -> (shard path, row). Published part-* only."""
    paths, ids, loc = [], [], []
    for k, c in enumerate(sorted(PLAN.glob('shards/year=*/rg-*/part-*/commit.json'))):
        commit = json.loads(c.read_text())
        if commit['plan_id'] != doc['plan_id']:
            raise ValueError(f'Wrong plan in {c}')
        w = np.load(c.parent / 'work_ids.npy')
        paths.append(c.parent)
        ids.append(w)
        loc.append(np.stack([np.full(len(w), k, np.int64), np.arange(len(w), dtype=np.int64)], 1))
    ids, loc = np.concatenate(ids), np.concatenate(loc)
    o = np.argsort(ids, kind='stable')
    return paths, ids[o], loc[o]


def read_backfill(paths, loc, rows):
    """vectors and text hashes for the given index rows, reading each shard once."""
    vec = np.empty((len(rows), 768), np.float16)
    hashes = np.empty((len(rows), 16), np.uint8)
    order = np.lexsort((loc[rows, 1], loc[rows, 0]))
    shard_of = loc[rows, 0][order]
    starts = np.flatnonzero(np.r_[True, shard_of[1:] != shard_of[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        sel = order[a:b]
        p = paths[shard_of[a]]
        v = np.load(p / 'vectors.npy', mmap_mode='r')
        h = np.load(p / 'text_hashes.npy', mmap_mode='r')
        r = loc[rows[sel], 1]
        vec[sel] = v[r]
        hashes[sel] = h[r]
    return vec, hashes


GPU = os.environ.get('ATLAS_GPU', '1')                 # physical index; the backfill's GPU by default
ALLOCATION_GIB = float(os.environ.get('ATLAS_GPU_GIB', '1.0'))  # hard cap for this process's tensors
MIN_FREE_MIB = int(os.environ.get('ATLAS_MIN_FREE_MIB', '1800'))


def gpu_slot():
    """UUID of the chosen GPU, refusing to start without headroom (never evicts anyone)."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    gpus = {i: (u, int(f)) for i, u, f in (line.split(', ') for line in out.strip().splitlines())}
    uuid, free = gpus[GPU]
    if free < MIN_FREE_MIB:
        raise SystemExit(f'GPU {GPU} has only {free} MiB free (< {MIN_FREE_MIB}); not starting')
    return uuid


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_slot()
    doc = read_manifest(PLAN)
    t0 = time.time()
    table = pq.read_table(sorted(SAMPLE.glob('year=*.parquet')), columns=['work', 'title', 'abstract'])
    works = table['work'].to_numpy()
    o = np.argsort(works)
    works = works[o]
    if np.any(works[1:] == works[:-1]):
        raise ValueError('Duplicate work IDs in sample')
    titles = table['title'].take(o).to_pylist()
    abstracts = table['abstract'].take(o).to_pylist()
    print(f'{len(works):,} sampled works ({time.time() - t0:.0f}s)', flush=True)

    paths, ids, loc = backfill_index(doc)
    j = np.searchsorted(ids, works).clip(0, len(ids) - 1)
    have = ids[j] == works
    print(f'{have.sum():,} already in the backfill, {(~have).sum():,} to encode ({time.time() - t0:.0f}s)', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    chunks = OUT / 'chunks'
    chunks.mkdir(exist_ok=True)
    todo = np.flatnonzero(~have)
    encoder = None

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = DenseEncoder(doc['embedding'], 'cuda:0', int(os.environ.get('ATLAS_BATCH', 32)),
                                   cpu_threads=6, allocation_gib=ALLOCATION_GIB)
        return encoder

    # Agreement check: re-encode works the backfill already has (spread over the whole sample).
    check_file = OUT / 'check.json'
    if not check_file.exists():
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(np.flatnonzero(have), min(CHECK_ROWS, int(have.sum())), replace=False))
        ref, ref_hash = read_backfill(paths, loc, j[pick])
        texts = [text_of(titles[i], abstracts[i]) for i in pick]
        mine_hash = np.frombuffer(b''.join(hashlib.blake2b(t.encode(), digest_size=16).digest() for t in texts),
                                  np.uint8).reshape(-1, 16)
        mine = get_encoder().encode(texts)
        cos = (mine.astype(np.float32) * ref.astype(np.float32)).sum(1)
        same_text = (mine_hash == ref_hash).all(1)
        check = {'rows': int(len(pick)), 'same_input_text': int(same_text.sum()),
                 'cosine_min': float(cos.min()), 'cosine_p01': float(np.quantile(cos, .01)),
                 'cosine_median': float(np.median(cos)), 'batch_size': encoder.batch_size}
        print('check', check, flush=True)
        if check['same_input_text'] != check['rows'] or check['cosine_p01'] < 0.995:
            raise SystemExit('Re-encoded vectors disagree with the backfill; not mixing the two sources')
        check_file.write_text(json.dumps(check, indent=1))

    for c, start in enumerate(range(0, len(todo), CHUNK)):
        f = chunks / f'{c:05d}.npz'
        sel = todo[start:start + CHUNK]
        if f.exists():
            with np.load(f) as z:
                if np.array_equal(z['work'], works[sel]):
                    continue
            raise ValueError(f'{f} does not match the current sample; delete vectors/ to restart')
        t = time.time()
        vec = get_encoder().encode([text_of(titles[i], abstracts[i]) for i in sel])
        tmp = chunks / f'.{c:05d}.tmp.npz'
        np.savez(tmp, work=works[sel], vectors=vec)
        os.replace(tmp, f)
        print(f'chunk {c + 1}/{-(-len(todo) // CHUNK)}: {len(sel) / (time.time() - t):.0f} rows/s', flush=True)

    vectors = np.empty((len(works), 768), np.float16)
    source = np.zeros(len(works), np.uint8)
    vectors[have], _ = read_backfill(paths, loc, j[have])
    for c, start in enumerate(range(0, len(todo), CHUNK)):
        with np.load(chunks / f'{c:05d}.npz') as z:
            sel = todo[start:start + CHUNK]
            assert np.array_equal(z['work'], works[sel])
            vectors[sel] = z['vectors']
            source[sel] = 1
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    if not np.all(np.abs(norms - 1) < 1e-2):
        raise ValueError('Unnormalized vector in the assembled sample')
    for name, a in (('work', works), ('vectors', vectors), ('source', source)):
        np.save(OUT / f'.{name}.tmp.npy', a)
        os.replace(OUT / f'.{name}.tmp.npy', OUT / f'{name}.npy')
    print(f'done: {len(works):,} vectors, {int(source.sum()):,} encoded here ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
