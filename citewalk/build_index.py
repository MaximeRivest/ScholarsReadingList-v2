"""Build a compact, memory-mappable citation index from the published tidy tables.

    uv run --with-requirements citewalk/requirements.txt python citewalk/build_index.py

Output (one directory per tidy generation, e.g. /mnt/fast/citewalk/<generation>/):
  wn.npy          int64[N]   OpenAlex work number of each node, sorted (node id = position)
  year.npy        int16[N]   publication year (0 = unknown)
  cites.npy       int32[N]   OpenAlex cited_by_count (-1 = unknown)
  nrefs.npy       int32[N]   OpenAlex referenced_works_count (-1 = unknown)
  kind.npy        uint8[N]   index into meta.json "types" (255 = unknown)
  complete.npy    bool[N]    is_complete
  title_row.npy   uint32[N]  row of the node in titles.bin's file order
  title_off.npy   uint64[N+1], titles.bin   UTF-8 titles (capped at 600 bytes, cut on a char boundary)
  doi_hash.npy    uint64[D], doi_node.npy uint32[D]   sorted wyhash of normalized DOI -> node
  fwd_off.npy     uint64[N+1], fwd.npy uint32[E]   references of each node (CSR)
  bwd_off.npy     uint64[N+1], bwd.npy uint32[E]   papers citing each node (CSR)
  meta.json

Only citations whose target is in the snapshot are indexed (the rest have no
title or metadata to show); their count is recorded in meta.json.
"""
import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TITLE_CAP = 600
DOI_PREFIX = r'^(https?://(dx\.)?doi\.org/|doi:)'

G = {}  # per-worker globals


def log(msg):
    print(time.strftime('%H:%M:%S ') + msg, flush=True)


def normalized_doi(expr):
    return expr.str.strip_chars().str.to_lowercase().str.replace(DOI_PREFIX, '')


def init_worker(threads, tmp, types, wn_path):
    os.environ['POLARS_MAX_THREADS'] = str(threads)
    G['tmp'], G['types'] = Path(tmp), types
    if wn_path:
        G['wn'] = np.load(wn_path, mmap_mode='r')


def nodes_part(args):
    i, path = args
    import polars as pl
    import polars_hash  # noqa: F401  (registers .nchash)
    df = pl.read_parquet(path, columns=['content_id', 'title', 'year', 'citations', 'references_count',
                                        'type', 'is_complete', 'doi'])
    types = {t: k for k, t in enumerate(G['types'])}
    # exact titles; only titles over TITLE_CAP bytes are cut, on a character boundary
    encoded = [t.encode('utf-8') for t in df['title'].fill_null('').to_list()]
    encoded = [e if len(e) <= TITLE_CAP else e[:TITLE_CAP].decode('utf-8', 'ignore').encode('utf-8') for e in encoded]
    blob = b''.join(encoded)
    lengths = np.fromiter((len(e) for e in encoded), np.uint32, len(encoded))
    doi = df.select(normalized_doi(pl.col('doi')).alias('d'))['d']
    has = doi.is_not_null() & (doi != '')
    dh = pl.DataFrame({'d': doi.filter(has)}).select(pl.col('d').nchash.wyhash())['d'].to_numpy().astype(np.uint64)
    (G['tmp'] / 'titles').mkdir(exist_ok=True)
    (G['tmp'] / 'titles' / f'{i:05d}.bin').write_bytes(blob)
    np.savez(G['tmp'] / 'nodes' / f'{i:05d}.npz',
             wn=df['content_id'].str.slice(10).cast(pl.Int64).to_numpy(),
             year=df['year'].fill_null(0).clip(-32768, 32767).cast(pl.Int16).to_numpy(),
             cites=df['citations'].fill_null(-1).clip(-1, 2**31 - 1).cast(pl.Int32).to_numpy(),
             nrefs=df['references_count'].fill_null(-1).clip(-1, 2**31 - 1).cast(pl.Int32).to_numpy(),
             kind=df['type'].replace_strict(types, default=255, return_dtype=pl.UInt8).to_numpy(),
             complete=df['is_complete'].to_numpy(),
             lengths=lengths, doi_hash=dh, doi_row=np.flatnonzero(has.to_numpy()).astype(np.uint32))
    return i, df.height


def edges_part(args):
    i, path = args
    import polars as pl
    e = pl.read_parquet(path, columns=['from_content_id', 'to_content_id']).select(
        pl.col('from_content_id').str.slice(10).cast(pl.Int64).alias('s'),
        pl.col('to_content_id').str.slice(10).cast(pl.Int64, strict=False).alias('t'))
    wn = G['wn']
    def lookup(x):
        order = np.argsort(x, kind='stable')
        idx = np.searchsorted(wn, x[order])
        idx_c = np.minimum(idx, len(wn) - 1)
        found = wn[idx_c] == x[order]
        out = np.empty(len(x), np.int64)
        out[order] = np.where(found, idx_c, -1)
        return out
    s = lookup(e['s'].to_numpy())
    t = lookup(e['t'].fill_null(-1).to_numpy())
    if (s < 0).any():
        raise RuntimeError(f'{path}: citing work not in content table')
    keep = t >= 0
    s, t = s[keep].astype(np.uint32), t[keep].astype(np.uint32)
    order = np.argsort(s, kind='stable')
    np.savez(G['tmp'] / 'edges' / f'{i:05d}.npz', src=s[order], dst=t[order])
    return i, e.height, int(keep.sum())


def group_ranks(sorted_keys):
    """For a sorted array: (unique keys, counts, rank of each element within its key)."""
    n = len(sorted_keys)
    if n == 0:
        return sorted_keys, np.zeros(0, np.int64), np.zeros(0, np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    counts = np.diff(np.r_[starts, n])
    ranks = np.arange(n) - np.repeat(starts, counts)
    return sorted_keys[starts], counts, ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('/mnt/fast/citewalk'))
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--limit-files', type=int, help='smoke test: only the first N content/citation files')
    args = ap.parse_args()
    os.environ['POLARS_MAX_THREADS'] = str(args.workers * args.threads)
    import polars as pl
    from pipeline.tidy.dataset import manifest
    m = manifest()
    gen = m['generation']
    content = m['tables']['content']['files']
    citations = m['tables']['links_content_content']['files']
    if args.limit_files:
        content, citations = content[:args.limit_files], citations[:args.limit_files]
        gen += f'-first{args.limit_files}'
    out = args.out / gen
    if (out / 'meta.json').exists():
        log(f'index for {gen} already built at {out}')
        return
    tmp = out / 'tmp'
    t0 = time.time()
    nodes_done, edges_done = tmp / 'nodes.done', tmp / 'edges.done'
    if not nodes_done.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        for d in ('nodes', 'titles', 'edges'):
            (tmp / d).mkdir(parents=True)

    if not nodes_done.exists():
        build_nodes(args, out, tmp, content)
        nodes_done.write_text('ok')
    if not edges_done.exists():
        shutil.rmtree(tmp / 'edges', ignore_errors=True); (tmp / 'edges').mkdir()
        total, internal = build_edge_chunks(args, out, tmp, citations)
        edges_done.write_text(json.dumps(dict(total=total, internal=internal)))
    counts = json.loads(edges_done.read_text())
    build_adjacency(out, tmp, citations, counts['internal'])
    meta = dict(generation=gen, nodes=int(np.load(out / 'wn.npy', mmap_mode='r').shape[0]), edges=counts['internal'],
                citation_links_total=counts['total'], citation_links_outside_snapshot=counts['total'] - counts['internal'],
                types=json.loads((tmp / 'types.json').read_text()), title_cap_bytes=TITLE_CAP,
                built_at=time.time())
    shutil.rmtree(tmp)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    log(f"done: {meta['nodes']:,} nodes, {meta['edges']:,} edges ({meta['citation_links_outside_snapshot']:,} outside the snapshot skipped) in {(time.time() - t0) / 60:.0f} min")


def build_nodes(args, out, tmp, content):
    import polars as pl
    t0 = time.time()
    log('types')
    types = sorted(pl.scan_parquet(content).select(pl.col('type').drop_nulls().unique()).collect()['type'].to_list())
    (tmp / 'types.json').write_text(json.dumps(types))
    log(f'nodes: {len(content)} files')
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker, initargs=(args.threads, tmp, types, None)) as pool:
        for k, (i, n) in enumerate(pool.map(nodes_part, enumerate(content)), 1):
            if k % 200 == 0:
                log(f'  nodes {k}/{len(content)}')
    parts = [np.load(tmp / 'nodes' / f'{i:05d}.npz') for i in range(len(content))]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    wn = cat('wn')
    N = len(wn)
    order = np.argsort(wn, kind='stable')
    wn_sorted = wn[order]
    if (np.diff(wn_sorted) == 0).any():
        raise RuntimeError('duplicate content ids')
    np.save(out / 'wn.npy', wn_sorted)
    for k in ('year', 'cites', 'nrefs', 'kind', 'complete'):
        np.save(out / f'{k}.npy', cat(k)[order])
    np.save(out / 'title_row.npy', order.astype(np.uint32))
    lengths = cat('lengths')
    off = np.zeros(N + 1, np.uint64)
    np.cumsum(lengths, out=off[1:])
    np.save(out / 'title_off.npy', off)
    with open(out / 'titles.bin', 'wb') as f:
        for i in range(len(content)):
            with open(tmp / 'titles' / f'{i:05d}.bin', 'rb') as g:
                shutil.copyfileobj(g, f, 1 << 24)
    node_of_row = np.empty(N, np.uint32)
    node_of_row[order] = np.arange(N, dtype=np.uint32)
    row_base = np.cumsum([0] + [len(p['wn']) for p in parts])[:-1]
    dh = np.concatenate([p['doi_hash'] for p in parts])
    drow = np.concatenate([p['doi_row'].astype(np.int64) + b for p, b in zip(parts, row_base)])
    o = np.argsort(dh, kind='stable')
    np.save(out / 'doi_hash.npy', dh[o])
    np.save(out / 'doi_node.npy', node_of_row[drow[o]])
    del parts, wn, order, node_of_row, dh, drow, o, lengths, off
    log(f'nodes done: {N:,} ({time.time() - t0:.0f}s)')


def build_edge_chunks(args, out, tmp, citations):
    types = json.loads((tmp / 'types.json').read_text())
    log(f'edges: {len(citations)} files')
    total = internal = 0
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker,
                             initargs=(args.threads, tmp, types, str(out / 'wn.npy'))) as pool:
        for k, (i, n_all, n_in) in enumerate(pool.map(edges_part, enumerate(citations)), 1):
            total += n_all
            internal += n_in
            if k % 200 == 0:
                log(f'  edges {k}/{len(citations)} · {internal:,} internal of {total:,}')
    return total, internal


def build_adjacency(out, tmp, citations, E):
    N = int(np.load(out / 'wn.npy', mmap_mode='r').shape[0])
    log('degrees')
    outdeg = np.zeros(N, np.uint32)
    indeg = np.zeros(N, np.uint32)
    for i in range(len(citations)):
        p = np.load(tmp / 'edges' / f'{i:05d}.npz')
        u, c, _ = group_ranks(p['src'])
        outdeg[u] += c.astype(np.uint32)
        u, c = np.unique(p['dst'], return_counts=True)
        indeg[u] += c.astype(np.uint32)
    fwd_off = np.zeros(N + 1, np.uint64); np.cumsum(outdeg, out=fwd_off[1:])
    bwd_off = np.zeros(N + 1, np.uint64); np.cumsum(indeg, out=bwd_off[1:])
    assert int(fwd_off[-1]) == E == int(bwd_off[-1])
    np.save(out / 'fwd_off.npy', fwd_off)
    np.save(out / 'bwd_off.npy', bwd_off)
    del outdeg, indeg
    log('adjacency')
    # Fill in RAM, write once: random writes into a memory-mapped file made the kernel
    # write the same pages back to disk over and over (hundreds of GB for 25 GB).
    fwd = np.empty(E, np.uint32)
    bwd = np.empty(E, np.uint32)
    cursor = bwd_off[:-1].copy()
    for i in range(len(citations)):
        p = np.load(tmp / 'edges' / f'{i:05d}.npz')
        src, dst = p['src'], p['dst']            # sorted by src; each source lives in one file
        u, c, r = group_ranks(src)
        fwd[(fwd_off[src] + r.astype(np.uint64))] = dst
        o = np.argsort(dst, kind='stable')
        u, c, r = group_ranks(dst[o])
        bwd[(cursor[dst[o]] + r.astype(np.uint64))] = src[o]
        cursor[u] += c.astype(np.uint64)
        if (i + 1) % 200 == 0:
            log(f'  adjacency {i + 1}/{len(citations)}')
    assert (cursor == bwd_off[1:]).all()
    del cursor
    for name, arr in (('fwd', fwd), ('bwd', bwd)):
        np.save(out / f'{name}.npy.tmp.npy', arr)
        os.replace(out / f'{name}.npy.tmp.npy', out / f'{name}.npy')
    log('adjacency written')


if __name__ == '__main__':
    main()
