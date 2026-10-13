"""Build the authorship index beside the citation index: who wrote each paper, in which position.

    uv run --with-requirements citewalk/requirements.txt python citewalk/build_authors.py

Output, in the citation index's own directory (/mnt/fast/citewalk/<generation>/authors/), aligned with its nodes
(node i = position i in ../wn.npy):

  nauth.npy        uint16[N]    authors listed on the paper (distinct author positions, capped at 65,535),
                                identified or not; the denominator for position-based credit
  auth_off.npy     uint64[N+1]  CSR offsets into the two arrays below
  auth.npy         uint32[E]    author (row in author_an.npy) of each identified authorship
  auth_pos.npy     uint16[E]    the author's position on the paper (1 = first), capped at 65,535
  author_an.npy    int64[M]     OpenAlex author number (A-number) of each author, sorted
  name_off.npy     uint64[M+1], names.bin   UTF-8 display names (OpenAlex's disambiguated author entity)
  meta.json

Authors are OpenAlex's disambiguated persons (authorships[].author.id), not name strings: two "Wei Wang"s stay two
people. One authorship per (paper, position): a person listed with several affiliations has several signatures.
Positions OpenAlex could not tie to a person are counted in nauth but have no row.
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CAP = 65535
G = {}


def log(msg):
    print(time.strftime('%H:%M:%S ') + msg, flush=True)


def init_worker(threads, tmp, wn_path):
    os.environ['POLARS_MAX_THREADS'] = str(threads)
    G['tmp'] = Path(tmp)
    G['wn'] = np.load(wn_path, mmap_mode='r')


def lookup(wn, x):
    order = np.argsort(x, kind='stable')
    idx = np.searchsorted(wn, x[order])
    idx_c = np.minimum(idx, len(wn) - 1)
    found = wn[idx_c] == x[order]
    out = np.empty(len(x), np.int64)
    out[order] = np.where(found, idx_c, -1)
    return out


def part(args):
    i, (lcs_path, sig_path, lse_path) = args
    import polars as pl
    lcs = pl.read_parquet(lcs_path, columns=['content_id', 'signature_id'])
    sig = pl.read_parquet(sig_path, columns=['signature_id', 'author_order']).unique('signature_id')
    lse = (pl.read_parquet(lse_path, columns=['signature_id', 'entity_id', 'relation'])
           .filter(pl.col('relation') == 'identifies_person')
           .select('signature_id', pl.col('entity_id').str.slice(10).cast(pl.Int64, strict=False).alias('a'))
           .drop_nulls().unique('signature_id'))
    rows = (lcs.join(sig, on='signature_id', how='inner')
            .select(pl.col('content_id').str.slice(10).cast(pl.Int64).alias('w'), 'signature_id',
                    pl.col('author_order').cast(pl.Int64).alias('o'))
            .drop_nulls(['w', 'o']))
    n = rows.group_by('w').agg(pl.col('o').n_unique().alias('n'))
    who = (rows.join(lse, on='signature_id', how='inner')
           .group_by('w', 'o').agg(pl.col('a').min())            # several signatures, one person per position
           .sort('w', 'o'))
    wn = G['wn']
    node = lookup(wn, who['w'].to_numpy())
    nnode = lookup(wn, n['w'].to_numpy())
    keep, nkeep = node >= 0, nnode >= 0
    np.savez(G['tmp'] / f'{i:05d}.npz',
             node=node[keep].astype(np.uint32), pos=np.minimum(who['o'].to_numpy()[keep], CAP).astype(np.uint16),
             a=who['a'].to_numpy()[keep],
             nnode=nnode[nkeep].astype(np.uint32), n=np.minimum(n['n'].to_numpy()[nkeep], CAP).astype(np.uint16))
    return i, int(keep.sum()), int((~keep).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--citations', type=Path, help='citation index directory (default: /mnt/fast/citewalk/<generation>)')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--limit-files', type=int, help='smoke test: only the first N parts, into authors-firstN/')
    args = ap.parse_args()
    import polars as pl
    from pipeline.tidy.dataset import manifest
    m = manifest()
    cit = args.citations or Path('/mnt/fast/citewalk') / m['generation']
    meta_c = json.loads((cit / 'meta.json').read_text())
    if meta_c['generation'] != m['generation']:
        raise SystemExit(f"citation index {meta_c['generation']} is not the tidy generation {m['generation']}")
    T = m['tables']
    trip = list(zip(T['links_content_signatures']['files'], T['signatures']['files'], T['links_signatures_entities']['files']))
    for a, b, c in trip:   # the three lists must name the same extraction part, in the same order
        if not (Path(a).parent == Path(b).parent == Path(c).parent):
            raise SystemExit(f'manifest lists are not aligned: {a} {b} {c}')
    out = cit / 'authors'
    if args.limit_files:
        trip = trip[:args.limit_files]
        out = cit / f'authors-first{args.limit_files}'
    if (out / 'meta.json').exists():
        log(f'authorship index already built at {out}')
        return
    tmp = out / 'tmp'
    shutil.rmtree(out, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.time()
    wn_path = cit / 'wn.npy'
    N = int(np.load(wn_path, mmap_mode='r').shape[0])
    log(f'{len(trip)} parts, {N:,} nodes')
    E = missing = 0
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker,
                             initargs=(args.threads, tmp, str(wn_path))) as pool:
        for k, (i, e, miss) in enumerate(pool.map(part, enumerate(trip)), 1):
            E += e
            missing += miss
            if k % 100 == 0:
                log(f'  parts {k}/{len(trip)} · {E:,} authorships')
    log(f'authorships: {E:,} ({missing:,} on works missing from the citation index)')

    log('authors')
    uniq = np.unique(np.concatenate([np.unique(np.load(tmp / f'{i:05d}.npz')['a']) for i in range(len(trip))]))
    M = len(uniq)
    np.save(out / 'author_an.npy', uniq)
    log(f'authors: {M:,}')

    log('offsets')
    count = np.zeros(N, np.uint32)
    nauth = np.zeros(N, np.uint16)
    for i in range(len(trip)):
        p = np.load(tmp / f'{i:05d}.npz')
        u, c = np.unique(p['node'], return_counts=True)
        count[u] += c.astype(np.uint32)
        nauth[p['nnode']] = p['n']
    off = np.zeros(N + 1, np.uint64)
    np.cumsum(count, out=off[1:])
    assert int(off[-1]) == E
    np.save(out / 'auth_off.npy', off)
    np.save(out / 'nauth.npy', nauth)
    del count, nauth

    log('fill')
    auth = np.empty(E, np.uint32)
    pos = np.empty(E, np.uint16)
    for i in range(len(trip)):
        p = np.load(tmp / f'{i:05d}.npz')
        node = p['node']                          # sorted by (work number, position) = by node, then position
        if len(node) == 0:
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(node)) + 1]
        counts = np.diff(np.r_[starts, len(node)])
        rank = np.arange(len(node)) - np.repeat(starts, counts)
        dest = off[node] + rank.astype(np.uint64)
        auth[dest] = np.searchsorted(uniq, p['a']).astype(np.uint32)
        pos[dest] = p['pos']
        if (i + 1) % 200 == 0:
            log(f'  fill {i + 1}/{len(trip)}')
    np.save(out / 'auth.npy', auth)
    np.save(out / 'auth_pos.npy', pos)
    del auth, pos

    log('names')
    frames = []
    for f in T['entities']['files']:
        d = (pl.read_parquet(f, columns=['entity_id', 'entity_type', 'name'])
             .filter(pl.col('entity_type') == 'person')
             .select(pl.col('entity_id').str.slice(10).cast(pl.Int64, strict=False).alias('a'),
                     pl.col('name').fill_null('').str.slice(0, 200))
             .drop_nulls('a'))
        frames.append(d)
    ents = pl.concat(frames).unique('a', keep='first')
    del frames
    full = (pl.DataFrame({'a': uniq}).with_row_index('r')
            .join(ents, on='a', how='left').sort('r'))
    del ents
    found = full['name'].is_not_null().to_numpy()
    arr = full['name'].fill_null('').cast(pl.String).to_arrow()
    if hasattr(arr, 'combine_chunks'):
        arr = arr.combine_chunks()
    import pyarrow as pa
    arr = arr.cast(pa.large_string())
    bufs = arr.buffers()
    offs = np.frombuffer(bufs[1], np.int64, count=len(arr) + 1, offset=arr.offset * 8)
    data = np.frombuffer(bufs[2], np.uint8) if bufs[2] is not None else np.zeros(0, np.uint8)
    noff = (offs - offs[0]).astype(np.uint64)
    np.save(out / 'name_off.npy', noff)
    with open(out / 'names.bin', 'wb') as fh:
        fh.write(data[offs[0]:offs[-1]].tobytes())
    del full, arr, data

    meta = dict(generation=meta_c['generation'], nodes=N, authorships=E, authors=M,
                authors_with_name=int(found.sum()), parts=len(trip),
                works_missing_from_citation_index=missing, built_at=time.time(),
                source='tidy links_content_signatures + signatures + links_signatures_entities (identifies_person); '
                       'names from entities (person)')
    shutil.rmtree(tmp)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    log(f"done: {E:,} authorships of {M:,} authors ({found.sum():,} named) in {(time.time() - t0) / 60:.0f} min")


if __name__ == '__main__':
    main()
