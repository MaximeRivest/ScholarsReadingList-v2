"""Build the source index beside the citation index: where each paper was published (its journal, conference,
repository or book series), each source's papers, and per-source figures.

    uv run --with-requirements citewalk/requirements.txt python citewalk/build_sources.py

Output, in /mnt/fast/citewalk/<generation>/sources/, aligned with the citation index's nodes (node i = position i in
../wn.npy):

  src_in.npy     int64[K]     OpenAlex source number (S-number) of each source, sorted
  name_off.npy, names.bin     display names;  issn_off.npy, issns.bin  its ISSN-L ('' when none)
  kind.npy       uint8[K]     index into meta.json's kinds (journal, repository, conference, book series, ...)
  pub.npy        int32[K]     row of its publisher in pub_names (-1: none known)
  pub_name_off.npy, pub_names.bin, pub_ids.npy (P- or I-number as 'P4310320595')
  w_src.npy      int32[N]     each paper's source row (-1: none): its primary location's source, as OpenAlex marks it
  s_off.npy      uint64[K+1]  CSR: the papers of each source, oldest first
  s_node.npy     uint32[E]      nodes
  oa.npy         bool[N]      the paper is open access (OpenAlex's is_oa)
  s_works.npy, s_cites.npy, s_h.npy, s_year0.npy, s_year1.npy, s_oa.npy   per source, over its publications
  meta.json

A paper counts for one source only, its primary location's (where it was published, or the repository holding it
when it was not published elsewhere): a journal article copied to PubMed Central counts for the journal, not for
PubMed Central. ~20 minutes with 12 workers.
"""
import argparse
import json
import multiprocessing
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_institutions import G, log, lookup, strings_out  # noqa: E402


def init(threads, tmp, wn_path):
    import os
    os.environ['POLARS_MAX_THREADS'] = str(threads)
    G['tmp'] = Path(tmp)
    G['wn'] = np.load(wn_path, mmap_mode='r')


def part(args):
    i, (lcc_path, content_path) = args
    import polars as pl
    done = G['tmp'] / f'{i:05d}.npz'
    if done.exists():
        try:
            return i, int(len(np.load(done)['node']))
        except Exception:
            done.unlink()
    lcc = (pl.read_parquet(lcc_path, columns=['content_id', 'classification_id', 'relation', 'is_primary', 'source_field'])
           .filter(pl.col('classification_id').str.starts_with('openalex:S')
                   & pl.col('relation').is_in(['published_in', 'hosted_by'])))
    # the primary location's source; without one, the first location's
    best = (lcc.with_columns(pl.col('source_field').str.extract(r'locations\[(\d+)\]', 1).cast(pl.Int32, strict=False)
                             .fill_null(999).alias('loc'),
                             pl.col('is_primary').fill_null(False))
            .sort(['content_id', 'is_primary', 'loc'], descending=[False, True, False])
            .unique('content_id', keep='first', maintain_order=True)
            .select(pl.col('content_id').str.slice(10).cast(pl.Int64).alias('w'),
                    pl.col('classification_id').str.slice(10).cast(pl.Int64, strict=False).alias('s'))
            .drop_nulls())
    node = lookup(G['wn'], best['w'].to_numpy())
    keep = node >= 0
    c = (pl.read_parquet(content_path, columns=['content_id', 'is_oa'])
         .filter(pl.col('is_oa').fill_null(False))
         .select(pl.col('content_id').str.slice(10).cast(pl.Int64, strict=False).alias('w')).drop_nulls())
    onode = lookup(G['wn'], c['w'].to_numpy())
    np.savez(done, node=node[keep].astype(np.uint32), s=best['s'].to_numpy()[keep], oa=onode[onode >= 0].astype(np.uint32))
    return i, int(keep.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--citations', type=Path)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    import polars as pl
    from pipeline.tidy.dataset import manifest
    m = manifest()
    cit = args.citations or Path('/mnt/fast/citewalk') / m['generation']
    meta_c = json.loads((cit / 'meta.json').read_text())
    if meta_c['generation'] != m['generation']:
        raise SystemExit(f"citation index {meta_c['generation']} is not the tidy generation {m['generation']}")
    out = cit / 'sources'
    if (out / 'meta.json').exists():
        log(f'source index already built at {out}')
        return
    T = m['tables']
    pairs = list(zip(T['links_content_classifications']['files'], T['content']['files']))
    for a, b in pairs:
        if Path(a).parent != Path(b).parent:
            raise SystemExit(f'manifest lists are not aligned: {a} {b}')
    tmp = out / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wn_path = cit / 'wn.npy'
    N = int(np.load(wn_path, mmap_mode='r').shape[0])
    log(f'{len(pairs)} parts, {N:,} nodes')
    E = 0
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=init,
                             initargs=(args.threads, tmp, str(wn_path))) as pool:
        for k, (i, e) in enumerate(pool.map(part, enumerate(pairs)), 1):
            E += e
            if k % 200 == 0:
                log(f'  parts {k}/{len(pairs)} · {E:,} papers with a source')

    log('sources')
    zs = [np.load(tmp / f'{i:05d}.npz') for i in range(len(pairs))]
    snum = np.concatenate([z['s'] for z in zs])
    node = np.concatenate([z['node'] for z in zs])
    oa = np.zeros(N, bool)
    for z in zs:
        oa[z['oa']] = True
    del zs
    # a paper seen in two parts (an updated record): the last part's answer
    order = np.argsort(node, kind='stable')
    node, snum = node[order], snum[order]
    last = np.r_[node[1:] != node[:-1], True] if len(node) else np.zeros(0, bool)
    node, snum = node[last], snum[last]
    uniq = np.unique(snum)
    K = len(uniq)
    srow = np.searchsorted(uniq, snum).astype(np.int32)
    np.save(out / 'src_in.npy', uniq)
    w_src = np.full(N, -1, np.int32)
    w_src[node] = srow
    np.save(out / 'w_src.npy', w_src)
    np.save(out / 'oa.npy', oa)
    log(f'{K:,} sources, {len(node):,} papers with one')

    log('source -> papers, figures')
    year = np.asarray(np.load(cit / 'year.npy', mmap_mode='r'))
    cites = np.asarray(np.load(cit / 'cites.npy', mmap_mode='r')).astype(np.int64)
    kind = np.asarray(np.load(cit / 'kind.npy', mmap_mode='r'))
    sys.path.insert(0, str(REPO / 'search'))
    from srl_search.scope import SCHOLARLY_TYPES
    is_pub = np.array([t in SCHOLARLY_TYPES for t in meta_c['types']] + [False] * (256 - len(meta_c['types'])), bool)
    o = np.lexsort((year[node], srow))                   # by source, oldest first
    s_node, s_row = node[o], srow[o]
    s_off = np.zeros(K + 1, np.uint64)
    np.cumsum(np.bincount(s_row, minlength=K), out=s_off[1:])
    np.save(out / 's_off.npy', s_off)
    np.save(out / 's_node.npy', s_node.astype(np.uint32))
    pub = is_pub[kind[s_node]]
    pn, pr = s_node[pub], s_row[pub]
    c = np.maximum(cites[pn], 0)
    y = year[pn].astype(np.int64)
    np.save(out / 's_works.npy', np.bincount(pr, minlength=K).astype(np.int32))
    np.save(out / 's_cites.npy', np.bincount(pr, weights=c, minlength=K).astype(np.int64))
    np.save(out / 's_oa.npy', np.bincount(pr, weights=oa[pn], minlength=K).astype(np.int32))
    # pr is sorted (by source, then year): first and last years by reduceat over each source's run
    y0 = np.zeros(K, np.int64); y1 = np.zeros(K, np.int64)
    if len(pr):
        starts = np.flatnonzero(np.r_[True, pr[1:] != pr[:-1]])
        yy = np.where(y > 0, y, 1 << 20)
        lo = np.minimum.reduceat(yy, starts)
        y0[pr[starts]] = np.where(lo == 1 << 20, 0, lo)
        y1[pr[starts]] = np.maximum.reduceat(y, starts)
    np.save(out / 's_year0.npy', y0.astype(np.uint16))
    np.save(out / 's_year1.npy', y1.astype(np.uint16))
    # h-index: within each source, citations high to low; h = how many rank r (from 1) have >= r citations
    oh = np.lexsort((-c, pr))
    cr, rr = c[oh], pr[oh]
    start = np.zeros(K + 1, np.int64)
    np.cumsum(np.bincount(rr, minlength=K), out=start[1:])
    rank = np.arange(len(rr)) - start[rr] + 1
    np.save(out / 's_h.npy', np.bincount(rr, weights=cr >= rank, minlength=K).astype(np.int32))
    del o, s_node, s_row, pn, pr, c, y, oh, cr, rr, rank

    log('names, publishers')
    cl = (pl.scan_parquet(T['classifications']['files'])
          .filter((pl.col('scheme') == 'openalex.sources') & pl.col('classification_id').str.starts_with('openalex:S'))
          .select(pl.col('classification_id').str.slice(10).cast(pl.Int64, strict=False).alias('s'),
                  pl.col('name').fill_null('').str.slice(0, 300), pl.col('notation').fill_null(''),
                  'classification_type', 'observed_at')
          .drop_nulls('s').sort('observed_at', descending=True).unique('s', keep='first').collect())
    host = (pl.scan_parquet(T['links_classifications_entities']['files'])
            .filter(pl.col('relation') == 'host_organization')
            .select(pl.col('classification_id').str.slice(10).cast(pl.Int64, strict=False).alias('s'),
                    pl.col('entity_id').str.slice(9).alias('p'), 'observed_at')
            .drop_nulls().sort('observed_at', descending=True).unique('s', keep='first').select('s', 'p').collect())
    ents = (pl.scan_parquet(T['entities']['files'])
            .filter(pl.col('entity_id').str.starts_with('openalex:P') | pl.col('entity_id').str.starts_with('openalex:I'))
            .select(pl.col('entity_id').str.slice(9).alias('p'), pl.col('name').fill_null('').alias('pname'))
            .unique('p', keep='first').collect())
    full = (pl.DataFrame({'s': uniq}).with_row_index('r').join(cl, on='s', how='left')
            .join(host, on='s', how='left').join(ents, on='p', how='left').sort('r'))
    strings_out(out, 'name', full['name'].fill_null('').to_list())
    strings_out(out, 'issn', full['notation'].fill_null('').to_list())
    kinds = sorted(set(full['classification_type'].fill_null('').to_list()))
    kind_of = {k: j for j, k in enumerate(kinds)}
    np.save(out / 'kind.npy', np.array([kind_of[k] for k in full['classification_type'].fill_null('').to_list()], np.uint8))
    pubs = sorted({p for p in full['p'].to_list() if p})
    pub_of = {p: j for j, p in enumerate(pubs)}
    pname = dict(zip(full['p'].to_list(), full['pname'].to_list()))
    np.save(out / 'pub.npy', np.array([pub_of.get(p, -1) if p else -1 for p in full['p'].to_list()], np.int32))
    strings_out(out, 'pub_name', [pname.get(p) or '' for p in pubs])
    np.save(out / 'pub_ids.npy', np.array(pubs, dtype='U16'))

    meta = dict(generation=meta_c['generation'], nodes=N, sources=K, papers_with_source=int(len(node)),
                sources_with_name=int((full['name'].fill_null('') != '').sum()), kinds=kinds, publishers=len(pubs),
                built_at=time.time(),
                source='tidy links_content_classifications (published_in / hosted_by an openalex:S source, the '
                       'primary location first), classifications (openalex.sources), links_classifications_entities '
                       '(host_organization), content (is_oa)')
    shutil.rmtree(tmp)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    log(f'done: {K:,} sources, {len(node):,} papers in {(time.time() - t0) / 60:.0f} min')


if __name__ == '__main__':
    main()
