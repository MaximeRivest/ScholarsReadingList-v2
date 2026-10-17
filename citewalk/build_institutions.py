"""Build the institution index beside the citation index: which organisations each paper came from, which papers
each organisation produced, and where each author has worked.

    uv run --with-requirements citewalk/requirements.txt python citewalk/build_institutions.py

Output, in /mnt/fast/citewalk/<generation>/institutions/, aligned with the citation index's nodes (node i =
position i in ../wn.npy) and with the authorship index's author rows (../authors/author_an.npy):

  inst_in.npy    int64[K]     OpenAlex institution number (I-number) of each institution, sorted
  name_off.npy, names.bin     UTF-8 display names;  ror_off.npy, rors.bin  ROR ids ('' when none)
  country.npy    S2[K]        ISO country code ('' when none);  kind.npy  uint8[K] index into meta.json's kinds
  w_off.npy      uint64[N+1]  CSR: the institutions of each paper (distinct; any author's affiliation)
  w_inst.npy     uint32[E]      institution rows
  i_off.npy      uint64[K+1]  CSR: the papers of each institution, oldest first
  i_node.npy     uint32[E]      nodes
  ai_off.npy     uint64[M+1]  CSR by author row: where the author has published from
  ai_inst.npy    uint32[F]      institution row
  ai_n.npy       uint32[F]      papers from there;  ai_year0.npy, ai_year1.npy uint16[F] first and last year
  meta.json

Institutions are OpenAlex's resolved institution entities (authorships[].institutions / affiliations[].institution_ids),
not raw affiliation strings: only the ~60-80 % of recent papers OpenAlex could match to an organisation appear.
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
G = {}


def log(msg):
    print(time.strftime('%H:%M:%S ') + msg, flush=True)


def init_worker(threads, tmp, wn_path, an_path):
    os.environ['POLARS_MAX_THREADS'] = str(threads)
    G['tmp'] = Path(tmp)
    G['wn'] = np.load(wn_path, mmap_mode='r')
    G['an'] = np.load(an_path, mmap_mode='r')


def lookup(sorted_arr, x):
    order = np.argsort(x, kind='stable')
    idx = np.searchsorted(sorted_arr, x[order])
    idx_c = np.minimum(idx, len(sorted_arr) - 1)
    found = sorted_arr[idx_c] == x[order]
    out = np.empty(len(x), np.int64)
    out[order] = np.where(found, idx_c, -1)
    return out


def part(args):
    i, (lcs_path, sig_path, lse_path) = args
    import polars as pl
    done = G['tmp'] / f'{i:05d}.npz'
    if done.exists():                                      # a run that stopped: keep its parts
        try:
            z = np.load(done)
            return i, int(len(z['node'])), 0
        except Exception:
            done.unlink()
    lcs = pl.read_parquet(lcs_path, columns=['content_id', 'signature_id'])
    lse = pl.read_parquet(lse_path, columns=['signature_id', 'entity_id', 'relation'])
    inst = (lse.filter(pl.col('relation') == 'identifies_institution')
            .select('signature_id', pl.col('entity_id').str.slice(10).cast(pl.Int64, strict=False).alias('i'))
            .drop_nulls().unique())
    per = (lse.filter(pl.col('relation') == 'identifies_person')
           .select('signature_id', pl.col('entity_id').str.slice(10).cast(pl.Int64, strict=False).alias('a'))
           .drop_nulls().unique('signature_id'))
    rows = (lcs.join(inst, on='signature_id', how='inner')
            .select(pl.col('content_id').str.slice(10).cast(pl.Int64).alias('w'), 'signature_id', 'i'))
    wi = rows.select('w', 'i').unique().sort('w', 'i')
    ai = (rows.join(per, on='signature_id', how='inner').select('w', 'a', 'i').unique())
    wn = G['wn']
    node = lookup(wn, wi['w'].to_numpy())
    keep = node >= 0
    anode = lookup(wn, ai['w'].to_numpy())
    arow = lookup(np.asarray(G['an']), ai['a'].to_numpy())
    akeep = (anode >= 0) & (arow >= 0)
    np.savez(G['tmp'] / f'{i:05d}.npz',
             node=node[keep].astype(np.uint32), i=wi['i'].to_numpy()[keep],
             anode=anode[akeep].astype(np.uint32), arow=arow[akeep].astype(np.uint32), ai=ai['i'].to_numpy()[akeep])
    return i, int(keep.sum()), int((~keep).sum())


def strings_out(out: Path, stem: str, values):
    """A list of strings as <stem>_off.npy + <stem>s.bin."""
    data = [v.encode('utf-8') for v in values]
    off = np.zeros(len(data) + 1, np.uint64)
    np.cumsum([len(d) for d in data], out=off[1:])
    np.save(out / f'{stem}_off.npy', off)
    with open(out / f'{stem}s.bin', 'wb') as fh:
        for d in data:
            fh.write(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--citations', type=Path, help='citation index directory (default: /mnt/fast/citewalk/<generation>)')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--limit-files', type=int, help='smoke test: only the first N parts, into institutions-firstN/')
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
    for a, b, c in trip:
        if not (Path(a).parent == Path(b).parent == Path(c).parent):
            raise SystemExit(f'manifest lists are not aligned: {a} {b} {c}')
    out = cit / 'institutions'
    if args.limit_files:
        trip = trip[:args.limit_files]
        out = cit / f'institutions-first{args.limit_files}'
    if (out / 'meta.json').exists():
        log(f'institution index already built at {out}')
        return
    tmp = out / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wn_path = cit / 'wn.npy'
    an_path = cit / 'authors' / 'author_an.npy'
    N = int(np.load(wn_path, mmap_mode='r').shape[0])
    M = int(np.load(an_path, mmap_mode='r').shape[0])
    log(f'{len(trip)} parts, {N:,} nodes, {M:,} authors')
    E = missing = 0
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker,
                             initargs=(args.threads, tmp, str(wn_path), str(an_path))) as pool:
        for k, (i, e, miss) in enumerate(pool.map(part, enumerate(trip)), 1):
            E += e
            missing += miss
            if k % 100 == 0:
                log(f'  parts {k}/{len(trip)} · {E:,} paper-institution pairs')
    log(f'paper-institution pairs: {E:,} ({missing:,} on works missing from the citation index)')

    log('institutions')
    uniq = np.unique(np.concatenate([np.unique(np.load(tmp / f'{i:05d}.npz')['i']) for i in range(len(trip))]))
    K = len(uniq)
    np.save(out / 'inst_in.npy', uniq)
    log(f'institutions: {K:,}')

    log('paper -> institutions')
    node = np.concatenate([np.load(tmp / f'{i:05d}.npz')['node'] for i in range(len(trip))])
    inst = np.concatenate([np.searchsorted(uniq, np.load(tmp / f'{i:05d}.npz')['i']).astype(np.uint32) for i in range(len(trip))])
    key = node.astype(np.int64) << 24 | inst.astype(np.int64)
    assert K < 1 << 24
    key = np.unique(key)                                 # distinct pairs, sorted by node then institution
    node, inst = (key >> 24).astype(np.uint32), (key & ((1 << 24) - 1)).astype(np.uint32)
    del key
    E = len(node)
    w_off = np.zeros(N + 1, np.uint64)
    np.cumsum(np.bincount(node, minlength=N), out=w_off[1:])
    np.save(out / 'w_off.npy', w_off)
    np.save(out / 'w_inst.npy', inst)
    del w_off

    log('institution -> papers')
    year = np.load(cit / 'year.npy', mmap_mode='r')
    o1 = np.argsort(np.asarray(year)[node], kind='stable')
    o2 = np.argsort(inst[o1], kind='stable')
    order = o1[o2]
    del o1, o2
    i_node = node[order]
    i_inst = inst[order]
    del order
    i_off = np.zeros(K + 1, np.uint64)
    np.cumsum(np.bincount(i_inst, minlength=K), out=i_off[1:])
    np.save(out / 'i_off.npy', i_off)
    np.save(out / 'i_node.npy', i_node)
    del i_node, i_inst, i_off, node, inst

    log('author -> institutions')
    arow = np.concatenate([np.load(tmp / f'{i:05d}.npz')['arow'] for i in range(len(trip))])
    ainst = np.concatenate([np.searchsorted(uniq, np.load(tmp / f'{i:05d}.npz')['ai']).astype(np.uint32) for i in range(len(trip))])
    anode = np.concatenate([np.load(tmp / f'{i:05d}.npz')['anode'] for i in range(len(trip))])
    ay = np.asarray(year)[anode].astype(np.int64)
    del anode
    key = arow.astype(np.int64) << 24 | ainst.astype(np.int64)
    del arow, ainst
    u, inv = np.unique(key, return_inverse=True)
    del key
    F = len(u)
    ai_n = np.bincount(inv, minlength=F).astype(np.uint32)
    ai_y0 = np.full(F, 1 << 20, np.int64)
    np.minimum.at(ai_y0, inv, np.where(ay > 0, ay, 1 << 20))
    ai_y0[ai_y0 == 1 << 20] = 0
    ai_y1 = np.zeros(F, np.int64)
    np.maximum.at(ai_y1, inv, ay)
    del inv, ay
    a_of = (u >> 24).astype(np.int64)
    ai_off = np.zeros(M + 1, np.uint64)
    np.cumsum(np.bincount(a_of, minlength=M), out=ai_off[1:])
    np.save(out / 'ai_off.npy', ai_off)
    np.save(out / 'ai_inst.npy', (u & ((1 << 24) - 1)).astype(np.uint32))
    np.save(out / 'ai_n.npy', ai_n)
    np.save(out / 'ai_year0.npy', ai_y0.astype(np.uint16))
    np.save(out / 'ai_year1.npy', ai_y1.astype(np.uint16))
    del u, a_of, ai_off, ai_n, ai_y0, ai_y1

    log('names')
    frames = []
    for f in T['entities']['files']:
        d = (pl.read_parquet(f, columns=['entity_id', 'entity_type', 'name', 'ror', 'country_code', 'organization_type'])
             .filter((pl.col('entity_type') == 'organization') & pl.col('entity_id').str.starts_with('openalex:I'))
             .select(pl.col('entity_id').str.slice(10).cast(pl.Int64, strict=False).alias('i'),
                     pl.col('name').fill_null('').str.slice(0, 300), pl.col('ror').fill_null(''),
                     pl.col('country_code').fill_null('').str.slice(0, 2), pl.col('organization_type').fill_null(''))
             .drop_nulls('i'))
        frames.append(d)
    ents = pl.concat(frames).unique('i', keep='first')
    full = (pl.DataFrame({'i': uniq}).with_row_index('r').join(ents, on='i', how='left').sort('r'))
    found = full['name'].is_not_null().to_numpy()
    strings_out(out, 'name', full['name'].fill_null('').to_list())
    strings_out(out, 'ror', full['ror'].fill_null('').to_list())
    np.save(out / 'country.npy', np.array(full['country_code'].fill_null('').to_list(), dtype='S2'))
    kinds = sorted(set(full['organization_type'].fill_null('').to_list()))
    kind_of = {k: j for j, k in enumerate(kinds)}
    np.save(out / 'kind.npy', np.array([kind_of[k] for k in full['organization_type'].fill_null('').to_list()], np.uint8))

    meta = dict(generation=meta_c['generation'], nodes=N, institutions=K, institutions_with_name=int(found.sum()),
                paper_institution_pairs=E, author_institution_pairs=F, kinds=kinds, parts=len(trip),
                works_missing_from_citation_index=missing, built_at=time.time(),
                source='tidy links_content_signatures + links_signatures_entities (identifies_institution, identifies_person); '
                       'names from entities (organization, I-numbers)')
    shutil.rmtree(tmp)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    log(f'done: {E:,} paper-institution pairs, {K:,} institutions, {F:,} author-institution pairs in {(time.time() - t0) / 60:.0f} min')


if __name__ == '__main__':
    main()
