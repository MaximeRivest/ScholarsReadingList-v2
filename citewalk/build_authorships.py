"""Build the affiliations of each authorship: on a given paper, which organisation each author signed from.

    uv run --with-requirements citewalk/requirements.txt python citewalk/build_authorships.py

The institution index (build_institutions.py) knows a paper's organisations and every organisation an author has
ever published from, but not who signed from where on one paper. Guessing it (an author's organisations that are
also on the paper, in a year they published from there) was measured against OpenAlex's live API on 60 random
papers of 2000-2025 (2026-10-17): 70 % precision, 87 % recall, 17 of 60 papers right for every author. Too wrong to
show. This index keeps each authorship's organisations as the snapshot records them (one signature = one
authorship's affiliation: its person and its organisations). Against the live API on the same 60 papers it agrees
77 % (precision) / 85 % (recall), 21 of 60 papers right for every author: the rest is OpenAlex itself, which has
re-matched affiliations since the snapshot (e.g. one affiliation string matched to three organisations then, one
now). It is the snapshot's own answer, the same data as every other count on the site.

Output, in /mnt/fast/citewalk/<generation>/institutions/authorships/, aligned with the citation index's nodes and
with the authorship index's author rows:

  aw_off.npy   uint64[N+1]  CSR by node: the (author, institution) pairs of each paper, sorted by author row
  aw_arow.npy  uint32[E]      author row (../../authors/author_an.npy)
  aw_inst.npy  uint32[E]      institution row (../inst_in.npy)
  meta.json

Authors OpenAlex could not tie to a person, and affiliations it could not tie to an organisation, are not here
(the same ~60-80 % of recent papers as the institution index). Built 2026-10-17: 628M triples on 136M papers,
36 minutes with 8 workers, 44 GB of memory at the peak (the final sort), 8.8 GB on disk.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_institutions import init_worker, log, lookup, G  # noqa: E402  (same workers' state and helpers)


def part(args):
    i, (lcs_path, lse_path) = args
    import polars as pl
    done = G['tmp'] / f'{i:05d}.npz'
    if done.exists():                                      # a run that stopped: keep its parts
        try:
            return i, int(len(np.load(done)['anode']))
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
    ai = (lcs.join(inst, on='signature_id', how='inner').join(per, on='signature_id', how='inner')
          .select(pl.col('content_id').str.slice(10).cast(pl.Int64).alias('w'), 'a', 'i').unique())
    anode = lookup(G['wn'], ai['w'].to_numpy())
    arow = lookup(np.asarray(G['an']), ai['a'].to_numpy())
    keep = (anode >= 0) & (arow >= 0)
    np.savez(G['tmp'] / f'{i:05d}.npz', anode=anode[keep].astype(np.uint32), arow=arow[keep].astype(np.uint32),
             ai=ai['i'].to_numpy()[keep])
    return i, int(keep.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--citations', type=Path, help='citation index directory (default: /mnt/fast/citewalk/<generation>)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    from pipeline.tidy.dataset import manifest
    m = manifest()
    cit = args.citations or Path('/mnt/fast/citewalk') / m['generation']
    meta_c = json.loads((cit / 'meta.json').read_text())
    if meta_c['generation'] != m['generation']:
        raise SystemExit(f"citation index {meta_c['generation']} is not the tidy generation {m['generation']}")
    if not (cit / 'institutions' / 'meta.json').exists():
        raise SystemExit('build the institution index first (build_institutions.py): rows are its institution rows')
    out = cit / 'institutions' / 'authorships'
    if (out / 'meta.json').exists():
        log(f'authorship affiliations already built at {out}')
        return
    T = m['tables']
    pairs = list(zip(T['links_content_signatures']['files'], T['links_signatures_entities']['files']))
    for a, c in pairs:
        if Path(a).parent != Path(c).parent:
            raise SystemExit(f'manifest lists are not aligned: {a} {c}')
    tmp = out / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wn_path, an_path = cit / 'wn.npy', cit / 'authors' / 'author_an.npy'
    N = int(np.load(wn_path, mmap_mode='r').shape[0])
    uniq = np.load(cit / 'institutions' / 'inst_in.npy')
    log(f'{len(pairs)} parts, {N:,} nodes, {len(uniq):,} institutions')
    E = 0
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker,
                             initargs=(args.threads, tmp, str(wn_path), str(an_path))) as pool:
        for k, (i, e) in enumerate(pool.map(part, enumerate(pairs)), 1):
            E += e
            if k % 100 == 0:
                log(f'  parts {k}/{len(pairs)} · {E:,} author-institution-paper triples')

    log('sorting by paper, then author')
    keys = []
    for i in range(len(pairs)):
        z = np.load(tmp / f'{i:05d}.npz')
        r = np.searchsorted(uniq, z['ai'])
        ok = (r < len(uniq)) & (uniq[np.minimum(r, len(uniq) - 1)] == z['ai'])
        keys.append((z['anode'][ok], z['arow'][ok], r[ok].astype(np.uint32)))
    node = np.concatenate([k[0] for k in keys])
    arow = np.concatenate([k[1] for k in keys])
    inst = np.concatenate([k[2] for k in keys])
    del keys
    order = np.lexsort((inst, arow, node))
    node, arow, inst = node[order], arow[order], inst[order]
    del order
    # distinct triples (a part boundary can repeat one)
    same = np.zeros(len(node), bool)
    same[1:] = (node[1:] == node[:-1]) & (arow[1:] == arow[:-1]) & (inst[1:] == inst[:-1])
    node, arow, inst = node[~same], arow[~same], inst[~same]
    E = len(node)
    off = np.zeros(N + 1, np.uint64)
    np.cumsum(np.bincount(node, minlength=N), out=off[1:])
    np.save(out / 'aw_off.npy', off)
    np.save(out / 'aw_arow.npy', arow)
    np.save(out / 'aw_inst.npy', inst)
    meta = dict(generation=meta_c['generation'], nodes=N, triples=int(E), papers_with_any=int((np.diff(off) > 0).sum()),
                parts=len(pairs), built_at=time.time(),
                source='tidy links_content_signatures + links_signatures_entities: per signature (one authorship), '
                       'its person (identifies_person) and its organisations (identifies_institution)')
    shutil.rmtree(tmp)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    log(f'done: {E:,} triples on {meta["papers_with_any"]:,} papers in {(time.time() - t0) / 60:.0f} min')


if __name__ == '__main__':
    main()
