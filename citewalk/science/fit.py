"""Step 3 of the science-atlas pilot: fit candidate 2-D projections and measure them honestly.

    cd citewalk/science && uv run python fit.py

Only scholarly publication types (scope.py) are used.
Reference set R (shapes the geography): a per-year quota proportional to sqrt(eligible works
that year), total ~300k, lowest hash first. Square-root weighting is a deliberate compromise:
old periods get far more room than raw volume would give them, yet a 2025 still outweighs a 1950.
Evaluation set E (never used to fit): equal count per decade, highest hash first, disjoint from R.

Every method is scored on the same pool R ∪ E:
  knn10_out   share of each E paper's 10 nearest neighbours (cosine, original 768-D, within the
              pool) that are also among its 10 nearest on the map. Out-of-sample: what new papers get.
  knn10_in    the same for 20k fitted R papers.
  subfield    share of map neighbours that have the paper's OpenAlex subfield (the 768-D value is
              the ceiling); OpenAlex labels are themselves model output, so this is a sanity check.
  spearman    rank correlation of original vs map distance on random pairs (global layout).
  by decade   knn10_out per decade, to expose eras the map handles badly.
Held-out time: UMAP refitted without any 2020+ references, then 2020+ E papers placed on it.

Writes /mnt/fast/science-atlas/fit/: <method>.npz (reference IDs and coordinates), report.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import faiss
import numpy as np
import pyarrow.parquet as pq

import placement
import scope

ROOT = Path(os.environ.get('ATLAS_ROOT', '/mnt/fast/science-atlas'))
OUT = Path(os.environ.get('FIT_OUT', ROOT / 'fit'))
REF_TOTAL = int(os.environ.get('REF_TOTAL', 300_000))
EVAL_PER_DECADE = int(os.environ.get('EVAL_PER_DECADE', 1500))
K = 10
T0 = time.time()


def log(msg):
    print(f'{time.time() - T0:7.0f}s {msg}', flush=True)


def load():
    vec = ROOT / 'vectors'
    work = np.load(vec / 'work.npy')
    X = np.load(vec / 'vectors.npy', mmap_mode='r')
    meta = pq.read_table(sorted((ROOT / 'sample').glob('year=*.parquet')),
                         columns=['work', 'u', 'year', 'field', 'subfield', 'type'])
    o = np.argsort(meta['work'].to_numpy())
    meta = meta.take(o)
    assert np.array_equal(meta['work'].to_numpy(), work)
    return work, X, meta


def scholarly(meta):
    return np.isin(np.asarray(meta['type'].fill_null('').to_pylist(), dtype=object), list(scope.SCHOLARLY_TYPES))


def choose_sets(meta, eligible):
    """eligible: {year: count of scholarly eligible works}. Only scholarly works are used."""
    year = meta['year'].to_numpy().astype(int)
    u = meta['u'].to_numpy()
    kept = scholarly(meta)
    decade = year // 10 * 10
    E = []
    for d in range(1880, 2030, 10):
        idx = np.flatnonzero((decade == d) & kept)
        E.append(idx[np.argsort(-u[idx])][:EVAL_PER_DECADE])
    E = np.sort(np.concatenate(E))
    taken = ~kept; taken[E] = True
    years = sorted(int(y) for y in eligible if 1800 <= int(y) <= 2026 and eligible[y] > 0)
    weight = {y: np.sqrt(eligible[y]) for y in years}
    total = sum(weight.values())
    R = []
    for y in years:
        idx = np.flatnonzero((year == y) & ~taken)
        R.append(idx[np.argsort(u[idx])][:int(round(REF_TOTAL * weight[y] / total))])
    return np.sort(np.concatenate(R)), E


def exact_knn(base, queries, k, exclude=None):
    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)
    _, nn = index.search(queries, k + (exclude is not None))
    if exclude is None:
        return nn
    out = np.empty((len(queries), k), np.int64)
    for i, row in enumerate(nn):
        row = row[row != exclude[i]]
        out[i] = row[:k]
    return out


def knn_place(Xref, Yref, Xnew, exclude=None):
    ex = None if exclude is None else np.asarray(exclude)
    return placement.place_faiss(Xref, Yref, Xnew, exclude=ex, log=lambda m: None)


def score(Y_pool, hd_nn, rows, sub_pool):
    """rows index into the pool; hd_nn[i] are true neighbours of rows[i] (pool indices)."""
    lo = exact_knn_2d(Y_pool, Y_pool[rows], K, exclude=rows)
    overlap = np.array([len(set(a) & set(b)) for a, b in zip(lo, hd_nn)]) / K
    same = (sub_pool[lo] == sub_pool[rows][:, None]).mean()
    return overlap, float(same)


def exact_knn_2d(Y, Q, k, exclude):
    index = faiss.IndexFlatL2(2)
    index.add(np.ascontiguousarray(Y, dtype=np.float32))
    _, nn = index.search(np.ascontiguousarray(Q, dtype=np.float32), k + 1)
    out = np.empty((len(Q), k), np.int64)
    for i, row in enumerate(nn):
        out[i] = row[row != exclude[i]][:k]
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    faiss.omp_set_num_threads(int(os.environ.get('THREADS', 40)))
    OUT.mkdir(parents=True, exist_ok=True)
    work, X16, meta = load()
    R, E = choose_sets(meta, {y: k for y, (k, _) in scope.counts_per_year().items()})
    year = meta['year'].to_numpy().astype(int)
    _, sub_codes = np.unique(np.asarray(meta['subfield'].fill_null('').to_pylist(), dtype=object).astype(str),
                             return_inverse=True)
    log(f'R={len(R):,} references, E={len(E):,} held-out evaluation papers')
    pool = np.concatenate([R, E])            # pool position p -> sample row pool[p]
    nR = len(R)
    Xp = np.ascontiguousarray(X16[pool], dtype=np.float32)
    XR, XE = Xp[:nR], Xp[nR:]
    sub_pool = sub_codes[pool]
    rng = np.random.default_rng(0)
    in_rows = np.sort(rng.choice(nR, min(nR, 20000), replace=False))
    out_rows = np.arange(nR, len(pool))
    hd_out = exact_knn(Xp, XE, K, exclude=out_rows)
    hd_in = exact_knn(Xp, Xp[in_rows], K, exclude=in_rows)
    ceiling = float((sub_pool[hd_out] == sub_pool[out_rows][:, None]).mean())
    pairs = rng.integers(0, len(pool), (100_000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    hd_dist = 1 - (Xp[pairs[:, 0]] * Xp[pairs[:, 1]]).sum(1)
    log(f'exact 768-D neighbours done; subfield agreement of true neighbours = {ceiling:.3f}')

    E_decade = year[E] // 10 * 10
    report = {'references': int(nR), 'evaluation': int(len(E)), 'k': K,
              'types': sorted(scope.SCHOLARLY_TYPES), 'excluded_types': scope.EXCLUDED_NOTE,
              'subfield_ceiling_768d': ceiling, 'methods': {}}

    def record(name, Y_pool, seconds, note):
        o, same = score(Y_pool, hd_out, out_rows, sub_pool)
        i, _ = score(Y_pool, hd_in, in_rows, sub_pool)
        d2 = np.linalg.norm(Y_pool[pairs[:, 0]] - Y_pool[pairs[:, 1]], axis=1)
        by = {str(d): float(o[E_decade == d].mean()) for d in np.unique(E_decade)}
        report['methods'][name] = {'knn10_out': float(o.mean()), 'knn10_in': float(i.mean()), 'subfield_out': same,
                                   'spearman': spearman(hd_dist, d2), 'seconds': round(seconds), 'note': note,
                                   'knn10_out_by_decade': by}
        log(f"{name:24s} out {o.mean():.3f}  in {i.mean():.3f}  subfield {same:.3f}  "
            f"spearman {report['methods'][name]['spearman']:.3f}")
        (OUT / 'report.json').write_text(json.dumps(report, indent=1))

    def save(name, Y_R):
        np.savez(OUT / f'{name}.npz', work=work[R], coords=Y_R.astype(np.float32))

    # 1. PCA baseline (linear, the current citewalk approach)
    t = time.time()
    mu = XR.mean(0)
    _, _, Vt = np.linalg.svd(XR[rng.choice(nR, min(nR, 50000), replace=False)] - mu, full_matrices=False)
    record('pca', (Xp - mu) @ Vt[:2].T, time.time() - t, 'linear; fitted directions applied to all')

    # 2. UMAP fitted on R
    import umap
    t = time.time()
    um = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', low_memory=True, verbose=True,
                   n_jobs=int(os.environ.get('THREADS', 40)))
    Y_R = um.fit_transform(XR).astype(np.float32)
    fit_s = time.time() - t
    save('umap', Y_R)
    log(f'UMAP fit {fit_s:.0f}s')
    t = time.time()
    record('umap+knn', np.vstack([Y_R, knn_place(XR, Y_R, XE)]), fit_s + time.time() - t,
           'UMAP on references; new papers at the similarity-weighted mean of 10 nearest references')
    t = time.time()
    record('umap+transform', np.vstack([Y_R, um.transform(XE).astype(np.float32)]), fit_s + time.time() - t,
           "UMAP on references; new papers placed by umap's transform")
    Y_self = knn_place(XR, Y_R, XR[in_rows], exclude=in_rows)
    Y_tmp = np.vstack([Y_R, knn_place(XR, Y_R, XE)]); Y_tmp[in_rows] = Y_self
    i, _ = score(Y_tmp, hd_in, in_rows, sub_pool)
    report['methods']['umap+knn']['knn10_in_if_placed_too'] = float(i.mean())
    del um

    # 3. t-SNE fitted on R (openTSNE has an out-of-sample transform)
    from openTSNE import TSNE
    t = time.time()
    emb = TSNE(perplexity=30, metric='cosine', n_jobs=int(os.environ.get('THREADS', 40)),
               initialization='pca', random_state=0, verbose=True).fit(XR)
    Y_T = np.asarray(emb, dtype=np.float32)
    fit_s = time.time() - t
    save('tsne', Y_T)
    log(f't-SNE fit {fit_s:.0f}s')
    t = time.time()
    record('tsne+knn', np.vstack([Y_T, knn_place(XR, Y_T, XE)]), fit_s + time.time() - t,
           't-SNE on references; new papers at the similarity-weighted mean of 10 nearest references')
    t = time.time()
    record('tsne+transform', np.vstack([Y_T, np.asarray(emb.transform(XE), np.float32)]), fit_s + time.time() - t,
           "t-SNE on references; new papers placed by openTSNE's transform")
    del emb

    # 4. Time held-out: geography fitted without 2020+, then 2020+ papers placed on it
    old = year[R] < 2020
    t = time.time()
    um = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', low_memory=True,
                   n_jobs=int(os.environ.get('THREADS', 40)))
    Y_old = um.fit_transform(XR[old]).astype(np.float32)
    Y_pool = np.empty((len(pool), 2), np.float32)
    Y_pool[:nR][old] = Y_old
    Y_pool[:nR][~old] = knn_place(XR[old], Y_old, XR[~old])
    Y_pool[nR:] = knn_place(XR[old], Y_old, XE)
    o, _ = score(Y_pool, hd_out, out_rows, sub_pool)
    o_full = report['methods']['umap+knn']
    recent = E_decade == 2020
    report['time_holdout'] = {
        'fit_without_2020_plus': float(o[recent].mean()),
        'fit_with_all_years': o_full['knn10_out_by_decade'].get('2020'),
        'note': 'knn10_out of 2020s evaluation papers; both use UMAP + weighted-nearest placement',
        'seconds': round(time.time() - t)}
    log(f"time held-out: 2020s papers {o[recent].mean():.3f} (fit without them) vs "
        f"{o_full['knn10_out_by_decade'].get('2020'):.3f} (fit with them)")
    np.save(OUT / 'R.npy', work[R]); np.save(OUT / 'E.npy', work[E])
    (OUT / 'report.json').write_text(json.dumps(report, indent=1))
    log('done')


if __name__ == '__main__':
    main()
