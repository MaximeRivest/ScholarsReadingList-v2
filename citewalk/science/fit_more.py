"""Step 3b: follow-up measurements after fit.py (same R and E, same pool, same metrics).

    cd citewalk/science && uv run python fit_more.py

  - "neighbourhood" score: share of the 10 true neighbours found within the 100 nearest dots
  - placement sweep (K, TAU) for the t-SNE and UMAP layouts, out-of-sample
  - time held-out for t-SNE: fitted without 2020+ references, 2020s papers placed on it
  - quick previews (PNG) of the fitted reference layouts coloured by OpenAlex field
Updates /mnt/fast/science-atlas/fit/report.json in place (adds keys, keeps fit.py's).
"""
import json
import os
import time

import faiss
import numpy as np

import fit
import placement

T0 = time.time()
log = lambda m: print(f'{time.time() - T0:7.0f}s {m}', flush=True)


def knn2d(Y, rows, k):
    index = faiss.IndexFlatL2(2)
    index.add(np.ascontiguousarray(Y, dtype=np.float32))
    _, nn = index.search(np.ascontiguousarray(Y[rows], dtype=np.float32), k + 1)
    return np.array([r[r != q][:k] for r, q in zip(nn, rows)])


def recall(Y, rows, hd, k):
    lo = knn2d(Y, rows, k)
    return float(np.mean([len(set(a) & set(b)) for a, b in zip(lo, hd)]) / hd.shape[1])


def preview(Y, field_codes, path, n=200_000):
    from PIL import Image
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Y), min(n, len(Y)), replace=False)
    lo, hi = np.percentile(Y, .5, 0), np.percentile(Y, 99.5, 0)
    size = 1400
    p = ((Y[sel] - lo) / (hi - lo).max() * (size - 40) + 20).astype(int)
    img = np.zeros((size, size, 3), np.float32)
    pal = np.array([[(h * 0.61803 * 360) % 360] for h in range(64)])
    import colorsys
    cols = np.array([colorsys.hls_to_rgb(h[0] / 360, .6, .75) for h in pal])
    ok = (p >= 0).all(1) & (p < size).all(1)
    np.add.at(img, (p[ok, 1], p[ok, 0]), cols[field_codes[sel][ok] % 64])
    cnt = np.zeros((size, size), np.float32); np.add.at(cnt, (p[ok, 1], p[ok, 0]), 1)
    img = img / np.maximum(cnt, 1)[..., None] * (1 - np.exp(-cnt * 1.5))[..., None]
    Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).save(path)


FINAL = (placement.K, placement.TAU)


def main():
    faiss.omp_set_num_threads(int(os.environ.get('THREADS', 40)))
    out = fit.OUT
    report = json.loads((out / 'report.json').read_text())
    work, X16, meta = fit.load()
    R = np.searchsorted(work, np.load(out / 'R.npy')); E = np.searchsorted(work, np.load(out / 'E.npy'))
    year = meta['year'].to_numpy().astype(int)
    fit_sub = np.unique(np.asarray(meta['subfield'].fill_null('').to_pylist(), dtype=object).astype(str), return_inverse=True)[1]
    fields = np.unique(np.asarray(meta['field'].fill_null('').to_pylist(), dtype=object).astype(str), return_inverse=True)[1]
    pool = np.concatenate([R, E]); nR = len(R)
    Xp = np.ascontiguousarray(X16[pool], dtype=np.float32)
    XR, XE = Xp[:nR], Xp[nR:]
    out_rows = np.arange(nR, len(pool))
    hd = fit.exact_knn(Xp, XE, 10, exclude=out_rows)
    recent = (year[E] // 10 * 10) == 2020
    log('neighbours ready')

    layouts = {'tsne': np.load(out / 'tsne.npz')['coords'], 'umap': np.load(out / 'umap.npz')['coords']}
    for name, Y in layouts.items():
        preview(Y, fields[R], out / f'preview-{name}.png')
    sweep = {}
    for name, Y_R in (layouts.items() if os.environ.get('SWEEP', '1') == '1' else []):
        for k in (3, 5, 10, 20):
            for tau in (.01, .02, .03, .05):
                placement.K, placement.TAU = k, tau
                Y = np.vstack([Y_R, placement.place_faiss(XR, Y_R, XE, log=lambda m: None)])
                sweep[f'{name} k={k} tau={tau}'] = recall(Y, out_rows, hd, 10)
        log(f'{name} sweep: ' + ', '.join(f'{k.split(" ", 1)[1]}={v:.3f}' for k, v in sweep.items() if k.startswith(name)))
    placement.K, placement.TAU = FINAL
    hood = {}
    pca = report['methods']['pca']
    mu = XR.mean(0)
    _, _, Vt = np.linalg.svd(XR[np.random.default_rng(0).choice(nR, 50000, replace=False)] - mu, full_matrices=False)
    for name, Y in [('pca', (Xp - mu) @ Vt[:2].T)] + [
            (f'{n}+knn', np.vstack([Y_R, placement.place_faiss(XR, Y_R, XE, log=lambda m: None)])) for n, Y_R in layouts.items()]:
        hood[name] = recall(Y, out_rows, hd, 100)
        report['methods'][name]['knn10_in_100_out'] = hood[name]
    log('neighbourhood (10 true within 100 nearest dots): ' + ', '.join(f'{k}={v:.3f}' for k, v in hood.items()))

    from openTSNE import TSNE
    old = year[R] < 2020
    emb = TSNE(perplexity=30, metric='cosine', n_jobs=int(os.environ.get('THREADS', 40)),
               initialization='pca', random_state=0).fit(XR[old])
    Y_old = np.asarray(emb, np.float32)
    Y_pool = np.empty((len(pool), 2), np.float32)
    Y_pool[:nR][old] = Y_old
    Y_pool[:nR][~old] = placement.place_faiss(XR[old], Y_old, XR[~old], log=lambda m: None)
    Y_pool[nR:] = placement.place_faiss(XR[old], Y_old, XE, log=lambda m: None)
    Y_full = np.vstack([layouts['tsne'], placement.place_faiss(XR, layouts['tsne'], XE, log=lambda m: None)])
    rows = out_rows[recent]
    th = {'method': 'tsne+knn',
          'fit_without_2020_plus': recall(Y_pool, rows, hd[recent], 10),
          'fit_with_all_years': recall(Y_full, rows, hd[recent], 10),
          'neighbourhood_without_2020_plus': recall(Y_pool, rows, hd[recent], 100),
          'neighbourhood_with_all_years': recall(Y_full, rows, hd[recent], 100),
          'references_removed': int((~old).sum()),
          'note': '2020s evaluation papers; geography fitted without any 2020+ reference, then placed'}
    report['time_holdout_umap'] = report.pop('time_holdout', None)
    report['time_holdout'] = th
    if sweep:
        report['placement_sweep'] = sweep
    report['placement_final'] = {'k': FINAL[0], 'tau': FINAL[1]}
    # out-of-sample scores at the final placement parameters (fit.py used k=10, tau=0.03)
    for name, Y_R in layouts.items():
        Y = np.vstack([Y_R, placement.place_faiss(XR, Y_R, XE, log=lambda m: None)])
        o = fit.score(Y, hd, out_rows, fit_sub[pool])
        m = report['methods'][f'{name}+knn']
        m['knn10_out'], m['subfield_out'] = float(o[0].mean()), o[1]
        dec = year[E] // 10 * 10
        m['knn10_out_by_decade'] = {str(d): float(o[0][dec == d].mean()) for d in np.unique(dec)}
        m['note'] += f' (k={FINAL[0]}, tau={FINAL[1]})'
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    log(f'time held-out (t-SNE): {th}')


if __name__ == '__main__':
    main()
