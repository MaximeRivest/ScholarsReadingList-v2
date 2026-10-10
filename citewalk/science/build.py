"""Step 4 of the science-atlas pilot: freeze one map and write what the viewer needs.

    cd citewalk/science && uv run python build.py --method umap

Positions: references keep their fitted coordinates; every other paper goes through the frozen
placement function (placement.py). Only the display sample (u < 1%, every year) is drawn; the
extra historical papers in the reference set shaped the geography but are never shown, so the
volumes on screen stay proportional to the real ones.

Output /mnt/fast/science-atlas/build/<id>/ :
  points.f32     x,y per display paper (unit square), sorted by year
  year.u16, field.u8, subfield.u16
  meta.json      years, per-year offsets, field/subfield names, colours, label anchors,
                 per-year counts per label, quality report
  info.parquet   work, year, title, venue, field, subfield, topic, type, citations (display order)
  vectors.npy    float16 display vectors (display order), for true-neighbour checks
  reference.npz  the frozen map: reference work IDs, vectors, unit-square coordinates, K, TAU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter

import placement
import scope

ROOT = Path(os.environ.get('ATLAS_ROOT', '/mnt/fast/science-atlas'))
T0 = time.time()

DOMAIN_HUES = {  # hue ranges per OpenAlex domain, so related fields share a colour family
    'Physical Sciences': (185, 250), 'Life Sciences': (85, 150),
    'Health Sciences': (345, 400), 'Social Sciences': (265, 330)}


def log(msg):
    print(f'{time.time() - T0:7.0f}s {msg}', flush=True)


def hsl_hex(h, s, l):
    import colorsys
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360, l, s)
    return '#%02x%02x%02x' % (round(r * 255), round(g * 255), round(b * 255))


def field_colours(fields, domain_of):
    out = {}
    for dom, (lo, hi) in DOMAIN_HUES.items():
        members = sorted(f for f in fields if domain_of.get(f) == dom)
        for k, f in enumerate(members):
            h = lo + (hi - lo) * (k + .5) / max(1, len(members))
            out[f] = hsl_hex(h, .78, .58 if k % 2 else .66)
    for f in fields:
        out.setdefault(f, '#8a8f98')
    return out


def anchors(xy, codes, n_codes, years, year_list, sigma, grid=384, min_points=30):
    """Densest location of each label (not its mean, which can fall in empty space between lobes)."""
    res = []
    ix = np.clip((xy * grid).astype(int), 0, grid - 1)
    counts = np.zeros((n_codes, len(year_list)), np.int32)
    yi = years - year_list[0]
    np.add.at(counts, (codes, yi), 1)
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(n_codes + 1))
    for c in range(n_codes):
        sel = order[bounds[c]:bounds[c + 1]]
        if len(sel) < min_points:
            res.append(None); continue
        h = np.zeros((grid, grid), np.float32)
        np.add.at(h, (ix[sel, 1], ix[sel, 0]), 1)
        h = gaussian_filter(h, sigma)
        r, q = np.unravel_index(np.argmax(h), h.shape)
        share = float(h[r, q] / h.sum())
        res.append({'x': (q + .5) / grid, 'y': (r + .5) / grid, 'peak_share': round(share, 4)})
    return res, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='umap', choices=['umap', 'tsne'])
    ap.add_argument('--threads', type=int, default=int(os.environ.get('THREADS', 40)))
    args = ap.parse_args()
    import faiss
    faiss.omp_set_num_threads(args.threads)

    vec = ROOT / 'vectors'
    work = np.load(vec / 'work.npy')
    X = np.load(vec / 'vectors.npy', mmap_mode='r')
    cols = ['work', 'u', 'year', 'title', 'venue', 'type', 'topic', 'subfield', 'field', 'domain', 'citations']
    meta = pq.read_table(sorted((ROOT / 'sample').glob('year=*.parquet')), columns=cols)
    meta = meta.take(np.argsort(meta['work'].to_numpy()))
    assert np.array_equal(meta['work'].to_numpy(), work)
    sample_summary = json.loads((ROOT / 'sample' / 'summary.json').read_text())
    report = json.loads((ROOT / 'fit' / 'report.json').read_text())
    fit = np.load(ROOT / 'fit' / f'{args.method}.npz')
    ref_work, ref_xy = fit['work'], fit['coords']
    ref_rows = np.searchsorted(work, ref_work)
    assert np.array_equal(work[ref_rows], ref_work)
    assert report.get('types') == sorted(scope.SCHOLARLY_TYPES), 'fit was made with a different type scope'
    log(f'{len(work):,} sampled, {len(ref_rows):,} references ({args.method})')

    u = meta['u'].to_numpy()
    types = np.asarray(meta['type'].fill_null('').to_pylist(), dtype=object)
    display = np.flatnonzero((u < sample_summary['display_rate']) & np.isin(types, list(scope.SCHOLARLY_TYPES)))
    year = meta['year'].to_numpy().astype(np.int32)
    display = display[np.lexsort((u[display], year[display]))]        # by year, then hash
    is_ref = np.full(len(work), -1, np.int64)
    is_ref[ref_rows] = np.arange(len(ref_rows))

    XR = np.ascontiguousarray(X[ref_rows], dtype=np.float32)
    xy = np.empty((len(display), 2), np.float32)
    fitted = is_ref[display] >= 0
    xy[fitted] = ref_xy[is_ref[display][fitted]]
    rest = np.flatnonzero(~fitted)
    xy[rest] = placement.place_faiss(XR, ref_xy, X[display[rest]], log=log)

    # Unit square from robust bounds (0.1–99.9 %) of the reference layout; nothing is clipped.
    lo, hi = np.percentile(ref_xy, .1, 0), np.percentile(ref_xy, 99.9, 0)
    span = float((hi - lo).max()) * 1.04
    centre = (lo + hi) / 2
    norm = lambda p: ((p - centre) / span + .5).astype(np.float32)
    xy_n, ref_n = norm(xy), norm(ref_xy)

    fields = meta['field'].fill_null('').to_numpy(zero_copy_only=False)[display].astype(str)
    subfields = meta['subfield'].fill_null('').to_numpy(zero_copy_only=False)[display].astype(str)
    domains = meta['domain'].fill_null('').to_numpy(zero_copy_only=False)[display].astype(str)
    field_names, field_code = np.unique(fields, return_inverse=True)
    sub_names, sub_code = np.unique(subfields, return_inverse=True)
    domain_of = {}
    for f, d in zip(fields, domains):
        domain_of.setdefault(f, d)
    colours = field_colours([f for f in field_names if f], domain_of)
    sub_field = {}
    for s, f in zip(subfields, fields):
        sub_field.setdefault(s, f)

    dy = year[display]
    year_list = np.arange(int(dy.min()), int(dy.max()) + 1)
    offsets = np.searchsorted(dy, np.r_[year_list, year_list[-1] + 1]).tolist()
    f_anchor, f_counts = anchors(xy_n, field_code, len(field_names), dy, year_list, sigma=10)
    s_anchor, s_counts = anchors(xy_n, sub_code, len(sub_names), dy, year_list, sigma=3)
    log('labels done')

    ident = time.strftime('%Y%m%dT%H%M%S') + '-' + args.method
    out = ROOT / 'build' / ident
    tmp = ROOT / 'build' / f'.{ident}.tmp'
    tmp.mkdir(parents=True)
    xy_n.tofile(tmp / 'points.f32')
    dy.astype(np.uint16).tofile(tmp / 'year.u16')
    field_code.astype(np.uint8).tofile(tmp / 'field.u8')
    sub_code.astype(np.uint16).tofile(tmp / 'subfield.u16')
    info = meta.take(pa.array(display)).select(['work', 'year', 'title', 'venue', 'type', 'topic', 'subfield',
                                                'field', 'citations'])
    pq.write_table(info, tmp / 'info.parquet', compression='zstd')
    np.save(tmp / 'vectors.npy', np.asarray(X[display]))
    np.savez(tmp / 'reference.npz', work=ref_work, coords=ref_n, vectors=XR.astype(np.float16),
             k=placement.K, tau=placement.TAU, method=args.method)
    per_year = scope.counts_per_year()
    doc = {
        'id': ident, 'method': args.method, 'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'embedding_plan': sample_summary['plan_id'], 'display_rate': sample_summary['display_rate'],
        'n': int(len(display)), 'references': int(len(ref_rows)),
        'reference_ids_sha256': hashlib.sha256(ref_work.tobytes()).hexdigest(),
        'years': year_list.tolist(), 'offsets': offsets,
        'eligible_per_year': [int(per_year.get(int(y), (0, 0))[0]) for y in year_list],
        'all_types_per_year': [int(per_year.get(int(y), (0, 0))[1]) for y in year_list],
        'types': sorted(scope.SCHOLARLY_TYPES), 'excluded_types': scope.EXCLUDED_NOTE,
        'fields': [{'name': n or 'Unclassified', 'colour': colours.get(n, '#8a8f98'),
                    'domain': domain_of.get(n, ''), 'anchor': a, 'per_year': c.tolist()}
                   for n, a, c in zip(field_names, f_anchor, f_counts)],
        'subfields': [{'name': n or 'Unclassified', 'field': sub_field.get(n, ''), 'anchor': a,
                       'per_year': c.tolist()} for n, a, c in zip(sub_names, s_anchor, s_counts)],
        'quality': report, 'placement': {'k': placement.K, 'tau': placement.TAU},
    }
    (tmp / 'meta.json').write_text(json.dumps(doc))
    os.replace(tmp, out)
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
