"""Step 1 of the science-atlas pilot: draw a reproducible sample of the embedding plan's works.

    cd pipeline/embeddings && uv run python ../../citewalk/science/sample.py

Eligibility is exactly the DenseOn backfill's (`language == 'en'` and a non-empty abstract),
read from the same pinned input snapshot, plus a valid publication date in 1800..2026.

Every work gets a fixed pseudo-random number u in [0, 1) from its OpenAlex ID (splitmix64),
so the sample is reproducible and can be grown later by raising a threshold.

  display sample : u < 0.01 in every year (1%). Volumes stay proportional to the real ones.
  balance extra  : for 1880..1979, u < 3000 / eligible(year) when that exceeds 1%.
                   Used only to shape the geography, so that old fields get room on the map.
                   Never shown or counted as part of the display sample.

Output: /mnt/fast/science-atlas/sample/<year-file>.parquet and sample/summary.json
"""
from __future__ import annotations

import collections
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PLAN = Path('/mnt/fast/openalex-embeddings/denseon-english-abstracts')
OUT = Path('/mnt/fast/science-atlas/sample')
DISPLAY_RATE = 0.01
BALANCE_PER_YEAR = 3000
BALANCE_YEARS = (1880, 1979)
YEARS = (1800, 2026)
KEEP = ['title', 'abstract', 'type', 'topic', 'subfield', 'field', 'domain', 'citations', 'venue']


def splitmix_u(ids: np.ndarray) -> np.ndarray:
    """Deterministic uniform [0,1) per uint64 ID (splitmix64 finalizer, top 53 bits)."""
    with np.errstate(over='ignore'):
        z = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def eligible_per_year(manifest):
    c = collections.Counter()
    for f in manifest['files']:
        for g in f['row_groups']:
            for b in g['dates']:
                if b['date']:
                    c[int(b['date'][:4])] += b['rows']
    return c


def rates(counts):
    r = np.zeros(YEARS[1] - YEARS[0] + 1)
    for y in range(YEARS[0], YEARS[1] + 1):
        r[y - YEARS[0]] = DISPLAY_RATE
        if BALANCE_YEARS[0] <= y <= BALANCE_YEARS[1] and counts.get(y):
            r[y - YEARS[0]] = max(DISPLAY_RATE, min(1.0, BALANCE_PER_YEAR / counts[y]))
    return r


def work(args):
    entry, rate = args
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline' / 'embeddings'))
    from embed_content import check_source, selection_mask
    path = check_source(entry)   # refuses a changed source file
    pf = pq.ParquetFile(path)
    parts = []
    for g in range(pf.metadata.num_row_groups):
        head = pf.read_row_group(g, columns=['id', 'language', 'date'])
        prefix = 'https://openalex.org/W'
        if not pc.all(pc.starts_with(head['id'], prefix)).as_py():
            raise ValueError(f'Unexpected work ID format in {path} row group {g}')
        ids = pc.cast(pc.utf8_slice_codeunits(head['id'], len(prefix)), pa.uint64()).to_numpy()
        years = pc.year(head['date']).to_numpy(zero_copy_only=False).astype(float)  # NaN = undated
        ok = ~np.isnan(years) & (years >= YEARS[0]) & (years <= YEARS[1])
        yi = np.where(ok, years, YEARS[0]).astype(int) - YEARS[0]
        u = splitmix_u(ids)
        cand = ok & (u < rate[yi]) & pc.fill_null(pc.equal(head['language'], 'en'), False).to_numpy(zero_copy_only=False)
        if not cand.any():
            continue
        idx = np.flatnonzero(cand)
        rest = pf.read_row_group(g, columns=KEEP + ['language']).take(pa.array(idx))
        keep = selection_mask(rest, 'has_english_abstract').to_numpy(zero_copy_only=False)
        idx = idx[keep]
        rest = rest.filter(pa.array(keep)).drop_columns(['language'])
        t = pa.table({'work': pa.array(ids[idx]), 'u': pa.array(u[idx]),
                      'year': pa.array(years[idx].astype(np.int16)), 'date': head['date'].take(pa.array(idx))})
        for name in KEEP:
            t = t.append_column(name, rest[name])
        parts.append(t)
    if not parts:
        return entry['relative_path'], 0
    table = pa.concat_tables(parts)
    name = Path(entry['relative_path']).parent.name
    tmp = OUT / f'.{name}.parquet.tmp'
    pq.write_table(table, tmp, compression='zstd')
    os.replace(tmp, OUT / f'{name}.parquet')
    return entry['relative_path'], table.num_rows


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline' / 'embeddings'))
    from embed_content import read_manifest
    doc = read_manifest(PLAN)
    counts = eligible_per_year(doc)
    rate = rates(counts)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [e for e in doc['files'] if not (OUT / f"{Path(e['relative_path']).parent.name}.parquet").exists()]
    # biggest files first so the slowest ones do not end up alone at the tail
    todo.sort(key=lambda e: -e['signature']['bytes'])
    t0 = time.time(); total = 0
    with ProcessPoolExecutor(int(os.environ.get('WORKERS', 12))) as ex:
        for k, (rel, n) in enumerate(ex.map(work, [(e, rate) for e in todo])):
            total += n
            if k % 20 == 0:
                print(f'{time.time() - t0:7.0f}s {k + 1}/{len(todo)} files, {total:,} sampled (last {rel})', flush=True)
    summary = {'plan_id': doc['plan_id'], 'display_rate': DISPLAY_RATE, 'balance_per_year': BALANCE_PER_YEAR,
               'balance_years': BALANCE_YEARS, 'years': YEARS,
               'eligible_per_year': {str(y): counts[y] for y in sorted(counts)},
               'rate_per_year': {str(y): float(rate[y - YEARS[0]]) for y in range(YEARS[0], YEARS[1] + 1)},
               'hash': 'splitmix64(work number) top 53 bits'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
