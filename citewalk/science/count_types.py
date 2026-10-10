"""Exact count of embedding-plan-eligible works per publication year and OpenAlex type.

    cd pipeline/embeddings && uv run python ../../citewalk/science/count_types.py

Same eligibility as the backfill (language 'en', non-empty abstract), same pinned snapshot,
year from the publication date. Writes /mnt/fast/science-atlas/sample/type_counts.json:
{"<year>": {"<type>": count}}.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

PLAN = Path('/mnt/fast/openalex-embeddings/denseon-english-abstracts')
OUT = Path('/mnt/fast/science-atlas/sample/type_counts.json')


def work(entry):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline' / 'embeddings'))
    from embed_content import check_source, selection_mask
    pf = pq.ParquetFile(check_source(entry))
    c = Counter()
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=['language', 'abstract', 'date', 'type'])
        t = t.filter(selection_mask(t, 'has_english_abstract')).drop_columns(['abstract'])
        t = t.filter(pc.is_valid(t['date']))
        years = pc.year(t['date']).to_pylist()
        for y, ty in zip(years, t['type'].fill_null('unknown').to_pylist()):
            c[(y, ty)] += 1
    return c


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline' / 'embeddings'))
    from embed_content import read_manifest
    doc = read_manifest(PLAN)
    files = sorted(doc['files'], key=lambda e: -e['signature']['bytes'])
    total = Counter()
    with ProcessPoolExecutor(int(os.environ.get('WORKERS', 12))) as ex:
        for c in ex.map(work, files):
            total.update(c)
    out = defaultdict(dict)
    for (y, ty), n in sorted(total.items()):
        out[str(y)][ty] = n
    expected = doc['expected_rows'] - sum(g['rows'] for f in doc['files'] for rg in f['row_groups']
                                          for g in rg['dates'] if not g['date'])
    got = sum(total.values())
    if got != expected:
        raise SystemExit(f'Counted {got:,} dated eligible works, plan says {expected:,}')
    OUT.write_text(json.dumps(out, indent=0))
    print(f'{got:,} dated eligible works counted, matches the plan', flush=True)


if __name__ == '__main__':
    main()
