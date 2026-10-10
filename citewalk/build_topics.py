"""Add each paper's primary OpenAlex topic to the citation index (for naming map regions).

    uv run --with-requirements citewalk/requirements.txt python citewalk/build_topics.py

Writes into the existing index directory (/mnt/fast/citewalk/<generation>/):
  topic.npy     int16[N]  index into topics.json (-1 = no primary topic)
  topics.json   [{id, name, subfield, field}]
Source: links_content_classifications rows with is_primary = true and an openalex:T… id.
"""
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def part(path):
    os.environ.setdefault('POLARS_MAX_THREADS', '2')
    import polars as pl
    d = (pl.scan_parquet(path)
         .select('content_id', 'classification_id', 'is_primary')
         .filter(pl.col('is_primary') & pl.col('classification_id').str.starts_with('openalex:T'))
         .select(pl.col('content_id').str.slice(10).cast(pl.Int64).alias('w'),
                 pl.col('classification_id').str.slice(10).cast(pl.Int32).alias('t'))
         .collect())
    return d['w'].to_numpy(), d['t'].to_numpy()


def main():
    import polars as pl
    from pipeline.tidy.dataset import manifest
    m = manifest()
    out = Path('/mnt/fast/citewalk') / m['generation']
    if (out / 'topic.npy').exists():
        print('already built'); return
    wn = np.load(out / 'wn.npy', mmap_mode='r')
    T = m['tables']
    t0 = time.time()
    # topic names and their subfield/field
    cls = pl.scan_parquet(T['classifications']['files']).filter(
        pl.col('classification_type').is_in(['topic', 'subfield', 'field'])).select(
        'classification_id', 'classification_type', 'name').collect()
    name = dict(zip(cls['classification_id'], cls['name']))
    up = (pl.scan_parquet(T['links_classifications_classifications']['files'])
          .filter(pl.col('relation') == 'broader').select('from_classification_id', 'to_classification_id').collect())
    parent = dict(zip(up['from_classification_id'], up['to_classification_id']))
    topic_ids = sorted((c for c, k in zip(cls['classification_id'], cls['classification_type']) if k == 'topic'),
                       key=lambda c: int(c[10:]))
    lut = np.full(max(int(c[10:]) for c in topic_ids) + 1, -1, np.int16)
    for i, c in enumerate(topic_ids):
        lut[int(c[10:])] = i
    topics = []
    for c in topic_ids:
        sub = parent.get(c); fld = parent.get(sub) if sub else None
        topics.append(dict(id=c, name=name.get(c), subfield=name.get(sub), field=name.get(fld)))
    # primary topic per paper
    topic = np.full(len(wn), -1, np.int16)
    files = T['links_content_classifications']['files']
    done = 0
    with ProcessPoolExecutor(12, mp_context=multiprocessing.get_context('spawn')) as pool:
        for w, t in pool.map(part, files, chunksize=4):
            i = np.searchsorted(wn, w)
            ok = (i < len(wn)) & (wn[np.minimum(i, len(wn) - 1)] == w)
            tt = t[ok].astype(np.int64)
            topic[i[ok]] = np.where(tt < len(lut), lut[np.minimum(tt, len(lut) - 1)], -1)
            done += 1
            if done % 200 == 0:
                print(f'{done}/{len(files)} files · {time.time() - t0:.0f}s', flush=True)
    np.save(out / 'topic.npy.tmp.npy', topic)
    (out / 'topics.json').write_text(json.dumps(topics))
    os.replace(out / 'topic.npy.tmp.npy', out / 'topic.npy')
    print(f'done: {int((topic >= 0).sum()):,} of {len(topic):,} papers have a primary topic ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
