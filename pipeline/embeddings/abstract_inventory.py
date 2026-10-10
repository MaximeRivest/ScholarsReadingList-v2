"""Bounded-memory count of non-empty abstracts and language coverage.

Uses the already pinned content snapshot; does not read the references column or
collect the whole table into memory. Does not start an embedding job.
"""
from collections import Counter
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import embed_content as e

OUT=Path('/mnt/fast/openalex-embeddings/abstract-inventory.json')
doc=e.read_manifest(Path('/mnt/fast/openalex-embeddings/denseon-complete'))
started=time.monotonic()
rows=nonempty=old_complete=empty_or_missing=0
languages=Counter()
files=[]
for i,entry in enumerate(doc['files']):
    path=e.check_source(entry)
    count=0
    for b in pq.ParquetFile(path).iter_batches(batch_size=32768,
            columns=['abstract','language','is_complete'],use_threads=False):
        mask=pc.fill_null(pc.not_equal(pc.utf8_trim_whitespace(b.column('abstract')),''),False)
        n=int(pc.sum(pc.cast(mask,pa.int64())).as_py() or 0)
        rows+=len(b);nonempty+=n;count+=n;empty_or_missing+=len(b)-n
        old_complete+=int(pc.sum(pc.cast(pc.fill_null(b.column('is_complete'),False),pa.int64())).as_py() or 0)
        for item in pc.value_counts(pc.filter(b.column('language'),mask)).to_pylist():
            languages[item['values'] or 'unknown']+=item['counts']
    e.check_source(entry)
    files.append({'relative_path':entry['relative_path'],'nonempty_abstracts':count})
    if (i+1)%20==0:
        e.atomic_json(OUT,{'state':'running','files_completed':i+1,'files_total':len(doc['files']),
                          'rows_scanned':rows,'nonempty_abstracts':nonempty,
                          'languages':dict(languages.most_common()),'updated_utc':e.now()})
        print(f'{i+1}/{len(doc["files"])} files; {nonempty:,} non-empty abstracts',flush=True)
assert rows==sum(g['source_rows'] for f in doc['files'] for g in f['row_groups'])
report={'state':'complete','updated_utc':e.now(),'input_plan_id':doc['plan_id'],
        'rows_scanned':rows,'previous_complete_rows':old_complete,'nonempty_abstracts':nonempty,
        'empty_or_missing_abstracts':empty_or_missing,'languages':dict(languages.most_common()),
        'files':files,'seconds':time.monotonic()-started,
        'float16_vector_bytes_at_768_dimensions':nonempty*768*2,
        'id_and_hash_bytes':nonempty*(8+16)}
e.atomic_json(OUT,report)
print({k:v for k,v in report.items() if k not in ('files',)},flush=True)
