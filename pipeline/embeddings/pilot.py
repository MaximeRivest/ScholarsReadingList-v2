"""Small real-data CPU pilot in a separate output; never mix it into GPU production."""
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

import embed_content as e

master=Path('/mnt/fast/openalex-embeddings/denseon-complete')
doc=e.read_manifest(master)
root=Path('/mnt/fast/openalex-embeddings/validation-denseon')
source=root/'source/year=2025/data.parquet'
source.parent.mkdir(parents=True,exist_ok=True)
if not source.exists():
    entry=next(x for x in doc['files'] if x['relative_path']=='year=2025/data.parquet')
    pf=pq.ParquetFile(e.check_source(entry))
    tables=[]
    # Take a spread across the year rather than just one contiguous abstract group.
    import pyarrow as pa
    for group in np.linspace(0,pf.metadata.num_row_groups-1,8,dtype=int):
        t=pf.read_row_group(int(group),columns=['id','title','abstract','is_complete'],use_threads=False)
        t=t.filter(pc.fill_null(t['is_complete'],False))
        tables.append(t.slice(0,8))
    pq.write_table(pa.concat_tables(tables),source,row_group_size=16)
if not (root/'output/manifest.json').exists():
    e.plan(root/'source',root/'snapshot',root/'output',doc['embedding']['revision'],shard_rows=16)
t0=time.monotonic()
e.run(root/'output',device='cpu',batch_size=8,cpu_threads=4)
assert e.verify(root/'output')['complete']
first_elapsed=time.monotonic()-t0
# Second run must be a no-op: the implementation exits before trying to load a model.
def forbidden_model(*a,**kw):
    raise AssertionError('Resume tried to re-encode a committed shard')
e.run(root/'output',device='cpu',encoder_factory=forbidden_model)
report={'rows':e.read_manifest(root/'output')['expected_rows'], 'device':'cpu',
        'first_run_seconds':first_elapsed, 'checksum_verification':'passed',
        'resume_without_reencoding':'passed', 'production_output_untouched':str(master),
        'model_revision':doc['embedding']['revision']}
e.atomic_json(root/'pilot-result.json',report)
print(json.dumps(report,indent=2))
