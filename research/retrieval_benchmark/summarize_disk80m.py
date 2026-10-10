"""Report the 80M-entry capacity prototype without implying retrieval accuracy."""
import json
from datetime import datetime, timezone
from pathlib import Path

from bench.disk80m import ROOT

OUT = Path(__file__).parent / 'disk80m-results.md'
manifest = json.loads((ROOT/'manifest.json').read_text()) if (ROOT/'manifest.json').exists() else None
results = []
for f in sorted((ROOT/'results').glob('*.json')):
    if f.name == 'status.json':
        continue
    results.append({'file': str(f), **json.loads(f.read_text())})
lines = ['# 80 million vectors on SSD with 4 GiB RAM', '',
         f'Refreshed {datetime.now(timezone.utc).isoformat()}.', '',
         '**This is a capacity and speed simulation, not search over 80 million distinct papers.**', '',
         'It physically stores repeated 8-bit vector codes from our 1.2M-paper DenseOn index, with new synthetic IDs. Coarse regions are synthetic perturbations of the original centroids. No accuracy or recall result at 80M is established.', '',
         '## Architecture', '',
         '- A small in-RAM HNSW graph routes queries to selected regions.',
         '- FAISS scans the selected lists of 8-bit vectors through mmap on a WD SN850X NVMe SSD.',
         '- 65,536 balanced lists; 768 dimensions; top 100 results; 4 CPU cores.',
         '- Serving workers have a hard **4 GiB total cgroup limit** with swap disabled; the host has more RAM but workers cannot use it for their newly faulted index pages.',
         '- Query set: 128 scientific benchmark queries, 32 per dataset; sequential requests, no concurrent-user load test.', '',
         '## Physical storage', '']
if manifest:
    lines += [f"- Entries: **{manifest['entries']:,}**.",
              f"- Posting-list file: **{manifest['postings_bytes']/1e9:.2f} GB**.",
              f"- Allocated posting-list blocks: **{manifest['allocated_postings_bytes']/1e9:.2f} GB** (not a sparse pretend-sized file).",
              f"- Routing/index metadata: **{manifest['metadata_bytes']/1e6:.1f} MB**."]
else:
    lines.append('Build pending.')
lines += ['', '## Measured serving times', '',
          'Each run starts with the prototype files evicted from the OS page cache. The reported distribution includes its cache warming during real queries. All configurations use the same query set.', '',
          '| Workload / prefetch | Lists searched | Approx. vectors scanned/query | Encoding included? | Median ms | p95 ms | Disk read MB/query | Peak RAM (GiB) |',
          '|---|---:|---:|---|---:|---:|---:|---:|']
for r in results:
    if 'error' in r:
        lines.append(f"| {Path(r['file']).stem}: {r['error']} | — | — | — | — | — | — | — |")
        continue
    lat = r['total_latency']
    policy = 'prefetch on' if r.get('prefetch') else 'prefetch off'
    lines.append(f"| {r['workload']}, {policy} | {r['nprobe']} | {r['entries']/r['lists']*r['nprobe']:,.0f} | {'Yes' if r['includes_query_encoding'] else 'No'} | {lat['median_ms']:.1f} | {lat['p95_ms']:.1f} | {r['mean_disk_MB_per_query']:.1f} | {r['memory_peak_bytes']/2**30:.2f} |")
if not results:
    lines.append('| Pending | — | — | — | — | — | — | — |')
lines += ['', '## How to interpret this', '',
          '- **Routed:** real query vectors traverse the HNSW router, then scan lists of synthetic/repeated entries.',
          '- **Dispersed:** the router is still timed, but its choices are replaced by seeded random lists spread across the file. This checks how much favorable locality helps the routed workload.',
          '- **Encoding included:** actually runs DenseOn on the CPU for each question under the same 4 GiB budget; not a sum of separately measured percentiles.',
          '- The split router/scanner prototype first ran without prefetch. A follow-up enables the prefetch stage normally called by FAISS IndexIVF.search. Search-only timings were similar (108 vs 112 ms median at 64 lists); do not attribute the full-request timing difference to a demonstrated prefetch speedup.',
          '- The two full-request runs were 177–202 ms median and 272–418 ms p95. Encoding variability accounts for much of that range; neither a latency guarantee nor a statistically established comparison is claimed.',
          '- Model initialization and index loading are outside request latency, but happen inside the memory-limited worker. HTTP, metadata lookup, result text fetching and network travel time are not included.',
          '- Build used up to 12 GiB; the 4 GiB limit applies to serving, including the model in the encoding-enabled runs.', '',
          '## Limits and next validation', '',
          'This can establish that a particular amount of SSD scanning fits the memory budget and takes a measured amount of time. It cannot establish which amount of scanning will retain the required accuracy on 80M distinct papers.', '',
          'Repeated vectors and balanced lists can make top-k selection, routing, cache locality and load balance more favorable than a real corpus. The dispersed run is an I/O control, not a solution to that accuracy limitation. A real 10M/80M corpus test remains necessary before committing to hardware or a latency guarantee.', '',
          'The full compression benchmark was paused during this experiment to avoid SSD contention, and is automatically resumed when the experiment exits. Other user services are not stopped.', '',
          '## Reproduce and raw data', '',
          '- `run_disk80m.sh` builds and runs the experiment.',
          '- Code: `bench/disk80m.py`; compact mmap layout is covered by `tests/test_disk80m.py`.',
          f'- Source index and generated files: `{ROOT}`.',
          '- [Full raw records, including individual request timings](disk80m-raw.json).', '']
OUT.write_text('\n'.join(lines))
OUT.with_name('disk80m-raw.json').write_text(json.dumps({'manifest': manifest, 'measurements': results}, indent=2)+'\n')
print(OUT)
