"""Generate an honest, refreshable report from completed compression measurements."""
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from bench.common import DATA, DATASETS

R = DATA / 'results'
OUT = Path(__file__).parent / 'compression-results.md'

def load(p):
    return json.loads(p.read_text())

rows = []
base_n = load(DATA / 'index2/warp-nb2/lateon/beir4+openalex1m/metadata.json')['num_embeddings']
base_bytes = base_n * 128 * 2
for model, label in [('lateon', 'LateOn'), ('lateon_pool2', 'LateOn, 2× pooling'), ('lateon_pool4', 'LateOn, 4× pooling'), ('denseon', 'DenseOn')]:
    n = 1205228 if model == 'denseon' else load(DATA / f'index2/warp-nb2/{model}/beir4+openalex1m/metadata.json')['num_embeddings']
    dim = 768 if model == 'denseon' else 128
    for bits in (16, 1):
        method = f"{model}{'-1bit' if bits == 1 else ''}-exact"
        paths = [R / 'beir4+openalex1m' / ds / (method + '.json') for ds in DATASETS]
        if not all(p.exists() for p in paths):
            continue
        values = [load(p)['ndcg@10'] for p in paths]
        payload = n * dim * bits / 8
        rows.append({'label': label, 'bits': bits, 'values': values, 'average': sum(values)/4, 'payload': payload})

lines = [
    '# Compression benchmarks: title + abstract retrieval', '',
    f'Report refreshed {datetime.now(timezone.utc).isoformat()}. Corpus: **1,205,228 works**.', '',
    '## Status', '',
    '- Pooling and exhaustive quality measurements: complete.',
    '- All nine WARP index builds and two dense index builds: complete.',
    '- Released WARP cannot **search** its 1-bit indexes. A successful build is not proof of usable search.',
    '- The October 4 speed sweep stopped early; its constrained-memory failures are not speed measurements.',
    '- Revised speed sweep: fixed total memory budget, separate NVMe and spinning-HDD measurements. See completed records below; missing records remain pending.', '',
    '## Quality and raw vector storage', '',
    'Query vectors stay full precision. The 1-bit quality test uses document **sign bits**; this is different from WARP’s centroid-plus-quantized-residual encoding.', '',
    '**Payload is not total search-index storage.** It excludes document offsets, identifiers, candidate indexes and engine metadata. One-bit payload sizes below are computed from vector counts, not sizes of a deployed packed-bit search engine.', '',
    '| Representation | Vector payload (GB) | Mean nDCG@10 | SciFact | NFCorpus | SciDocs | TREC-COVID |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    lines.append(f"| {r['label']}, {r['bits']}-bit | {r['payload']/1e9:.3f} | {r['average']:.4f} | " + ' | '.join(f'{v:.4f}' for v in r['values']) + ' |')
base = next(r for r in rows if r['label']=='LateOn' and r['bits']==16)
compressed = next(r for r in rows if r['label']=='LateOn, 4× pooling' and r['bits']==1)
lines += ['', f"**4× pooling plus sign bits:** {base['payload']/compressed['payload']:.1f}× smaller vector payload; mean score {base['average']:.4f} → {compressed['average']:.4f} ({100*(1-compressed['average']/base['average']):.1f}% relative reduction).",
          '', 'The loss is not uniform: related-paper retrieval (SciDocs) falls more than the average. This is a general-purpose LateOn title/abstract experiment, not a reproduction of the medical fine-tuned model in the social-media chart.', '',
          '## Completed patched-engine speed measurements', '',
          'WARP uses the locally patched **header-only loader** documented in `patches/README.md`. Scoring is unchanged and 2,000 ranked results were checked against the released wheel. The normal installed wheel is intact.', '',
          '4 CPU cores, sequential queries, top 100. Query encoding is **excluded**. RAM and NVMe use matching query IDs; HDD uses a smaller subset, so compare matched `query_timings` records rather than blindly comparing aggregate percentiles. Memory-limited runs allow **2 GiB total**, including program memory and file cache—not 2 GiB dedicated to the index.', '',
          '| Configuration | Device / mode | Search files (GB) | Mean quality | Median ms | p95 ms | Disk read MB/query | Cached at end |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
count = 0
raw_records = []
for folder, device in [('compress-v3-headerfix-nvme','NVMe'), ('compress-v3-headerfix-hdd','HDD')]:
    for path in sorted((R/folder).glob('*/*.json')):
        d=load(path)
        raw_records.append({'source_file': str(path), 'device': device, **d})
        if 'search_latency' not in d:
            continue
        count += 1
        q = d.get('ndcg@10_mean')
        lat=d['search_latency']
        mode='RAM-resident' if d['regime']=='ram' else 'mmap / 2 GiB limit'
        quality = f'{q:.4f}' if q is not None else '—'
        lines.append(f"| {d['name']} | {device}: {mode} | {d['search_bytes']/1e9:.2f} | {quality} | {lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} | {(d.get('disk_read_bytes_per_query') or 0)/1e6:.2f} | {d['cached_fraction_at_end']:.0%} |")
if not count:
    lines += ['| Pending | — | — | — | — | — | — | — |']
lines += ['', 'A nearly fully cached index with negligible disk reads is **not** a disk-bound result, even when opened with mmap. Smaller dense indexes may fit at the same memory budget where the larger multi-vector index must page data.', '',
          '## Search-index quality, dataset by dataset', '',
          '| Configuration | Mean | SciFact | NFCorpus | SciDocs | TREC-COVID |',
          '|---|---:|---:|---:|---:|---:|']
for d in raw_records:
    if d.get('ndcg@10'):
        q = d['ndcg@10']
        lines.append(f"| {d['name']} | {d['ndcg@10_mean']:.4f} | " + ' | '.join(f'{q[ds]:.4f}' for ds in DATASETS) + ' |')
lines += ['', '## HDD comparison on identical queries', '',
          'These numbers use exactly the same 100 query IDs for all three conditions. Values are median / p95 milliseconds; query encoding excluded.', '',
          '| Configuration | RAM | NVMe, 2 GiB total | HDD, 2 GiB total |',
          '|---|---:|---:|---:|']
for name in ('lateon-pool4-2bit-np32', 'denseon-ivfsq8-np64'):
    hdd = next((d for d in raw_records if d['name'] == name and d['device'] == 'HDD' and 'query_timings' in d), None)
    if hdd is None:
        continue
    keys = {(x['dataset'], x['qid']) for x in hdd['query_timings']}
    cells = []
    for device, regime in [('NVMe', 'ram'), ('NVMe', 'disk'), ('HDD', 'disk')]:
        r = next((d for d in raw_records if d['name'] == name and d['device'] == device and d['regime'] == regime and 'query_timings' in d), None)
        if r is None:
            cells.append('pending')
            continue
        times = sorted(x['milliseconds'] for x in r['query_timings'] if (x['dataset'], x['qid']) in keys)
        assert len(times) == len(keys), 'Matched query set incomplete'
        cells.append(f'{statistics.median(times):.1f} / {times[round(.95*(len(times)-1))]:.1f}')
    lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
lines += ['', '## Failures and unavailable configurations', '']
for d in raw_records:
    reason = d.get('error') or d.get('unsupported') or d.get('skipped')
    if reason:
        lines.append(f"- {d['name']} / {d['regime']} / {d['device']}: {reason}.")
lines += ['', '## Limits', '',
          '- OpenAlex distractors are unjudged, not proven irrelevant. Scores at this scale are incomplete-judgment scores, not a measured true quality drop. There is no established bound on that drop.',
          '- The OpenAlex sample came from the partially downloaded snapshot and is not representative of all works.',
          '- Searches use warm model/runtime state. Index load time is recorded separately.',
          '- The node is shared. Memory pressure and actual I/O are recorded, but this is not an isolated dedicated-server benchmark.',
          '- Earlier “0.5 GB cache allowance” claims were invalid: the original cap calculation missed startup allocations. Do not use those pilot timings as low-RAM results.',
          '- Earlier 25.7 GB/million WARP storage counted both build chunks and serving files. Read-only serving needs only the packed copy plus metadata; retaining mutable build data costs more.', '',
          '## Files and reproduction', '',
          '- Data and machine-readable results: `/mnt/fast/scholarsreadinglist/retrieval-bench/`.',
          '- Quality completion: `finish_compression_quality.sh`.',
          '- Repaired speed sweep: `resume_compression_speed.sh`.',
          '- Refresh this report: `.venv/bin/python summarize_compression.py`.',
          '- The GPU services remain running; this speed sweep uses CPU only.', '']
# Preserve all recorded fields, including per-query latency, read bytes, memory,
# cache residency, dataset scores, and failures. CSV is a flat overview.
raw_path = OUT.with_name('compression-raw.json')
raw_path.write_text(json.dumps({'refreshed_utc': datetime.now(timezone.utc).isoformat(),
                                'vector_quality': rows, 'speed_records': raw_records}, indent=2) + '\n')
columns = ['name', 'device', 'regime', 'status', 'index_bytes', 'memory_limit_bytes',
           'n_queries', 'median_ms', 'p95_ms', 'mean_ndcg10', *DATASETS,
           'read_bytes_per_query', 'anon_peak_bytes', 'cached_fraction_at_end']
with OUT.with_name('compression-raw.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for d in raw_records:
        q = d.get('ndcg@10', {})
        lat = d.get('search_latency', {})
        writer.writerow({'name': d['name'], 'device': d['device'], 'regime': d['regime'],
                         'status': d.get('error') or d.get('unsupported') or d.get('skipped') or 'measured',
                         'index_bytes': d.get('search_bytes'), 'memory_limit_bytes': d.get('memory_limit_bytes'),
                         'n_queries': lat.get('n'), 'median_ms': lat.get('p50_ms'), 'p95_ms': lat.get('p95_ms'),
                         'mean_ndcg10': d.get('ndcg@10_mean'), **{ds: q.get(ds) for ds in DATASETS},
                         'read_bytes_per_query': d.get('disk_read_bytes_per_query'),
                         'anon_peak_bytes': d.get('rss_anon_peak'), 'cached_fraction_at_end': d.get('cached_fraction_at_end')})
lines += ['## Raw exports', '', '[All recorded fields and individual query timings (JSON)](compression-raw.json)', '',
          '[Flat results table (CSV)](compression-raw.csv)', '']
OUT.write_text('\n'.join(lines))
print(OUT)
