"""Refresh the readable report and raw data from the latest exclusive-GPU test."""
from pathlib import Path
import json
from datetime import datetime, timezone

from throughput import ROOT, TOTAL

run=Path((ROOT/'latest-run.txt').read_text().strip())
result=json.loads((run/'results.json').read_text()) if (run/'results.json').exists() else {}
sample=json.loads((ROOT/'sample.json').read_text())
services=json.loads((run/'services.json').read_text()) if (run/'services.json').exists() else {}
lines=['# DenseOn: dedicated RTX 3090 throughput', '',
       f'Refreshed {datetime.now(timezone.utc).isoformat()}.', '',
       f'Run: `{run}`.', '',
       f"- Measurement state: **{result.get('state', 'in progress')}**.",
       f"- Test cleanup restored services: **{services.get('restored', False)}**. Subsequent shared-session changes are noted below.",
       '- Current check: speech, semantic search and Qwen embeddings are ready. The home demo containers were subsequently removed by the parallel H100 deployment (conversation 01a10c50, 2026-10-05); they have not been recreated by this test.',
       '- GPU 0 was not stopped. No production embedding backfill was started by this test.', '',
       '## Workload', '',
       f'- Target population: **{TOTAL:,}** rows with `language == en` and a non-empty abstract; references optional.',
       '- Model: pinned DenseOn, 768 dimensions; BF16 GPU inference, float32 normalization, float16 stored vectors.',
       '- Same 512-token cap and document prompt as the prepared backfill; no lower-precision inference shortcut.',
       '- 20,000 English records sampled through 40 probability-proportional-to-size row-group draws, then uniform raw rows, filtering English abstracts afterward. Cluster sampling can differ from the full population.',
       f"- Sample: mean **{sample['mean_capped_tokens']:.1f} capped tokens**, median **{sample['p50_capped_tokens']:.0f}**; **{100*sample['truncated_fraction']:.2f}%** exceed the cap.",
       '- Trials time reading/preparation, encoding, normalization, IDs/text hashes, array writes, checksums, and synced atomic commits. Model loading/warm-up is outside throughput timing.', '',
       '## Batch-size sweep', '',
       '| Requested batch | Actual batch | Records | Records/sec, full measured path | Encoding-only records/sec | GPU busy average | Peak tensor memory (GiB) | Min cosine vs batch 64 |',
       '|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in result.get('trials',[]):
    lines.append(f"| {r['requested_batch']} | {r['actual_batch']} | {r['rows']:,} | {r['rows_per_second']:.1f} | {r['encode_rows_per_second']:.1f} | {r.get('gpu_utilization_mean_percent',0):.1f}% | {r['tensor_memory_peak_bytes']/2**30:.2f} | {r['cosine_min_vs_batch64']:.6f} |")
lines += ['', 'The 64-record batch is repeated at the end to expose order/thermal effects. Device memory reported by NVIDIA after larger batches includes cached allocations; active tensor peaks are shown above.', '', '## Sustained validation', '']
if 'validation' in result:
    r=result['validation']
    lines += [f"- **{r['rows']:,} records** in **{r['seconds']:.1f} seconds**: **{r['rows_per_second']:.1f} records/sec**.",
              f"- Actual batch size: **{r['actual_batch']}**, with production-sized 10,000-record output shards.",
              f"- GPU busy average: **{r.get('gpu_utilization_mean_percent',0):.1f}%**; average board power **{r.get('power_mean_watts',0):.0f} W** (280 W limit).",
              f"- Straight-line encoding estimate for {TOTAL:,} records: **{r['estimated_population_days']:.2f} days**, continuously on one card. Not a completion guarantee; excludes interruptions and future index construction."]
if 'hdd_validation' in result:
    r=result['hdd_validation']
    lines += ['', '### Original HDD source to saved vectors', '',
              f"- **{r['rows']:,} records**, **{r['seconds']:.1f} seconds**, **{r['rows_per_second']:.1f} records/sec**.",
              f"- Reading/filtering the source row group took **{r['source_read_filter_seconds']:.2f} seconds**.",
              '- This is one source group, not an independent corpus-wide rate estimate. Its length distribution differs; OS cache may be warm after sample preparation.']
lines += ['', '## Interpretation and limits', '',
          '- Larger batches were tested, not assumed better. Extra padding and batching overhead can outweigh higher GPU occupancy.',
          '- A high GPU-utilization percentage is not itself the goal; completed records per second is.',
          '- The earlier ~263 records/sec measurement used a different 1M sample and encoding path. Do not attribute the entire difference to freeing GPU memory.',
          '- Input sampling, abstract length, CPU contention, power/thermal behavior and long-run interruptions affect the real completion date.',
          '- Float16 **storage** is distinct from float16 **inference**: inference stays BF16 to avoid the NaNs observed previously.', '',
          '## Operational notes', '',
          '- The first attempt stopped during service shutdown because Podman reported a conmon error after a successful container stop. The retry checks actual stopped state.',
          '- Restoring all services exposed a Parakeet startup memory spike. Both Parakeet loaders now deserialize the checkpoint on CPU before the existing GPU move, avoiding duplicated GPU weights without changing the model or precision.',
          '- Related changes: `/home/maxime/Projects/parakeet-stream/parakeet_stream/parakeet.py` and `transcriber.py`. Existing unrelated changes in that repository were left alone.',
          '- Fixed-memory vLLM servers are restored before speech/search caches. Cleanup and ExecStopPost restore only services that were active before the test.',
          '- Container supervisors must run in an independent systemd scope, not the temporary benchmark cgroup; the restoration helper now does this so the demo survives benchmark exit.', '',
          '## Raw measurements', '', '[Raw timings, GPU samples summary and restoration state](throughput-raw.json)', '']
path=Path(__file__).parent/'throughput-results.md'
path.write_text('\n'.join(lines))
path.with_name('throughput-raw.json').write_text(json.dumps({'sample':sample,'results':result,'services':services},indent=2)+'\n')
print(path)
