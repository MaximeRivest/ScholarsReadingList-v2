# DenseOn: dedicated RTX 3090 throughput

Refreshed 2026-10-05T15:15:04.710508+00:00.

Run: `/mnt/fast/openalex-embeddings/denseon-throughput/runs/20261005T145334Z`.

- Measurement state: **complete**.
- Test cleanup restored services: **True**. Subsequent shared-session changes are noted below.
- Current check: speech, semantic search and Qwen embeddings are ready. The home demo containers were subsequently removed by the parallel H100 deployment (conversation 01a10c50, 2026-10-05); they have not been recreated by this test.
- GPU 0 was not stopped. No production embedding backfill was started by this test.

## Workload

- Target population: **203,486,035** rows with `language == en` and a non-empty abstract; references optional.
- Model: pinned DenseOn, 768 dimensions; BF16 GPU inference, float32 normalization, float16 stored vectors.
- Same 512-token cap and document prompt as the prepared backfill; no lower-precision inference shortcut.
- 20,000 English records sampled through 40 probability-proportional-to-size row-group draws, then uniform raw rows, filtering English abstracts afterward. Cluster sampling can differ from the full population.
- Sample: mean **184.9 capped tokens**, median **149**; **5.57%** exceed the cap.
- Trials time reading/preparation, encoding, normalization, IDs/text hashes, array writes, checksums, and synced atomic commits. Model loading/warm-up is outside throughput timing.

## Batch-size sweep

| Requested batch | Actual batch | Records | Records/sec, full measured path | Encoding-only records/sec | GPU busy average | Peak tensor memory (GiB) | Min cosine vs batch 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 64 | 4,096 | 386.5 | 389.1 | 91.7% | 1.08 | 1.000000 |
| 128 | 128 | 4,096 | 361.3 | 363.2 | 98.1% | 1.86 | 0.999341 |
| 256 | 256 | 4,096 | 330.2 | 332.0 | 90.9% | 3.43 | 0.999341 |
| 512 | 512 | 4,096 | 309.0 | 310.7 | 92.3% | 6.56 | 0.999341 |
| 64 | 64 | 4,096 | 396.0 | 398.4 | 91.4% | 1.08 | 1.000000 |

The 64-record batch is repeated at the end to expose order/thermal effects. Device memory reported by NVIDIA after larger batches includes cached allocations; active tensor peaks are shown above.

## Sustained validation

- **20,000 records** in **50.2 seconds**: **398.6 records/sec**.
- Actual batch size: **64**, with production-sized 10,000-record output shards.
- GPU busy average: **92.9%**; average board power **277 W** (280 W limit).
- Straight-line encoding estimate for 203,486,035 records: **5.91 days**, continuously on one card. Not a completion guarantee; excludes interruptions and future index construction.

### Original HDD source to saved vectors

- **32,768 records**, **65.9 seconds**, **497.3 records/sec**.
- Reading/filtering the source row group took **0.41 seconds**.
- This is one source group, not an independent corpus-wide rate estimate. Its length distribution differs; OS cache may be warm after sample preparation.

## Interpretation and limits

- Larger batches were tested, not assumed better. Extra padding and batching overhead can outweigh higher GPU occupancy.
- A high GPU-utilization percentage is not itself the goal; completed records per second is.
- The earlier ~263 records/sec measurement used a different 1M sample and encoding path. Do not attribute the entire difference to freeing GPU memory.
- Input sampling, abstract length, CPU contention, power/thermal behavior and long-run interruptions affect the real completion date.
- Float16 **storage** is distinct from float16 **inference**: inference stays BF16 to avoid the NaNs observed previously.

## Operational notes

- The first attempt stopped during service shutdown because Podman reported a conmon error after a successful container stop. The retry checks actual stopped state.
- Restoring all services exposed a Parakeet startup memory spike. Both Parakeet loaders now deserialize the checkpoint on CPU before the existing GPU move, avoiding duplicated GPU weights without changing the model or precision.
- Related changes: `/home/maxime/Projects/parakeet-stream/parakeet_stream/parakeet.py` and `transcriber.py`. Existing unrelated changes in that repository were left alone.
- Fixed-memory vLLM servers are restored before speech/search caches. Cleanup and ExecStopPost restore only services that were active before the test.
- Container supervisors must run in an independent systemd scope, not the temporary benchmark cgroup; the restoration helper now does this so the demo survives benchmark exit.

## Raw measurements

[Raw timings, GPU samples summary and restoration state](throughput-raw.json)
