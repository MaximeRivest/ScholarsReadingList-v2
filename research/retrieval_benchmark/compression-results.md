# Compression benchmarks: title + abstract retrieval

Report refreshed 2026-10-05T12:33:57.996678+00:00. Corpus: **1,205,228 works**.

## Status

- Pooling and exhaustive quality measurements: complete.
- All nine WARP index builds and two dense index builds: complete.
- Released WARP cannot **search** its 1-bit indexes. A successful build is not proof of usable search.
- The October 4 speed sweep stopped early; its constrained-memory failures are not speed measurements.
- Revised speed sweep: fixed total memory budget, separate NVMe and spinning-HDD measurements. See completed records below; missing records remain pending.

## Quality and raw vector storage

Query vectors stay full precision. The 1-bit quality test uses document **sign bits**; this is different from WARP’s centroid-plus-quantized-residual encoding.

**Payload is not total search-index storage.** It excludes document offsets, identifiers, candidate indexes and engine metadata. One-bit payload sizes below are computed from vector counts, not sizes of a deployed packed-bit search engine.

| Representation | Vector payload (GB) | Mean nDCG@10 | SciFact | NFCorpus | SciDocs | TREC-COVID |
|---|---:|---:|---:|---:|---:|---:|
| LateOn, 16-bit | 54.945 | 0.4294 | 0.6123 | 0.1644 | 0.1387 | 0.8023 |
| LateOn, 1-bit | 3.434 | 0.4106 | 0.5960 | 0.1736 | 0.1114 | 0.7614 |
| LateOn, 2× pooling, 16-bit | 27.550 | 0.4251 | 0.6000 | 0.1687 | 0.1330 | 0.7987 |
| LateOn, 2× pooling, 1-bit | 1.722 | 0.4026 | 0.5798 | 0.1729 | 0.1089 | 0.7488 |
| LateOn, 4× pooling, 16-bit | 13.851 | 0.4246 | 0.5960 | 0.1758 | 0.1304 | 0.7961 |
| LateOn, 4× pooling, 1-bit | 0.866 | 0.4049 | 0.5746 | 0.1712 | 0.1062 | 0.7675 |
| DenseOn, 16-bit | 1.851 | 0.4304 | 0.6030 | 0.1720 | 0.1411 | 0.8052 |
| DenseOn, 1-bit | 0.116 | 0.4052 | 0.5842 | 0.1591 | 0.1326 | 0.7449 |

**4× pooling plus sign bits:** 63.5× smaller vector payload; mean score 0.4294 → 0.4049 (5.7% relative reduction).

The loss is not uniform: related-paper retrieval (SciDocs) falls more than the average. This is a general-purpose LateOn title/abstract experiment, not a reproduction of the medical fine-tuned model in the social-media chart.

## Completed patched-engine speed measurements

WARP uses the locally patched **header-only loader** documented in `patches/README.md`. Scoring is unchanged and 2,000 ranked results were checked against the released wheel. The normal installed wheel is intact.

4 CPU cores, sequential queries, top 100. Query encoding is **excluded**. RAM and NVMe use matching query IDs; HDD uses a smaller subset, so compare matched `query_timings` records rather than blindly comparing aggregate percentiles. Memory-limited runs allow **2 GiB total**, including program memory and file cache—not 2 GiB dedicated to the index.

| Configuration | Device / mode | Search files (GB) | Mean quality | Median ms | p95 ms | Disk read MB/query | Cached at end |
|---|---|---:|---:|---:|---:|---:|---:|
| denseon-ivfsq8-np16 | NVMe: mmap / 2 GiB limit | 0.95 | — | 2.0 | 4.2 | 1.41 | 99% |
| denseon-ivfsq8-np64 | NVMe: mmap / 2 GiB limit | 0.95 | — | 4.8 | 5.8 | 0.49 | 100% |
| lateon-pool1-2bit-np32 | NVMe: mmap / 2 GiB limit | 8.62 | — | 860.3 | 1658.6 | 2310.30 | 14% |
| lateon-pool1-2bit-npauto | NVMe: mmap / 2 GiB limit | 8.62 | — | 184.5 | 376.1 | 456.62 | 14% |
| lateon-pool1-4bit-np32 | NVMe: mmap / 2 GiB limit | 15.49 | — | 925.5 | 1724.7 | 2446.72 | 7% |
| lateon-pool1-4bit-npauto | NVMe: mmap / 2 GiB limit | 15.49 | — | 195.7 | 407.5 | 490.57 | 7% |
| lateon-pool2-2bit-np32 | NVMe: mmap / 2 GiB limit | 4.34 | — | 557.7 | 1092.8 | 1487.36 | 31% |
| lateon-pool2-2bit-npauto | NVMe: mmap / 2 GiB limit | 4.34 | — | 129.0 | 251.8 | 283.80 | 25% |
| lateon-pool2-4bit-np32 | NVMe: mmap / 2 GiB limit | 7.78 | — | 652.2 | 1319.2 | 1822.03 | 16% |
| lateon-pool2-4bit-npauto | NVMe: mmap / 2 GiB limit | 7.78 | — | 152.5 | 290.1 | 344.16 | 13% |
| lateon-pool4-2bit-np32 | NVMe: mmap / 2 GiB limit | 2.18 | — | 207.1 | 451.4 | 461.58 | 58% |
| lateon-pool4-2bit-npauto | NVMe: mmap / 2 GiB limit | 2.18 | — | 52.8 | 106.1 | 83.32 | 60% |
| lateon-pool4-4bit-np32 | NVMe: mmap / 2 GiB limit | 3.91 | — | 392.9 | 780.2 | 1017.67 | 35% |
| lateon-pool4-4bit-npauto | NVMe: mmap / 2 GiB limit | 3.91 | — | 79.5 | 167.4 | 178.93 | 33% |
| denseon-hnsw32-ef128 | NVMe: RAM-resident | 4.03 | 0.4278 | 0.9 | 1.2 | 0.00 | 100% |
| denseon-ivfsq8-np16 | NVMe: RAM-resident | 0.95 | 0.4073 | 1.9 | 2.4 | 0.00 | 100% |
| denseon-ivfsq8-np64 | NVMe: RAM-resident | 0.95 | 0.4197 | 4.8 | 5.7 | 0.00 | 100% |
| lateon-pool1-2bit-np32 | NVMe: RAM-resident | 8.62 | 0.4022 | 127.3 | 275.9 | 0.00 | 100% |
| lateon-pool1-2bit-npauto | NVMe: RAM-resident | 8.62 | 0.3598 | 50.5 | 90.4 | 0.00 | 100% |
| lateon-pool1-4bit-np32 | NVMe: RAM-resident | 15.49 | 0.4054 | 114.8 | 256.7 | 0.04 | 100% |
| lateon-pool2-2bit-np32 | NVMe: RAM-resident | 4.34 | 0.4001 | 90.7 | 198.4 | 0.00 | 100% |
| lateon-pool2-2bit-npauto | NVMe: RAM-resident | 4.34 | 0.3439 | 40.3 | 76.0 | 0.00 | 100% |
| lateon-pool2-4bit-np32 | NVMe: RAM-resident | 7.78 | 0.4077 | 78.8 | 177.5 | 0.00 | 100% |
| lateon-pool2-4bit-npauto | NVMe: RAM-resident | 7.78 | 0.3500 | 39.7 | 73.2 | 0.00 | 100% |
| lateon-pool4-2bit-np32 | NVMe: RAM-resident | 2.18 | 0.3935 | 71.1 | 168.7 | 0.00 | 100% |
| lateon-pool4-2bit-npauto | NVMe: RAM-resident | 2.18 | 0.3431 | 26.5 | 48.8 | 0.00 | 100% |
| lateon-pool4-4bit-np32 | NVMe: RAM-resident | 3.91 | 0.4002 | 64.8 | 161.7 | 0.00 | 100% |
| lateon-pool4-4bit-npauto | NVMe: RAM-resident | 3.91 | 0.3471 | 25.1 | 46.7 | 0.00 | 100% |
| denseon-ivfsq8-np64 | HDD: mmap / 2 GiB limit | 0.95 | — | 5.3 | 51.4 | 1.67 | 99% |
| lateon-pool4-2bit-np32 | HDD: mmap / 2 GiB limit | 2.18 | — | 1741.3 | 3003.0 | 331.16 | 62% |

A nearly fully cached index with negligible disk reads is **not** a disk-bound result, even when opened with mmap. Smaller dense indexes may fit at the same memory budget where the larger multi-vector index must page data.

## Search-index quality, dataset by dataset

| Configuration | Mean | SciFact | NFCorpus | SciDocs | TREC-COVID |
|---|---:|---:|---:|---:|---:|
| denseon-hnsw32-ef128 | 0.4278 | 0.6011 | 0.1692 | 0.1354 | 0.8055 |
| denseon-ivfsq8-np16 | 0.4073 | 0.5765 | 0.1153 | 0.1347 | 0.8027 |
| denseon-ivfsq8-np64 | 0.4197 | 0.5882 | 0.1432 | 0.1376 | 0.8099 |
| lateon-pool1-2bit-np32 | 0.4022 | 0.5861 | 0.1479 | 0.1213 | 0.7537 |
| lateon-pool1-2bit-npauto | 0.3598 | 0.5497 | 0.1305 | 0.1160 | 0.6430 |
| lateon-pool1-4bit-np32 | 0.4054 | 0.5965 | 0.1513 | 0.1306 | 0.7434 |
| lateon-pool2-2bit-np32 | 0.4001 | 0.5763 | 0.1562 | 0.1163 | 0.7518 |
| lateon-pool2-2bit-npauto | 0.3439 | 0.5269 | 0.1354 | 0.1076 | 0.6059 |
| lateon-pool2-4bit-np32 | 0.4077 | 0.5982 | 0.1579 | 0.1232 | 0.7516 |
| lateon-pool2-4bit-npauto | 0.3500 | 0.5486 | 0.1402 | 0.1139 | 0.5974 |
| lateon-pool4-2bit-np32 | 0.3935 | 0.5547 | 0.1554 | 0.1151 | 0.7488 |
| lateon-pool4-2bit-npauto | 0.3431 | 0.5188 | 0.1394 | 0.1050 | 0.6092 |
| lateon-pool4-4bit-np32 | 0.4002 | 0.5683 | 0.1580 | 0.1224 | 0.7521 |
| lateon-pool4-4bit-npauto | 0.3471 | 0.5279 | 0.1427 | 0.1133 | 0.6045 |

## HDD comparison on identical queries

These numbers use exactly the same 100 query IDs for all three conditions. Values are median / p95 milliseconds; query encoding excluded.

| Configuration | RAM | NVMe, 2 GiB total | HDD, 2 GiB total |
|---|---:|---:|---:|
| lateon-pool4-2bit-np32 | 69.6 / 132.7 | 204.8 / 357.5 | 1741.3 / 3003.0 |
| denseon-ivfsq8-np64 | 5.0 / 5.7 | 5.1 / 6.5 | 5.3 / 51.4 |

## Failures and unavailable configurations

- denseon-hnsw32-ef128 / disk / NVMe: index must be fully loaded in RAM (no mmap support).
- lateon-pool1-1bit-np32 / disk / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool1-1bit-npauto / disk / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool2-1bit-np32 / disk / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool2-1bit-npauto / disk / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool4-1bit-np32 / disk / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool4-1bit-npauto / disk / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool1-1bit-np32 / ram / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool1-1bit-npauto / ram / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool1-4bit-npauto / ram / NVMe: exit 1.
- lateon-pool2-1bit-np32 / ram / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool2-1bit-npauto / ram / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool4-1bit-np32 / ram / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.
- lateon-pool4-1bit-npauto / ram / NVMe: xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring.

## Limits

- OpenAlex distractors are unjudged, not proven irrelevant. Scores at this scale are incomplete-judgment scores, not a measured true quality drop. There is no established bound on that drop.
- The OpenAlex sample came from the partially downloaded snapshot and is not representative of all works.
- Searches use warm model/runtime state. Index load time is recorded separately.
- The node is shared. Memory pressure and actual I/O are recorded, but this is not an isolated dedicated-server benchmark.
- Earlier “0.5 GB cache allowance” claims were invalid: the original cap calculation missed startup allocations. Do not use those pilot timings as low-RAM results.
- Earlier 25.7 GB/million WARP storage counted both build chunks and serving files. Read-only serving needs only the packed copy plus metadata; retaining mutable build data costs more.

## Files and reproduction

- Data and machine-readable results: `/mnt/fast/scholarsreadinglist/retrieval-bench/`.
- Quality completion: `finish_compression_quality.sh`.
- Repaired speed sweep: `resume_compression_speed.sh`.
- Refresh this report: `.venv/bin/python summarize_compression.py`.
- The GPU services remain running; this speed sweep uses CPU only.

## Raw exports

[All recorded fields and individual query timings (JSON)](compression-raw.json)

[Flat results table (CSV)](compression-raw.csv)
