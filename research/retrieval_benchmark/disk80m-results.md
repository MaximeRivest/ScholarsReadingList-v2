# 80 million vectors on SSD with 4 GiB RAM

Refreshed 2026-10-05T12:33:56.016389+00:00.

**This is a capacity and speed simulation, not search over 80 million distinct papers.**

It physically stores repeated 8-bit vector codes from our 1.2M-paper DenseOn index, with new synthetic IDs. Coarse regions are synthetic perturbations of the original centroids. No accuracy or recall result at 80M is established.

## Architecture

- A small in-RAM HNSW graph routes queries to selected regions.
- FAISS scans the selected lists of 8-bit vectors through mmap on a WD SN850X NVMe SSD.
- 65,536 balanced lists; 768 dimensions; top 100 results; 4 CPU cores.
- Serving workers have a hard **4 GiB total cgroup limit** with swap disabled; the host has more RAM but workers cannot use it for their newly faulted index pages.
- Query set: 128 scientific benchmark queries, 32 per dataset; sequential requests, no concurrent-user load test.

## Physical storage

- Entries: **80,000,000**.
- Posting-list file: **62.08 GB**.
- Allocated posting-list blocks: **62.08 GB** (not a sparse pretend-sized file).
- Routing/index metadata: **212.4 MB**.

## Measured serving times

Each run starts with the prototype files evicted from the OS page cache. The reported distribution includes its cache warming during real queries. All configurations use the same query set.

| Workload / prefetch | Lists searched | Approx. vectors scanned/query | Encoding included? | Median ms | p95 ms | Disk read MB/query | Peak RAM (GiB) |
|---|---:|---:|---|---:|---:|---:|---:|
| routed, prefetch off | 16 | 19,531 | No | 30.6 | 46.9 | 63.3 | 4.00 |
| routed, prefetch off | 256 | 312,500 | No | 410.0 | 458.8 | 954.1 | 4.00 |
| dispersed, prefetch off | 64 | 78,125 | No | 123.6 | 150.0 | 265.6 | 4.00 |
| routed, prefetch on | 64 | 78,125 | Yes | 177.2 | 271.8 | 286.1 | 4.00 |
| routed, prefetch off | 64 | 78,125 | Yes | 201.9 | 417.9 | 260.0 | 4.00 |
| routed, prefetch on | 64 | 78,125 | No | 112.4 | 131.9 | 278.6 | 4.00 |
| routed, prefetch off | 64 | 78,125 | No | 107.9 | 127.0 | 241.2 | 4.00 |

## How to interpret this

- **Routed:** real query vectors traverse the HNSW router, then scan lists of synthetic/repeated entries.
- **Dispersed:** the router is still timed, but its choices are replaced by seeded random lists spread across the file. This checks how much favorable locality helps the routed workload.
- **Encoding included:** actually runs DenseOn on the CPU for each question under the same 4 GiB budget; not a sum of separately measured percentiles.
- The split router/scanner prototype first ran without prefetch. A follow-up enables the prefetch stage normally called by FAISS IndexIVF.search. Search-only timings were similar (108 vs 112 ms median at 64 lists); do not attribute the full-request timing difference to a demonstrated prefetch speedup.
- The two full-request runs were 177–202 ms median and 272–418 ms p95. Encoding variability accounts for much of that range; neither a latency guarantee nor a statistically established comparison is claimed.
- Model initialization and index loading are outside request latency, but happen inside the memory-limited worker. HTTP, metadata lookup, result text fetching and network travel time are not included.
- Build used up to 12 GiB; the 4 GiB limit applies to serving, including the model in the encoding-enabled runs.

## Limits and next validation

This can establish that a particular amount of SSD scanning fits the memory budget and takes a measured amount of time. It cannot establish which amount of scanning will retain the required accuracy on 80M distinct papers.

Repeated vectors and balanced lists can make top-k selection, routing, cache locality and load balance more favorable than a real corpus. The dispersed run is an I/O control, not a solution to that accuracy limitation. A real 10M/80M corpus test remains necessary before committing to hardware or a latency guarantee.

The full compression benchmark was paused during this experiment to avoid SSD contention, and is automatically resumed when the experiment exits. Other user services are not stopped.

## Reproduce and raw data

- `run_disk80m.sh` builds and runs the experiment.
- Code: `bench/disk80m.py`; compact mmap layout is covered by `tests/test_disk80m.py`.
- Source index and generated files: `/mnt/fast/scholarsreadinglist/retrieval-bench/disk80m`.
- [Full raw records, including individual request timings](disk80m-raw.json).
