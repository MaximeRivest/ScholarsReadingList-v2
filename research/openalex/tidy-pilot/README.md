# Native tidy-table pilot — October 6, 2026

These are **measured partition jobs**, not an estimated full-corpus completion
time. Production content files were not changed. Every job keeps its partition
in RAM and produces the four unit tables plus five link tables.

## Environment and method

- Host: lambda, 48 CPU-affinity-visible logical CPUs, about 220 GiB installed RAM.
- About 184 GiB was available when the whole-file tests began. No ancestor cgroup
  CPU quota or memory limit was found during this run.
- Pinned dependencies: Polars 1.44.2 and polars-hash 0.9.3.
- Each measurement ran in a new process, setting native threads before import.
- Timings separate read, native transforms, columnar validation, and Parquet writes.
- Peak memory is Linux process `ru_maxrss`, captured **before** optional Python
  reference comparisons. It includes native library allocations, not just Python.
- Larger/repeated runs benefit from the filesystem cache. No server cache was
  deliberately flushed. Write timings include Parquet compression and writes,
  **not a forced storage flush**; they do not prove sustained HDD throughput.
- Native tables were checked against the independent Python reference on the
  synthetic/real fixtures and prefixes sampled from these sources. All nine
  tables match exactly on those comparisons. Full-file outputs also passed the
  columnar integrity checks; a full-file Python oracle comparison was not run.

## Whole files, eight threads per job

These main results use the current `openalex-import-v2` adapter, including the
conservative published-versus-hosted source-role rule. Earlier JSON reports and
the thread comparison below retain the initial `openalex-import-v1` measurements.
The current reports also record adapter method and a source-code hash.

| Input | Works | Rows across nine output tables | Read | Transform | Validate | Write | Peak RAM | Output Parquet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `updated_date=2026-08-13/part_0012.parquet` | 301,168 | 6,175,211 | 2.12 s | 30.67 s | 6.29 s | 3.66 s | 7.62 GiB | 544 MiB |
| `updated_date=2026-09-23/part_0049.parquet` | 278,801 | 21,535,899 | 1.08 s | 74.70 s | 19.52 s | 13.65 s | 16.98 GiB | 2,032 MiB |

The first file contains relatively sparse/repetitive metadata, the second much
richer authorship/reference data. Similar input row counts do **not** imply
similar resource needs.

### Two whole-file jobs at once

Same two files, eight threads each:

- **579,969 works in 112.9 seconds** including process startup and both jobs.
- Sampled combined peak RSS: **19.7 GiB**, sampled every 0.25 seconds; sampling
  can miss brief peaks. Individual worker high-water marks are also recorded.
- The separate baseline stages total about 151.7 seconds, so this particular pair
  improved throughput by about **1.34×**, not 2×. Jobs have different durations;
  the sparse one finishes while the richer one is still running.
- Benchmark guardrail: 64 GiB combined process RSS, plus a minimum available-RAM
  reserve. This is protection for this measured experiment, not a global pipeline
  cap or a claim that every source file fits the same budget.

Results: `paired-20261006-152926-summary.json` and its two worker reports.
Reproduce with the timestamped-output script:

```sh
uv run --with-requirements pipeline/tidy/requirements.txt python research/openalex/tidy-pilot/run_concurrent.py
```

## Thread comparison, identical 25,000-work slice

Recent file above, starting at row 50,000; all nine outputs written:

| Threads | Total read + transform + validate + write | Peak RAM |
|---:|---:|---:|
| 4, first run | 13.87 s | 8.30 GiB |
| 4, repeat | 13.08 s | 8.25 GiB |
| 8 | 9.14 s | 2.08 GiB |
| 16 | 9.66 s | 2.06 GiB |

Eight threads is the starting candidate from this test. It is not proven optimal
for all files. The four-thread memory difference repeated and is retained in the
results rather than averaged away; we have not attributed it to a particular
allocator/operator. Do not assume memory changes monotonically with thread count.

A 100,000-work slice of the same file (offset 100,000) took 37.28 seconds including
read, validation, and writes, with a 6.32 GiB peak. That confirms useful-sized
in-memory partitions, not just tiny fixtures.

## Decisions and trade-offs

1. **Whole files are viable where measured.** Both tested full files fit easily
   on this server. Keep this simple starting partition; subdivide only when the
   observed costs of a particular file require it.
2. **Start with measured concurrency.** Two eight-thread jobs were tested. More
   concurrency is plausible given RAM headroom, but cannot be justified by linear
   extrapolation alone. Test larger/heavier files and additional concurrency
   before fixing the production configuration.
3. **Keep provenance.** Per-work source assertions make links larger than a bare
   pair of IDs. The richer file alone writes about 2 GiB. We did not drop evidence
   or silently change row definitions to make the benchmark smaller.
4. **Stable native keys need a dependency.** polars-hash supplies a versioned row
   encoder and native SHA-256. This replaces per-output-row Python hashing and
   avoids relying on Polars' non-portable built-in row hash. Prototype IDs are
   versioned as schema v2; existing production content IDs were not changed.
5. **Fingerprint complete input records.** Source revisions include a canonical
   source schema and row encoding. This reads/hashes fields not yet exported,
   costing time, but retains an unambiguous source revision. Preserve the source
   schema when slicing; do not infer a different schema independently per batch.
6. **Don't sort all output rows merely for display.** Native row order is
   unspecified; IDs and logical results are deterministic. Tests sort by keys
   when comparing. Sorting can be added to a published view if useful.
7. **Aggregate diagnostics.** Native diagnostics count affected records and keep
   a few examples rather than constructing millions of Python warning strings.
   Exact raw input and source pointers remain available for investigation.
8. **Don't overstate source roles.** Only recognized publication venues with
   supporting version/publication flags get `published_in`. Unknown venues,
   repositories, and unconfirmed/conflicting versions get `hosted_by`; conflicting
   flags are reported. No location link is dropped merely for this uncertainty.
9. **Scope remains explicit.** The new extractor covers core content metadata,
   signatures/entities, topics/hierarchy/keywords/sources, and citations. Remaining
   publisher-style columns and legacy concept/MeSH/SDG adapters are not implemented
   yet. An external-reference count means outside this partition, not necessarily
   absent from the entire OpenAlex snapshot.

## Reproduce one job

```sh
uv run --with-requirements pipeline/tidy/requirements.txt python -m pipeline.tidy.pilot \
  /mnt/bulk/openalex/works/updated_date=2026-09-23/part_0049.parquet \
  --rows 25000 --offset 50000 --threads 8 --compare-reference 200 \
  --write-dir /mnt/bulk/openalex/_tidy_pilot/a-new-unused-directory
```

The output directory must not exist. Per-table Parquet outputs are scratch data,
not a published dataset. Existing pilot outputs are under
`/mnt/bulk/openalex/_tidy_pilot/`; JSON measurements stay in this directory.

## Before a full run

- Compare more source-size/density extremes and realistic cold/sustained I/O.
- Implement cross-partition entity/classification reconciliation and latest-work
  selection. Local uniqueness is not global uniqueness.
- Carry over remaining desired content metadata and explicitly selected schemes.
- Add resumable partition state, update invalidation, and consistent publication.
- Estimate final space and elapsed time from a broader sample, including reduction
  and indexing costs. The content-only pipeline's old runtime is not an estimate
  for these much larger linked outputs.
