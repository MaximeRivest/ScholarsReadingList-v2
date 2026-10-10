# OpenAlex English title + abstract backfill

## Open the live dashboard

- Chattering artifact: **Live OpenAlex embedding progress**, folder `dashboard/`.
- Standalone (private tailnet): **https://lambda.tail69222b.ts.net:18797/**.

It refreshes every five seconds and shows committed records, rate, ETA, the current
publication date, progress per year, disk space and boot/retry status. It reads a live
API, not a frozen conversation snapshot. It is read-only; the optional "Ask to pause"
button drafts a request in Chattering rather than exposing an unauthenticated control API.

## Scope and model

- **203,486,035** rows with `language == 'en'` and a non-empty abstract.
- References are **not required**. No additional publication-type, open-access or title filter.
- English is the table's language label, not a new language-detection pass.
- `lightonai/DenseOn`, pinned commit `cb9947ebccb33862d24e3c7ca2edb25e51acd887`.
- One normalized, 768-dimensional vector per work.
- Input: `(title + ' ' + abstract).strip()`, with the model's `document: ` prompt.
- **512-token cap**, including prompt/special tokens. Long abstracts are truncated,
  not chunked; the original content table stays unchanged.
- BF16 GPU inference, float32 normalization, float16 storage. Ordinary float16
  inference was rejected because it produced NaNs in the earlier benchmark.
- About **312.55 GB** of vectors; about **317.44 GB** with IDs and text hashes,
  before filesystem headers and search-index construction.

The old 84M `is_complete` plan is retained but is **not** the active backfill.
The all-language expansion was superseded by the English-only clarification.

## Newest first

The pinned plan groups eligible records by their actual **publication date**, then
orders those buckets globally from newest to oldest—even when a source year folder
and a record's date disagree. Missing/invalid dates go last. Future dates supplied
by OpenAlex are preserved, not silently rewritten; the dashboard flags them.

Native Parquet `date32` values and string dates are both supported. Each date bucket
is split into shards of at most 10,000 records. Small date buckets may make smaller
shards. Source year folders in output paths are provenance, not the processing order.

## Automatically resumes after restart

These are persistent Home Manager units, declared in the OS repository and enabled
at boot on lambda:

- `openalex-embeddings.service`: planner and embedding worker.
- `openalex-embedding-progress.service`: read-only progress server.

Lambda's `users.users.maxime.linger = true` starts user services without an interactive
login. Both OS hosts were deployed/smoke-checked; the backfill runs only on lambda.
Source: `~/Projects/os/machines/hm/openalex-embeddings.nix`.

The supervisor waits for both data mounts, gives existing GPU services three minutes
to settle after boot, then resumes the pinned plan/checkpoints. Planning checkpoints
completed source files; an interrupted in-flight source file is recounted. Embedding
checkpoints completed shards; an interrupted uncommitted shard is retried.

Temporary failures retry after 60 seconds. Input/model-integrity errors exit with
status 78 and require attention rather than retrying forever. A manual pause writes
a durable `PAUSED` marker, so that pause also survives reboot.

**No reboot is performed just to test this setup.** Tests cover new-process checkpoint
reuse, date ordering, integrity failures and planning-cache resume. Service enablement
and user lingering can be checked with the commands below.

## Controls

From this directory:

```sh
bash service.sh status
bash service.sh pause        # stays paused after reboot
bash service.sh start        # removes pause marker and resumes
bash service.sh logs

systemctl --user is-enabled openalex-embeddings openalex-embedding-progress
loginctl show-user maxime -p Linger
```

A plain `systemctl --user stop openalex-embeddings` stops it for the current boot;
use `service.sh pause` when you want a persistent pause.

## GPU policy and expected duration

GPU **1** only, leaving GPU 0 for other model workloads. The worker waits for at least
4 GiB free and utilization <=30%, resolves the physical GPU through its UUID, and
keeps allocation headroom. It **never stops other GPU services**. They still share
compute, so throughput and their response times can vary.

Batch **64** was fastest in the exclusive-GPU test: about **399 records/second** in
a representative 20,000-row validation, including saving production-sized shards.
This implies **5.9 days** nonstop for 203.5M records, not a completion guarantee.
The dashboard's ETA uses actual progress and excludes waiting/model loading; it is
withheld until at least 10,000 records have been encoded in the current process.
See [throughput-results.md](throughput-results.md).

## Files

```
/mnt/fast/openalex-embeddings/denseon-english-abstracts/
  manifest.json                         # immutable model/input/selection/order plan
  plan-summary.json                     # small public progress summary
  planning/year=2025.json                # completed-file planning checkpoint
  status.json                           # live progress, errors, per-year counts
  PAUSED                                # only present for an intentional pause
  shards/year=2025/rg-000000/part-2025-09-30-000000000/
    work_ids.npy                        # uint64 IDs: W123 -> 123
    vectors.npy                         # float16 [rows,768], normalized
    text_hashes.npy                      # uint8 [rows,16], full-input BLAKE2b-128
    commit.json                         # checksums, source slice/date, model plan ID
```

The input snapshot uses hard links at
`/mnt/bulk/openalex/embedding-inputs/denseon-english-abstracts/`, from the previously
pinned full content snapshot. Despite its historical `denseon-complete` directory
name, that source snapshot contains **all content rows**, not just complete ones.
Hard links avoid duplicating the content files, but retain old versions if the
publisher later replaces them. The content table is never modified.

## Integrity

- One writer lock covers planning or encoding for this output.
- Model revision, dependency versions, source signatures and Parquet footer hashes
  are pinned. In-place source changes are rejected.
- A shard is published only after arrays/checksums are flushed and its directory is
  atomically renamed. Finished shards are checksum-checked before reuse.
- Invalid IDs, changed counts, wrong dimensions, NaN/Inf or zero vectors fail loudly.
- Completion requires the expected row count and a global duplicate-ID audit.
- Truncation frequency is estimated using a stable 0.1% ID-hash sample.
- Text hashes cover the full title/abstract, even beyond the model's token cap.

```sh
uv run python embed_content.py verify \
  --output /mnt/fast/openalex-embeddings/denseon-english-abstracts
uv run pytest -q
```

The 64-record CPU pilot lives separately at
`/mnt/fast/openalex-embeddings/validation-denseon/`; its vectors are not mixed into
production output. Changes of scope/model require a fresh plan rather than changing
an existing manifest. Other experiments should use the Python CLI directly; the
persistent service and dashboard manage the one approved English backfill.
