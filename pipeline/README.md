# Tidy OpenAlex pipeline

Turns the downloaded OpenAlex works snapshot (`/mnt/bulk/openalex/works`) into
eleven flat tables: four unit tables and seven link tables. The model is
described in [SCHEMA.md](SCHEMA.md).

## Read the published tables

```python
from pipeline.tidy.dataset import scan, manifest
papers  = scan('content').filter(pl.col('is_complete'), pl.col('type') == 'article')
authors = scan('signatures')                 # one row per author–affiliation pair
manifest()['rows']                           # row counts of the CURRENT generation
```

`scan()` reads the files listed in the CURRENT generation's manifest. Do not
glob `/mnt/bulk/openalex/tidy/extract`: it also holds in-progress and outdated
outputs.

Three tables exceed 2 billion rows (`links_content_classifications` 5.3 B,
`links_content_content` 3.1 B, `links_signatures_entities` 2.0 B). Standard
Polars cannot hold more than ~4.3 billion rows in one frame: filter or
aggregate before collecting, and use `.collect(engine='streaming')` for
whole-table passes.

## Run or update

All commands run from the repository root:

```sh
R="uv run --with-requirements pipeline/tidy/requirements.txt python -m pipeline.tidy.run"
$R plan        # resources, inventory, nothing changed
$R all         # versions -> extract -> reduce -> publish (resumable)
$R status      # progress
$R gc          # report outputs no longer used by the newest 2 generations (--yes deletes)
```

For a long run, start it as a service so it outlives the terminal:

```sh
systemd-run --user --unit=openalex-tidy --property=Nice=5 \
  --working-directory=$HOME/Projects/scholarsreadinglist \
  $(which uv) run --with-requirements pipeline/tidy/requirements.txt python -m pipeline.tidy.run all --threads 4
journalctl --user -u openalex-tidy -f      # or: tail -f /mnt/bulk/openalex/tidy/logs/run.log
```

When new OpenAlex files are downloaded, run `all` again. Unchanged files are
skipped. Changed files, files whose superseded rows changed, and changes to the
transform code are redone. Then the shared tables are rebuilt and a new
generation is published. The previous generation stays readable.

The runner refuses to start if the downloader's `status.json` is not `complete`
or a manifest file is missing or the wrong size.

## How it uses the machine

- **One job per source file, as its own process, entirely in RAM.** Measured:
  ~18–20 GiB peak for an 800 MiB file at 8 threads, up to ~26 GiB at 4 threads;
  ~11 GiB for a 95 MiB file.
- **Concurrency:** up to `CPU budget ÷ threads` jobs (default 75% of CPUs, 4
  threads → 9 jobs). A job starts only if its estimated peak fits the RAM budget
  and keeps `--reserve-gib` (24) available. The estimate starts at
  `1.25 × (6.4 GiB + 12.9 × file size)` and is raised whenever a job uses more.
- **Last-resort guard:** if available RAM falls below `--critical-gib` (8), the
  newest job is stopped and retried later with a larger estimate. This already
  triggered once, during a calibration with a memory bug that has since been
  fixed. The server never ran out of memory.
- **Calibration (Oct 6, with other users' load on the server):** large files
  reach about 21 MiB/s overall whether run 4×8, 6×6 or 9×4: a shared machine
  limit, not the job count. Small files run 52% faster at 9×4 than at 4×8.
  Hence `--threads 4`.

## Files

| Path | What |
|---|---|
| `tidy/contracts.py` | table schemas, keys, relations, integrity checks |
| `tidy/reference.py` | plain-Python reference extraction (the correctness standard) |
| `tidy/native.py` | fast Polars extraction; tests require identical output |
| `tidy/reduction.py` | the single ordering rule for one-row-per-key tables |
| `tidy/keys.py` | stable SHA-256 identifiers |
| `tidy/job.py`, `tidy/reduce_bucket.py` | worker processes |
| `tidy/run.py` | versions, scheduling, reduction, publication, gc |
| `tidy/dataset.py` | readers for the published generation |
| `tests/` | 73 tests: edge cases, real papers, native = reference, partition invariance, end-to-end runner |

```sh
uv run --with-requirements pipeline/tidy/requirements.txt python -m unittest discover -s pipeline/tests
```

The earlier `openalex_content.py` and its flat `content/year=…` table are
unchanged and still readable. The new `content` table replaces them once you
are satisfied with it.
