# PubMed Central Open Access mirror

Local copy of every article in NCBI's PMC Open Access cloud bucket
(`https://pmc-oa-opendata.s3.amazonaws.com`, free, no login): metadata JSON
(licence, DOI, PMID), JATS XML full text and NCBI's plain-text rendering.
No PDFs or images (images stay on PMC's open bucket, as ScienceMadeReadable already does).

## Current copy

- Inventory of 2026-10-05: **9,230,596 articles** (latest version of each).
- Output: `/mnt/bulk/pmc/oa-2026-10-05/` (HDD).
- Started 2026-10-06 16:52 UTC as the transient user unit `pmc-mirror`.
  First minutes: ~330–430 articles/s, ~48 MB/s download, ~6 h ETA, no retries,
  no errors; ~0.47 GB disk per 20,000 articles ⇒ **~210 GB** total (≈4.8× smaller than raw).
  Internet latency stayed at ~9 ms during the run.

```sh
cat /mnt/bulk/pmc/oa-2026-10-05/status.json          # live progress
systemctl --user status pmc-mirror
journalctl --user -u pmc-mirror -f
systemctl --user stop pmc-mirror                      # safe: unfinished shards are redone later
```

The unit is transient: it survives logout (user lingering) but not a reboot. After a
reboot, restart it with the same command; finished shards are skipped:

```sh
cd ~/Projects/scholarsreadinglist/pipeline/pmc
systemd-run --user --unit=pmc-mirror -p Nice=10 -p MemoryMax=16G -p MemorySwapMax=0 \
  -p Restart=on-failure -p RestartSec=60 -p WorkingDirectory=$PWD \
  $PWD/.venv/bin/python -u $PWD/mirror.py run --out /mnt/bulk/pmc/oa-2026-10-05
```

When finished: `uv run python mirror.py verify --out /mnt/bulk/pmc/oa-2026-10-05`.

## Layout

```
plan.json        inventory date, count, shard size (5,000), sha256 of ids.txt
ids.txt          pinned "PMC123.4" list, sorted by PMCID number
shards/shard-000123.parquet   pmcid, version, metadata (JSON as published), license_code,
                              doi, pmid, xml (bytes), txt (bytes), error
shards/shard-000123.json      commit record, written last (rows, counts, sha256)
status.json      live progress
```

Every XML/TXT body is checked against the md5 NCBI publishes in its metadata and stored
as exact bytes. A shard is written under a temporary name, read back and checked against
the plan, then renamed; its commit record comes after. `error` is null unless something
published could not be fetched and verified after retries.

Reading:

```python
import pyarrow.parquet as pq
t = pq.read_table("/mnt/bulk/pmc/oa-2026-10-05/shards/shard-000000.parquet",
                  columns=["pmcid", "license_code", "txt"])
print(t[0][0], t[1][0], t[2][0].as_py().decode()[:500])
```

## Licences (sample of 400)

CC BY 54%, CC BY-NC-ND 14%, TDM-only 12%, CC BY-NC 10%, none 5%, CC BY-NC-SA 3%,
CC0 1%. Serving rules are in `search/PLAN.md`: TDM-only text is for search/mining,
never shown.

## Inputs

- `/mnt/bulk/pmc/inventory-2026-10-05/`: NCBI's daily S3 inventory of `metadata/`
  (`latest.txt` = latest version per PMCID).
- `/mnt/bulk/pmc/PMC-ids-2026-10-06.csv.gz`: NCBI's PMCID↔PMID↔DOI map (all of PMC).

## Next: updates

NCBI publishes a new inventory daily. Diff it against `ids.txt` (new PMCIDs and new
versions), fetch those into a dated update directory with the same code, and let
readers prefer the newest version.
