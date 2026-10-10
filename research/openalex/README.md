# OpenAlex works snapshot download

Downloads **metadata and citation links**, not article PDFs or extracted full text.
The entire works snapshot includes OpenAlex's expansion collection as well as
its main collection; its record count is not a count of peer-reviewed articles.

## Current download

- Destination: `/mnt/bulk/openalex/works/` (12 TB HDD).
- Format: Parquet, partitioned by `updated_date`.
- Pinned manifest: `/mnt/bulk/openalex/manifest.parquet.json`.
- Snapshot: **2026-09-23**.
- Files: **2,040**.
- Records: **476,196,327**.
- Download bytes: **707,141,690,793** (~707 GB / 659 GiB).
- Source: `https://openalex.s3.amazonaws.com/data/parquet/manifest.json`.
- No API key or paid API calls are used.

The downloader uses two concurrent HTTP transfers to avoid excessive random
writes on the HDD. The NVMe at `/mnt/fast` remains available for search indexes.

## Monitor and stop

```sh
systemctl --user status openalex-works-download
journalctl --user -u openalex-works-download -f
python3 -m json.tool /mnt/bulk/openalex/status.json

# Pause; partial files are retained.
systemctl --user stop openalex-works-download

# Resume while the transient unit still exists.
systemctl --user restart openalex-works-download
```

This is a **transient user service**: it continues after the conversation closes,
but it does not automatically start after a reboot. To launch again after a reboot:

```sh
systemd-run --user --unit=openalex-works-download \
  --description='Download OpenAlex works Parquet snapshot to HDD' \
  --property=Nice=10 --property=MemoryMax=1G --property=TimeoutStopSec=90 \
  /run/current-system/sw/bin/python3 \
  /home/maxime/Projects/scholarsreadinglist/research/openalex/download_works.py \
  --root /mnt/bulk/openalex --workers 2
```

Alternatively run the Python command directly in a terminal.

## Resume and verification

- A file lock prevents two instances writing into this destination concurrently.
- The script refuses to start if `/mnt/bulk` is not mounted.
- The first run saves the manifest; later runs do not silently switch snapshots.
- Downloads use `.parquet.part` files and HTTP byte-range resume.
- Each object's size must match the manifest. Its ETag is recorded in an
  `.identity.json` sidecar on first contact; subsequent requests use `If-Match`.
- Completed downloads are checked for size, `PAR1` framing, and plausible footer
  length, then renamed atomically. These are structural checks, **not checksums**.
- The manifest itself does not provide a per-file checksum or immutable object
  version. ETags pin objects from first contact, not retroactively from the date
  the manifest was generated. A remotely replaced object with a different size
  or a previously recorded different ETag causes failure rather than overwrite.
- Previously completed files with expected size/framing and an identity sidecar
  are reused locally; they are not re-downloaded on resume.
- Individual network failures are retried. Persistent failures stop the job and
  are recorded in `status.json`; resume explicitly after investigating.
- A full successful run must report `state: complete`, 2,040 completed files,
  and 707,141,690,793 completed bytes. Full Parquet/schema validation across the
  corpus can be done separately after the transfer.

## Checks performed before starting

1. Downloaded the first three files.
2. Truncated one owned download to 50,000 bytes and successfully resumed it.
3. Opened all three with DuckDB; their row counts matched the manifest:
   1,578, 27, and 22.
4. Started the full background download and checked its service state.

The `--limit` option is for smoke tests only. Its status reports only the selected
subset; do not run it when checking whether the complete snapshot is downloaded.
