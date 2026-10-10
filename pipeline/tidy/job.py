"""One source-file extraction job, run as its own process by run.py.

Reads the whole file into RAM (measured to fit), drops superseded rows listed by
the version map, extracts all tables natively, validates the partition, writes
Parquet into a temporary directory and atomically renames it into place.
`done.json` inside the final directory is the only completion marker.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import time

BUCKET_ROW_GROUP = 65_536


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--exclude', type=Path, help='Parquet with column row_index: superseded rows to drop')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--buckets', type=int, required=True)
    p.add_argument('--job-json', type=Path, required=True, help='Job description to embed in done.json')
    args = p.parse_args()
    os.environ['POLARS_MAX_THREADS'] = str(args.threads)
    import polars as pl
    from . import native
    from .contracts import TABLES, REDUCED, OWNER
    from .validation import validate

    started = time.perf_counter()
    if args.out.exists():
        print(json.dumps({'skipped': 'already complete', 'out': str(args.out)}))
        return
    # The scheduler runs at most one job per output directory, so leftovers from a
    # stopped/killed earlier attempt can be removed safely.
    for stale in args.out.parent.glob(args.out.name + '.tmp-*'):
        shutil.rmtree(stale, ignore_errors=True)
    tmp = args.out.with_name(args.out.name + f'.tmp-{os.getpid()}')
    tmp.mkdir(parents=True)
    t = time.perf_counter()
    raw = pl.read_parquet(args.source)
    rows_in = raw.height
    excluded = 0
    if args.exclude:
        drop = pl.read_parquet(args.exclude).get_column('row_index').cast(pl.UInt32)
        keep = ~pl.int_range(0, raw.height, dtype=pl.UInt32, eager=True).is_in(drop)
        raw = raw.filter(keep)
        excluded = rows_in - raw.height
    read_s = time.perf_counter() - t
    phases = {}
    t = time.perf_counter()
    result = native.extract(raw, timings=phases)
    native_s = time.perf_counter() - t
    del raw
    t = time.perf_counter()
    checks = validate(result.tables)
    validate_s = time.perf_counter() - t
    t = time.perf_counter()
    files = {}
    for name, df in result.tables.items():
        key = TABLES[name].key
        if name in REDUCED:
            bucket = (pl.col(key).chash.sha2_256().str.slice(0, 8).str.to_integer(base=16) % args.buckets).cast(pl.Int32)
            df = df.with_columns(bucket.alias('_bucket')).sort('_bucket', key)
            row_group = BUCKET_ROW_GROUP
        else:
            df = df.sort(['year', key] if name == 'content' else [OWNER[name], key], nulls_last=True)
            row_group = 250_000
        target = tmp / f'{name}.parquet'
        df.write_parquet(target, compression='zstd', statistics=True, row_group_size=row_group)
        files[name] = dict(rows=df.height, bytes=target.stat().st_size)
    write_s = time.perf_counter() - t
    usage = resource.getrusage(resource.RUSAGE_SELF)
    done = dict(job=json.loads(args.job_json.read_text()), rows_in=rows_in, rows_excluded=excluded,
                tables=files, checks=checks, notes=result.notes, threads=args.threads,
                seconds=dict(read=read_s, native=native_s, validate=validate_s, write=write_s,
                             total=time.perf_counter() - started),
                phases=phases, peak_rss_bytes=usage.ru_maxrss * 1024,
                cpu_seconds=usage.ru_utime + usage.ru_stime, completed_at=time.time())
    (tmp / 'done.json').write_text(json.dumps(done, indent=1, default=str))
    try:
        os.rename(tmp, args.out)  # atomic on the same filesystem
    except OSError:
        if (args.out / 'done.json').exists():
            shutil.rmtree(tmp, ignore_errors=True)  # a concurrent identical job finished first
        else:
            raise
    print(json.dumps({'out': str(args.out), 'seconds': done['seconds'], 'peak_rss_bytes': done['peak_rss_bytes']}))


if __name__ == '__main__':
    main()
