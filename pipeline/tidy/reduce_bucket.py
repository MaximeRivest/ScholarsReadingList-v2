"""Reduce one bucket of a vocabulary/profile table across all extraction jobs.

Each job already reduced its own partition with the same total order, so taking
the maximum again across jobs equals reducing everything at once (tested).
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', required=True)
    p.add_argument('--bucket', type=int, required=True)
    p.add_argument('--inputs', type=Path, required=True, help='Text file: one Parquet path per line')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, required=True)
    args = p.parse_args()
    os.environ['POLARS_MAX_THREADS'] = str(args.threads)
    import polars as pl
    from .contracts import TABLES, ContractError
    from .reduction import reduce_frame

    started = time.perf_counter()
    if args.out.exists():
        return
    files = [line for line in args.inputs.read_text().splitlines() if line]
    df = pl.scan_parquet(files).filter(pl.col('_bucket') == args.bucket).drop('_bucket').collect()
    rows_in = df.height
    notes = []
    out = reduce_frame(args.table, df, notes)
    spec = TABLES[args.table]
    if dict(out.schema) != spec.columns:
        raise ContractError(f'{args.table}: reduced schema differs from contract')
    if out.select(pl.any_horizontal([pl.col(c).is_null() for c in spec.required]).any()).item():
        raise ContractError(f'{args.table}: missing required values after reduction')
    tmp = args.out.with_name(args.out.name + f'.tmp-{os.getpid()}')
    out.sort(spec.key).write_parquet(tmp, compression='zstd', statistics=True, row_group_size=250_000)
    stats = dict(table=args.table, bucket=args.bucket, rows_in=rows_in, rows_out=out.height, notes=notes,
                 seconds=time.perf_counter() - started,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    args.out.with_suffix('.json').write_text(json.dumps(stats))
    os.replace(tmp, args.out)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
