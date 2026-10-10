"""Measure one independent, in-memory partition job. Never publishes production data.

Run separate processes for reliable peak-RAM measurements; POLARS_MAX_THREADS is
set before importing the engine. Use --write-dir to include actual Parquet writes.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--rows',type=int,default=1000)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--compare-reference',type=int,default=0,metavar='ROWS')
    p.add_argument('--write-dir',type=Path)
    args=p.parse_args()
    if args.rows<1 or args.offset<0 or args.threads<1 or args.compare_reference<0:
        p.error('rows/threads must be positive; offsets and comparison sizes nonnegative')
    os.environ['POLARS_MAX_THREADS']=str(args.threads)
    import polars as pl
    from polars.testing import assert_frame_equal
    from . import native, reference
    from .contracts import TABLES, SCHEMA_VERSION
    from .validation import validate
    t=time.perf_counter()
    raw=pl.scan_parquet(args.input).slice(args.offset,args.rows).collect()
    read_s=time.perf_counter()-t
    if not raw.height:
        p.error('The requested source slice contains no rows')
    phases={}
    t=time.perf_counter(); result=native.extract(raw,timings=phases); native_s=time.perf_counter()-t
    t=time.perf_counter(); checks=validate(result.tables); validation_s=time.perf_counter()-t
    sizes={name:df.estimated_size() for name,df in result.tables.items()}
    write_s=0.0; disk_bytes=0
    if args.write_dir:
        args.write_dir.mkdir(parents=True,exist_ok=False)
        t=time.perf_counter()
        for name,df in result.tables.items():
            target=args.write_dir/(name+'.parquet')
            df.write_parquet(target,compression='zstd',statistics=True,row_group_size=100_000)
            disk_bytes+=target.stat().st_size
        write_s=time.perf_counter()-t
    # Linux ru_maxrss is KiB. Capture before the slower oracle is invoked.
    peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    cpu=resource.getrusage(resource.RUSAGE_SELF)
    code_files = sorted(Path(__file__).parent.glob('*.py'))
    code_hash = sha256(b''.join(p.name.encode()+b'\0'+p.read_bytes()+b'\0' for p in code_files)).hexdigest()
    output=dict(schema_version=SCHEMA_VERSION,adapter_method=reference.METHOD,code_sha256=code_hash,
                source=str(args.input.resolve()),source_bytes=args.input.stat().st_size,
                source_mtime_ns=args.input.stat().st_mtime_ns,
                offset=args.offset,input_rows=raw.height,threads=pl.thread_pool_size(),
                read_seconds=read_s,native_seconds=native_s,phases=phases,validation_seconds=validation_s,
                write_seconds=write_s,peak_process_bytes=peak_bytes,cpu_user_seconds=cpu.ru_utime,
                cpu_system_seconds=cpu.ru_stime,raw_estimated_bytes=raw.estimated_size(),
                output_estimated_bytes=sizes,output_parquet_bytes=disk_bytes,
                write_directory=str(args.write_dir) if args.write_dir else None,
                rows={name:df.height for name,df in result.tables.items()},checks=checks,notes=result.notes)
    if args.compare_reference:
        sample=raw.head(args.compare_reference)
        fast=native.extract(sample)
        t=time.perf_counter()
        slow=reference.extract(sample.to_dicts(),native.source_revisions(sample))
        ref_s=time.perf_counter()-t
        for name,spec in TABLES.items():
            assert_frame_equal(fast.tables[name].sort(spec.key),slow.tables[name].sort(spec.key),check_exact=True)
        output['reference_comparison']=dict(rows=sample.height,seconds=ref_s,all_nine_tables_equal=True)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
