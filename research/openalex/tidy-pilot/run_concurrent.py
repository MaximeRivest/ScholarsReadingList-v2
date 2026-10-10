"""Repeat the measured two-file pilot with two concurrent eight-thread processes.

Run from the repo root using the pinned pipeline/tidy/requirements.txt environment.
This is a benchmark, not the production scheduler. Outputs go to scratch only.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
REPORTS=Path(__file__).resolve().parent
GIB=2**30
BUDGET=64*GIB  # experiment guardrail: over twice the sum of measured individual peaks
RESERVE=16*GIB
INPUTS={
    'middle':Path('/mnt/bulk/openalex/works/updated_date=2026-08-13/part_0012.parquet'),
    'recent':Path('/mnt/bulk/openalex/works/updated_date=2026-09-23/part_0049.parquet'),
}


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
    raise RuntimeError('Cannot read available RAM')


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'): return int(line.split()[1])*1024
    except FileNotFoundError:
        pass
    return 0


def main():
    if available()<BUDGET+RESERVE:
        raise RuntimeError('Not enough current headroom for this particular two-file pilot')
    run='paired-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    children=[];streams=[];reports=[];peak=0;started=time.perf_counter()
    try:
        for label,path in INPUTS.items():
            report=REPORTS/f'{run}-{label}.json';reports.append(report)
            out=report.open('w');err=report.with_suffix('.log').open('w');streams.extend([out,err])
            target=Path('/mnt/bulk/openalex/_tidy_pilot')/f'{run}-{label}'
            cmd=[sys.executable,'-m','pipeline.tidy.pilot',str(path),'--rows','400000','--threads','8','--write-dir',str(target)]
            children.append(subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err))
        while any(p.poll() is None for p in children):
            current=sum(rss(p.pid) for p in children);peak=max(peak,current)
            if current>BUDGET or available()<RESERVE:
                raise RuntimeError('Pilot stopped: memory safety margin exceeded')
            if any(p.returncode not in (None,0) for p in children):
                raise RuntimeError('A pilot worker failed; inspect the .log files')
            time.sleep(0.25)
        if any(p.returncode for p in children):
            raise RuntimeError('A pilot worker failed')
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in streams:f.close()
    elapsed=time.perf_counter()-started
    result=dict(run=run,parallel_jobs=2,threads_per_job=8,elapsed_seconds=elapsed,
                sampled_combined_peak_bytes=peak,reports=[str(p) for p in reports],
                input_rows=sum(json.loads(p.read_text())['input_rows'] for p in reports),
                note='0.25s RSS sampling; use individual ru_maxrss as well. Warm-cache pilot, not a disk throughput guarantee.')
    (REPORTS/f'{run}-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
