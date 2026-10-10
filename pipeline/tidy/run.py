"""OpenAlex works snapshot -> tidy tables. Resumable, RAM-guarded, incremental.

    python -m pipeline.tidy.run plan                 # what would happen; changes nothing
    python -m pipeline.tidy.run all                  # versions -> extract -> reduce -> publish
    python -m pipeline.tidy.run status               # progress of the extraction store
    python -m pipeline.tidy.run extract --files 'updated_date=2026-09-2*'   # a subset (pilots)

Stages
  versions  Read (id, updated_date) of every source row; pick one current version
            per work (latest updated_date; ties -> later file, later row). Superseded
            rows are written per file as exclusion lists. Refuses if the downloader
            did not finish or a manifest file is missing/size-mismatched.
  extract   One job per source file, as its own process: read the file into RAM,
            drop superseded rows, extract all tables, validate, write atomically.
            A job's identity = transform code + source file signature + exclusions,
            so unchanged files are skipped and changed ones redone.
  reduce    One row per entity/classification/vocabulary edge across all jobs,
            by the same documented order used inside a job (partition invariant).
  publish   Global checks, then a generation manifest and an atomic CURRENT switch.
            Readers use pipeline.tidy.dataset.scan(table); old generations stay.

Concurrency is admitted against a RAM budget using a conservative per-file
estimate learned from measured jobs, plus a live MemAvailable guard that stops
admitting jobs, and as a last resort stops the newest job (it is retried later),
rather than letting the server run out of memory.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field as dc_field
from fnmatch import fnmatch
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GIB = 2**30
TRANSFORM_FILES = ('native.py', 'contracts.py', 'keys.py', 'reduction.py', 'source_roles.py',
                   'reference.py', 'job.py', 'reduce_bucket.py', 'requirements.txt')


# ── small utilities ──────────────────────────────────────────────────────────

def log(msg, logfile=None):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(line + '\n')


def meminfo(name='MemAvailable'):
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(name + ':'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('cannot read /proc/meminfo')


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def digest(obj):
    return sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def transform_hash(buckets):
    from .contracts import SCHEMA_VERSION
    h = sha256()
    for name in TRANSFORM_FILES:
        h.update(name.encode() + b'\0' + (HERE / name).read_bytes() + b'\0')
    h.update(f'{SCHEMA_VERSION}|buckets={buckets}'.encode())
    return h.hexdigest()


# ── inventory ────────────────────────────────────────────────────────────────

@dataclass
class Source:
    key: str
    path: Path
    size: int
    mtime_ns: int
    etag: str | None
    records: int | None

    def signature(self):
        return dict(key=self.key, size=self.size, mtime_ns=self.mtime_ns, etag=self.etag, records=self.records)


def inventory(root: Path, allow_incomplete=False, pattern=None):
    manifest = json.loads((root / 'manifest.parquet.json').read_text())
    works = next(e for e in manifest['entities'] if e['entity'] == 'works')
    status_path = root / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    problems = []
    if status.get('state') != 'complete':
        problems.append(f"download state is {status.get('state')!r}, not 'complete'")
    sources = []
    for f in works['files']:
        rel = f['url'].split('/works/', 1)[1]
        path = root / 'works' / rel
        want = f['meta']['content_length']
        if not path.exists() or path.stat().st_size != want:
            problems.append(f'missing or size-mismatched: {rel}')
            continue
        side = path.with_name(path.name + '.identity.json')
        etag = json.loads(side.read_text()).get('etag') if side.exists() else None
        key = rel.replace('/', '__').removesuffix('.parquet')
        st = path.stat()
        sources.append(Source(key, path, st.st_size, st.st_mtime_ns, etag, f['meta'].get('record_count')))
    if problems and not allow_incomplete:
        raise SystemExit('Refusing to plan from an incomplete inventory (use --allow-incomplete to override):\n  '
                         + '\n  '.join(problems[:10]) + (f'\n  … {len(problems) - 10} more' if len(problems) > 10 else ''))
    sources.sort(key=lambda s: s.key)  # updated_date=YYYY-MM-DD__part_NNNN: chronological
    selected = [s for s in sources if fnmatch(s.key, pattern)] if pattern else sources
    return dict(snapshot=manifest.get('date'), sources=sources, selected=selected, problems=problems)


# ── stage: versions ──────────────────────────────────────────────────────────

def versions(tidy: Path, sources, logfile, threads):
    key = digest([s.signature() for s in sources])
    state = tidy / 'state' / 'versions.json'
    if state.exists():
        old = json.loads(state.read_text())
        if old.get('key') == key:
            log(f"versions: unchanged inventory, reusing ({old['superseded_rows']:,} superseded rows)", logfile)
            return old
    os.environ.setdefault('POLARS_MAX_THREADS', str(threads))
    import polars as pl
    log(f'versions: reading id/updated_date from {len(sources)} files', logfile)
    t = time.time()
    frames = []
    batch = 32
    for start in range(0, len(sources), batch):
        lazies = []
        for i, s in enumerate(sources[start:start + batch], start):
            lazies.append(pl.scan_parquet(s.path).select(
                pl.col('id').str.extract(r'^https://openalex\.org/W([0-9]+)$').cast(pl.Int64).alias('w'),
                pl.col('id').is_null().alias('null_id'),
                pl.col('updated_date').cast(pl.Datetime('us', 'UTC')).dt.epoch('us').fill_null(-2**62).alias('u'),
            ).with_row_index('row').with_columns(pl.lit(i, pl.Int32).alias('f')))
        frames.extend(pl.collect_all(lazies))
        if (start // batch) % 8 == 0:
            log(f'versions: {min(start + batch, len(sources))}/{len(sources)} files', logfile)
    counts = {sources[df['f'][0]].key: df.height for df in frames if df.height}
    mismatched = [s.key for s in sources if s.records is not None and counts.get(s.key, 0) != s.records]
    if mismatched:
        raise SystemExit(f'Row counts differ from the download manifest for {len(mismatched)} files, e.g. {mismatched[:3]}')
    allrows = pl.concat(frames)
    del frames
    bad = allrows.filter(pl.col('w').is_null())
    if bad.height:
        raise SystemExit(f'{bad.height:,} rows have no openalex:W<number> id, e.g. file '
                         f"{sources[bad['f'][0]].key} row {bad['row'][0]}")
    dup = allrows.filter(pl.col('w').is_duplicated()).drop('null_id')
    total, distinct = allrows.height, allrows.height - dup.height + dup['w'].n_unique()
    del allrows
    losers = (dup.sort(['w', 'u', 'f', 'row'], descending=[False, True, True, True])
              .with_columns(pl.int_range(pl.len()).over('w').alias('rank')).filter(pl.col('rank') > 0))
    excl_dir = tidy / 'state' / 'exclusions' / key[:16]
    exclusions = {}
    for (f,), part in losers.group_by('f'):
        s = sources[f]
        path = excl_dir / f'{s.key}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = part.select(pl.col('row').cast(pl.UInt32).alias('row_index')).sort('row_index')
        rows.write_parquet(path)
        exclusions[s.key] = dict(path=str(path), rows=rows.height, sha256=digest(rows['row_index'].to_list()))
    out = dict(key=key, built_at=time.time(), seconds=time.time() - t, total_rows=total, distinct_works=distinct,
               duplicated_works=dup['w'].n_unique(), superseded_rows=losers.height, exclusions=exclusions,
               rule='latest updated_date; ties: later source file, then later row')
    write_json(state, out)
    log(f"versions: {total:,} rows, {distinct:,} works, {losers.height:,} superseded ({out['seconds']:.0f}s)", logfile)
    return out


# ── generic RAM-guarded process scheduler ────────────────────────────────────

@dataclass
class Task:
    name: str
    cmd: list
    est: float            # estimated peak bytes
    size: int = 0         # input bytes (for the estimate model and progress)
    out: Path | None = None
    attempts: int = 0
    proc: subprocess.Popen | None = None
    started: float = 0.0
    log: Path | None = None
    meta: dict = dc_field(default_factory=dict)


class Estimator:
    """Peak RSS ~ a + b * input_bytes, conservatively raised to cover every observed job."""
    def __init__(self, a_gib, b_per_gib, margin):
        self.a, self.b, self.margin = a_gib * GIB, b_per_gib, margin

    def __call__(self, size):
        return self.margin * (self.a + self.b * size)

    def observe(self, size, peak):
        self.a = max(self.a, peak / self.margin - self.b * size)


def schedule(tasks, *, max_jobs, budget, reserve, critical, logfile, on_done=None, label='jobs',
             estimator=None, progress_bytes=True):
    pending = sorted(tasks, key=lambda t: t.size, reverse=True)  # big first: best packing
    running, failed = [], []
    total = len(pending)
    done = 0
    bytes_total = sum(t.size for t in pending) or 1
    bytes_done = 0
    t0 = time.time()
    last_report = 0.0
    killed_for_memory = 0

    def start(t):
        t.attempts += 1
        t.log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(t.log, 'a')
        t.proc = subprocess.Popen(t.cmd, cwd=REPO, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                  env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        t.started = time.time()
        fh.close()
        running.append(t)

    try:
        while pending or running:
            avail = meminfo()
            # last-resort guard: stop the newest job before the machine runs out
            if avail < critical and running:
                victim = max(running, key=lambda t: t.started)
                victim.proc.send_signal(signal.SIGTERM)
                try:
                    victim.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    victim.proc.kill(); victim.proc.wait()
                running.remove(victim)
                killed_for_memory += 1
                victim.est = max(victim.est * 1.5, budget * 0.5)
                log(f'MEMORY GUARD: available {avail / GIB:.1f} GiB < {critical / GIB:.0f} GiB; '
                    f'stopped {victim.name} (will retry with a larger estimate)', logfile)
                if victim.attempts < 3:
                    pending.append(victim)
                else:
                    failed.append(victim)
                time.sleep(5)
                continue
            # admit
            in_use = sum(t.est for t in running)
            for t in list(pending):
                if len(running) >= max_jobs:
                    break
                fits = in_use + t.est <= budget and avail - t.est >= reserve
                if fits or (not running and avail - t.est >= critical):
                    pending.remove(t)
                    start(t)
                    in_use += t.est
                    avail -= t.est
            # reap
            for t in list(running):
                rc = t.proc.poll()
                if rc is None:
                    continue
                running.remove(t)
                if rc == 0 and (t.out is None or t.out.exists()):
                    done += 1
                    bytes_done += t.size
                    if on_done:
                        on_done(t)
                else:
                    log(f'FAILED {t.name} (exit {rc}, attempt {t.attempts}); log: {t.log}', logfile)
                    if t.attempts < 2:
                        pending.append(t)
                    else:
                        failed.append(t)
            now = time.time()
            if now - last_report > 60 or (not pending and not running):
                last_report = now
                el = now - t0
                frac = bytes_done / bytes_total if progress_bytes else done / max(total, 1)
                eta = (el / frac - el) if frac > 0 else float('nan')
                log(f'{label}: {done}/{total} done · {frac:.1%} · running {len(running)} '
                    f'(est {sum(t.est for t in running) / GIB:.0f} GiB, rss {sum(rss(t.proc.pid) for t in running) / GIB:.0f} GiB) · '
                    f'available {meminfo() / GIB:.0f} GiB · {el / 3600:.2f} h elapsed · ETA {eta / 3600:.1f} h', logfile)
            time.sleep(0.5)
    finally:
        for t in running:
            if t.proc and t.proc.poll() is None:
                t.proc.terminate()
        for t in running:
            try:
                t.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                t.proc.kill()
    return dict(done=done, failed=[t.name for t in failed], killed_for_memory=killed_for_memory)


# ── stage: extract ───────────────────────────────────────────────────────────

def jobs(tidy, sources, vers, buckets):
    code = transform_hash(buckets)
    out = []
    for s in sources:
        excl = vers['exclusions'].get(s.key)
        job = dict(code=code, source=s.signature(), exclusions=excl and excl['sha256'], buckets=buckets)
        jkey = digest(job)[:20]
        out.append(dict(source=s, job=job, key=jkey, dir=tidy / 'extract' / s.key / jkey,
                        exclude=Path(excl['path']) if excl else None))
    return out


def resources(args):
    cpus = len(os.sched_getaffinity(0))
    cpu_budget = args.cpu_budget or max(1, int(cpus * 0.75))
    max_jobs = args.jobs or max(1, cpu_budget // args.threads)
    avail = meminfo()
    reserve, critical = args.reserve_gib * GIB, args.critical_gib * GIB
    budget = min(args.ram_budget_gib * GIB if args.ram_budget_gib else avail, avail - reserve)
    return dict(cpus=cpus, cpu_budget=cpu_budget, max_jobs=max_jobs, budget=budget,
                reserve=reserve, critical=critical, available=avail)


def extract(args, tidy, sources, vers, logfile):
    todo = [j for j in jobs(tidy, sources, vers, args.buckets) if not (j['dir'] / 'done.json').exists()]
    if not todo:
        log('extract: nothing to do', logfile)
        return dict(done=0, failed=[])
    r = resources(args)
    est = Estimator(args.est_base_gib, args.est_per_gib, args.est_margin)
    learned = tidy / 'state' / 'estimator.json'
    if learned.exists():
        est.a = max(est.a, json.loads(learned.read_text())['a'])
    free = shutil.disk_usage(tidy).free
    need = sum(j['source'].size for j in todo) * args.output_ratio
    if free < need * 1.15 + args.disk_reserve_gib * GIB:
        raise SystemExit(f'Not enough disk: {free / GIB:.0f} GiB free, ~{need / GIB:.0f} GiB expected output')
    log(f"extract: {len(todo)} jobs, {sum(j['source'].size for j in todo) / GIB:.1f} GiB input · "
        f"{r['max_jobs']} jobs × {args.threads} threads (CPU budget {r['cpu_budget']}/{r['cpus']}) · "
        f"RAM budget {r['budget'] / GIB:.0f} GiB, reserve {r['reserve'] / GIB:.0f}, critical {r['critical'] / GIB:.0f}", logfile)
    tasks = []
    for j in todo:
        spec = j['dir'].with_name(j['dir'].name + '.job.json')
        write_json(spec, dict(job=j['job'], key=j['key']))
        cmd = [sys.executable, '-m', 'pipeline.tidy.job', '--source', str(j['source'].path), '--out', str(j['dir']),
               '--threads', str(args.threads), '--buckets', str(args.buckets), '--job-json', str(spec)]
        if j['exclude']:
            cmd += ['--exclude', str(j['exclude'])]
        tasks.append(Task(j['source'].key, cmd, est(j['source'].size), j['source'].size, j['dir'] / 'done.json',
                          log=tidy / 'logs' / 'jobs' / f"{j['source'].key}.log"))

    def on_done(t):
        d = json.loads(t.out.read_text())
        est.observe(t.size, d['peak_rss_bytes'])
        write_json(learned, dict(a=est.a, b=est.b, margin=est.margin))
        with open(tidy / 'state' / 'job_metrics.jsonl', 'a') as f:
            f.write(json.dumps(dict(key=t.name, bytes=t.size, peak=d['peak_rss_bytes'], seconds=d['seconds'],
                                    rows=sum(x['rows'] for x in d['tables'].values()),
                                    out_bytes=sum(x['bytes'] for x in d['tables'].values()))) + '\n')

    return schedule(tasks, max_jobs=r['max_jobs'], budget=r['budget'], reserve=r['reserve'], critical=r['critical'],
                    logfile=logfile, on_done=on_done, label='extract', estimator=est)


# ── stage: reduce ────────────────────────────────────────────────────────────

def reduce(args, tidy, sources, vers, logfile):
    from .contracts import REDUCED
    js = jobs(tidy, sources, vers, args.buckets)
    missing = [j['source'].key for j in js if not (j['dir'] / 'done.json').exists()]
    if missing:
        raise SystemExit(f'reduce: {len(missing)} extraction jobs incomplete, e.g. {missing[:3]}; run extract first')
    inputs_key = digest([j['key'] for j in js])[:16]
    staging = tidy / 'reduced' / inputs_key
    tasks = []
    for table in REDUCED:
        listing = staging / f'{table}.inputs'
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text('\n'.join(str(j['dir'] / f'{table}.parquet') for j in js) + '\n')
        size = sum((j['dir'] / f'{table}.parquet').stat().st_size for j in js) // args.buckets
        for b in range(args.buckets):
            out = staging / table / f'bucket={b:03d}.parquet'
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-m', 'pipeline.tidy.reduce_bucket', '--table', table, '--bucket', str(b),
                   '--inputs', str(listing), '--out', str(out), '--threads', str(args.threads)]
            # decoded rows are ~6-10x their compressed size; sort/unique roughly doubles it
            tasks.append(Task(f'{table}/{b}', cmd, max(2 * GIB, 24 * size), size, out,
                              log=tidy / 'logs' / 'reduce' / f'{table}-{b:03d}.log'))
    if tasks:
        r = resources(args)
        log(f'reduce: {len(tasks)} bucket tasks into {staging}', logfile)
        res = schedule(tasks, max_jobs=r['max_jobs'], budget=r['budget'], reserve=r['reserve'],
                       critical=r['critical'], logfile=logfile, label='reduce')
        if res['failed']:
            raise SystemExit(f"reduce failed: {res['failed'][:5]}")
    else:
        log('reduce: nothing to do', logfile)
    return staging, js


# ── stage: publish ───────────────────────────────────────────────────────────

def publish(args, tidy, snapshot, vers, staging, js, logfile):
    import polars as pl
    from .contracts import TABLES, REDUCED, CONTENT_OWNED, SCHEMA_VERSION
    t = time.time()
    tables, notes, checks = {}, {}, {}
    rows = {name: 0 for name in TABLES}
    for j in js:
        d = json.loads((j['dir'] / 'done.json').read_text())
        for name, info in d['tables'].items():
            if name in CONTENT_OWNED:
                rows[name] += info['rows']
        for n in d['notes']:
            notes[n['code']] = notes.get(n['code'], 0) + n.get('count', 1)
    for name in CONTENT_OWNED:
        tables[name] = [str(j['dir'] / f'{name}.parquet') for j in js]
    for name in REDUCED:
        files = sorted(str(p) for p in (staging / name).glob('bucket=*.parquet'))
        if len(files) != args.buckets:
            raise SystemExit(f'publish: {name} has {len(files)}/{args.buckets} buckets')
        tables[name] = files
        rows[name] = sum(json.loads(Path(f).with_suffix('.json').read_text())['rows_out'] for f in files)
    # Global checks that partition validation cannot see.
    ids = pl.scan_parquet(tables['content']).select(pl.col('content_id').str.slice(10).cast(pl.Int64)).collect()
    checks['content_rows'] = ids.height
    checks['content_ids_unique'] = ids.get_column('content_id').n_unique() == ids.height
    checks['content_rows_equal_distinct_works'] = ids.height == vers['distinct_works']
    del ids
    for name in TABLES:
        schema = pl.read_parquet_schema(tables[name][0]) if tables[name] else None
        checks[f'schema_{name}'] = dict(schema) == TABLES[name].columns
    bad = [k for k, v in checks.items() if v is False]
    if bad:
        raise SystemExit(f'publish: global checks failed: {bad}')
    gen_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + digest([j['key'] for j in js])[:8]
    manifest = dict(generation=gen_id, created_at=time.time(), schema_version=SCHEMA_VERSION,
                    transform=transform_hash(args.buckets), snapshot=snapshot, versions_key=vers['key'],
                    superseded_rows=vers['superseded_rows'], rows=rows, checks=checks, notes=notes,
                    tables={name: dict(files=files) for name, files in tables.items()},
                    note='Content-owned files are immutable extraction outputs; reduced files live in reduced/.')
    gdir = tidy / 'generations' / gen_id
    write_json(gdir / 'manifest.json', manifest)
    tmp = tidy / f'CURRENT.tmp-{os.getpid()}'
    tmp.write_text(gen_id + '\n')
    os.replace(tmp, tidy / 'CURRENT')
    log(f'publish: generation {gen_id} is CURRENT ({time.time() - t:.0f}s) · '
        + ', '.join(f'{k} {v:,}' for k, v in rows.items()), logfile)
    return manifest


# ── stage: garbage collection ────────────────────────────────────────────────

def gc(tidy, keep, really, logfile):
    """Delete extraction/reduction outputs not used by the newest `keep` generations."""
    gens = sorted((p for p in (tidy / 'generations').glob('*') if (p / 'manifest.json').exists()), reverse=True)
    if not gens:
        raise SystemExit('gc: no published generation yet; nothing is safe to delete')
    kept, used = gens[:keep], set()
    for g in kept:
        for t in json.loads((g / 'manifest.json').read_text())['tables'].values():
            used.update(str(Path(f).parent) for f in t['files'])
    victims = [p for p in (tidy / 'extract').glob('*/*') if p.is_dir() and str(p) not in used]
    victims += [p for p in (tidy / 'extract').glob('*/*.job.json') if str(p.with_suffix('').with_suffix('')) not in used]
    victims += [p for p in (tidy / 'reduced').glob('*') if not any(u.startswith(str(p)) for u in used)]
    victims += gens[keep:]
    size = sum(sum(f.stat().st_size for f in v.rglob('*') if f.is_file()) if v.is_dir() else v.stat().st_size for v in victims)
    log(f"gc: {len(victims)} unused paths, {size / GIB:.1f} GiB{'' if really else ' (dry run; add --yes to delete)'}", logfile)
    if really:
        for v in victims:
            shutil.rmtree(v) if v.is_dir() else v.unlink()


# ── commands ─────────────────────────────────────────────────────────────────

def status(tidy, sources, args):
    state = tidy / 'state' / 'versions.json'
    vers = json.loads(state.read_text()) if state.exists() else None
    print(f'source files: {len(sources)} ({sum(s.size for s in sources) / GIB:.1f} GiB)')
    if not vers:
        print('versions: not built yet')
        return
    js = jobs(tidy, sources, vers, args.buckets)
    done = [j for j in js if (j['dir'] / 'done.json').exists()]
    print(f"versions: {vers['distinct_works']:,} works, {vers['superseded_rows']:,} superseded rows")
    print(f"extract: {len(done)}/{len(js)} jobs complete ({sum(j['source'].size for j in done) / GIB:.1f}/"
          f"{sum(j['source'].size for j in js) / GIB:.1f} GiB input)")
    out = sum(sum(x['bytes'] for x in json.loads((j['dir'] / 'done.json').read_text())['tables'].values()) for j in done)
    print(f'extract output so far: {out / GIB:.1f} GiB')
    cur = tidy / 'CURRENT'
    print('CURRENT generation:', cur.read_text().strip() if cur.exists() else 'none')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['plan', 'versions', 'extract', 'reduce', 'publish', 'all', 'status', 'gc'])
    p.add_argument('--keep', type=int, default=2, help='gc: generations to keep (and everything they use)')
    p.add_argument('--yes', action='store_true', help='gc: actually delete (default: only report)')
    p.add_argument('--root', type=Path, default=Path(os.environ.get('OPENALEX_ROOT', '/mnt/bulk/openalex')))
    p.add_argument('--tidy-root', type=Path, help='output root (default: <root>/tidy)')
    p.add_argument('--files', help="fnmatch pattern on file keys, e.g. 'updated_date=2026-09-2*' (extract subset only)")
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--threads', type=int, default=8, help='Polars threads per job (measured best: 8)')
    p.add_argument('--jobs', type=int, help='max concurrent jobs (default: CPU budget // threads)')
    p.add_argument('--cpu-budget', type=int, help='threads across all jobs (default: 75%% of CPUs)')
    p.add_argument('--ram-budget-gib', type=float, help='cap on summed job estimates (default: available - reserve)')
    p.add_argument('--reserve-gib', type=float, default=24, help='keep at least this much RAM available')
    p.add_argument('--critical-gib', type=float, default=8, help='stop the newest job below this')
    p.add_argument('--buckets', type=int, default=32)
    p.add_argument('--est-base-gib', type=float, default=6.4)
    p.add_argument('--est-per-gib', type=float, default=12.9, help='peak RAM per GiB of input file')
    p.add_argument('--est-margin', type=float, default=1.25)
    p.add_argument('--output-ratio', type=float, default=3.0, help='assumed output/input bytes for the disk check')
    p.add_argument('--disk-reserve-gib', type=float, default=100, help='free disk to keep beyond expected output')
    args = p.parse_args(argv)
    tidy = args.tidy_root or args.root / 'tidy'
    tidy.mkdir(parents=True, exist_ok=True)
    (tidy / 'logs').mkdir(exist_ok=True)
    logfile = tidy / 'logs' / 'run.log'
    inv = inventory(args.root, args.allow_incomplete, args.files)
    sources = inv['sources']
    if args.files and args.command in ('reduce', 'publish', 'all'):
        raise SystemExit('--files is for extraction pilots; reduce/publish always use the whole inventory')
    if args.command == 'status':
        return status(tidy, sources, args)
    lock = tidy / 'run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{os.getpid()}\n'.encode()); os.close(fd)
    except FileExistsError:
        pid = lock.read_text().strip()
        if pid and Path(f'/proc/{pid}').exists():
            raise SystemExit(f'another run (pid {pid}) holds {lock}')
        lock.write_text(f'{os.getpid()}\n')  # stale lock from a dead process
    try:
        if args.command == 'plan':
            r = resources(args)
            state = tidy / 'state' / 'versions.json'
            print(f"snapshot {inv['snapshot']}: {len(sources)} files, {sum(s.size for s in sources) / GIB:.1f} GiB"
                  + (f", problems: {len(inv['problems'])}" if inv['problems'] else ''))
            print(f"resources: {r['cpus']} CPUs (budget {r['cpu_budget']}), {r['available'] / GIB:.0f} GiB available, "
                  f"RAM budget {r['budget'] / GIB:.0f} GiB → up to {r['max_jobs']} jobs × {args.threads} threads")
            print('versions:', 'built' if state.exists() else 'not built yet (first step of `all`)')
            print(f'disk free: {shutil.disk_usage(tidy).free / GIB:.0f} GiB at {tidy}')
            return
        if args.command == 'gc':
            return gc(tidy, args.keep, args.yes, logfile)
        vers = versions(tidy, sources, logfile, args.threads * 4)
        selected = inv['selected']
        if args.command == 'versions':
            return
        if args.command in ('extract', 'all'):
            res = extract(args, tidy, selected, vers, logfile)
            if res['failed']:
                raise SystemExit(f"extract: {len(res['failed'])} jobs failed, e.g. {res['failed'][:5]} — rerun to retry")
            if args.command == 'extract':
                return
        staging, js = reduce(args, tidy, sources, vers, logfile)
        if args.command == 'reduce':
            return
        publish(args, tidy, inv['snapshot'], vers, staging, js, logfile)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
