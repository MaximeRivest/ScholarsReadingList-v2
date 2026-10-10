#!/usr/bin/env python3
"""Resume the manifest-pinned OpenAlex works Parquet snapshot using public HTTPS.

No credentials required. Uses only Python's standard library. Completed files
are checked for expected size and Parquet framing (not a cryptographic checksum).
ETags pin each object's identity once first contacted; changed objects fail closed.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import threading
import time
import urllib.error
import urllib.request

MANIFEST_URL = 'https://openalex.s3.amazonaws.com/data/parquet/manifest.json'
STOP = threading.Event()
HEADERS = {'User-Agent': 'ScholarsReadingList-OpenAlexSnapshot/1.0', 'Accept-Encoding': 'identity'}


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def request(url, method='GET', headers=None):
    return urllib.request.urlopen(urllib.request.Request(
        url, method=method, headers={**HEADERS, **(headers or {})}), timeout=60)


def valid_parquet(path, expected):
    if not path.exists() or path.stat().st_size != expected or expected < 12:
        return False
    with path.open('rb') as stream:
        first = stream.read(4)
        stream.seek(-8, 2)
        tail = stream.read(8)
    footer_length = int.from_bytes(tail[:4], 'little')
    return first == b'PAR1' and tail[4:] == b'PAR1' and 0 < footer_length <= expected - 12


def object_paths(root, entry):
    prefix = 's3://openalex/data/parquet/works/'
    if not entry['url'].startswith(prefix):
        raise ValueError('Unexpected manifest object URL')
    relative = PurePosixPath(entry['url'][len(prefix):])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe manifest path')
    path = root / 'works' / relative
    url = 'https://openalex.s3.amazonaws.com/' + entry['url'][len('s3://openalex/'):]
    return path, url


def download_one(root, entry):
    path, url = object_paths(root, entry)
    expected = entry['meta']['content_length']
    part = path.with_suffix(path.suffix + '.part')
    identity_path = path.with_suffix(path.suffix + '.identity.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if valid_parquet(path, expected) and identity_path.exists():
            return 'cached'
        raise RuntimeError(f'Refusing to overwrite unexpected existing file: {path}')
    last_error = None
    for attempt in range(6):
        if STOP.is_set():
            return 'stopped'
        try:
            with request(url, method='HEAD') as response:
                remote_size = int(response.headers['Content-Length'])
                etag = response.headers.get('ETag')
                modified = response.headers.get('Last-Modified')
            if remote_size != expected:
                raise ValueError(f'Object size changed since manifest: {url}: {remote_size} != {expected}')
            if not etag:
                raise ValueError(f'Missing object ETag: {url}')
            identity = {'url': url, 'etag': etag, 'bytes': expected, 'last_modified': modified}
            if identity_path.exists():
                saved = json.loads(identity_path.read_text())
                if any(saved[k] != identity[k] for k in ('url', 'etag', 'bytes')):
                    raise ValueError(f'Object identity changed; refusing to mix versions: {url}')
            else:
                atomic_json(identity_path, identity)
            offset = part.stat().st_size if part.exists() else 0
            if offset > expected:
                raise ValueError(f'Partial file too large: {part}')
            if offset == expected:
                if not valid_parquet(part, expected):
                    raise ValueError(f'Invalid Parquet framing in complete partial file: {part}')
                part.replace(path)
                return 'downloaded'
            if shutil.disk_usage(root).free < expected - offset + 5_000_000_000:
                raise ValueError('Insufficient free disk space; retaining partial downloads')
            headers = {'If-Match': etag}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            with request(url, headers=headers) as response:
                if response.headers.get('ETag') != etag:
                    raise ValueError('GET returned a different object identity')
                if offset:
                    wanted = f'bytes {offset}-{expected - 1}/{expected}'
                    if response.status != 206 or response.headers.get('Content-Range') != wanted:
                        raise ValueError(f'Server did not honor resume range for {url}')
                elif response.status != 200:
                    raise ValueError(f'Unexpected HTTP status {response.status}')
                if int(response.headers['Content-Length']) != expected - offset:
                    raise ValueError('Unexpected response length')
                with part.open('ab' if offset else 'wb') as stream:
                    while not STOP.is_set():
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            if STOP.is_set():
                return 'stopped'
            if not valid_parquet(part, expected):
                raise OSError(f'Incomplete download or invalid Parquet framing: {path.name}')
            part.replace(path)
            print(f'DONE {path.relative_to(root)} {expected:,} bytes', flush=True)
            return 'downloaded'
        except ValueError:
            raise  # Never automatically replace a changed object or bad local state.
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            print(f'RETRY {attempt + 1}/6 {url}: {error}', flush=True)
            STOP.wait(min(60, 2 ** (attempt + 1)))
    raise RuntimeError(f'Download failed after retries: {url}: {last_error}')


def progress(root, files, snapshot, started, initial_bytes, state='running', failures=None):
    stored = complete_bytes = completed = 0
    for entry in files:
        path, _ = object_paths(root, entry)
        part = path.with_suffix(path.suffix + '.part')
        expected = entry['meta']['content_length']
        try:
            if path.exists():
                size = path.stat().st_size
                stored += size
                if size == expected:
                    completed += 1
                    complete_bytes += size
            elif part.exists():
                stored += part.stat().st_size
        except FileNotFoundError:
            pass  # Atomic rename can race this observation.
    total = sum(e['meta']['content_length'] for e in files)
    seconds = max(1, time.monotonic() - started)
    rate = max(0, stored - initial_bytes) / seconds
    status = {
        'state': state, 'snapshot': snapshot, 'updated_at_unix': time.time(),
        'completed_files': completed, 'total_files': len(files),
        'stored_bytes_including_partial': stored, 'completed_bytes': complete_bytes,
        'total_bytes': total, 'percent': round(100 * stored / total, 3),
        'average_MB_per_second_this_run': round(rate / 1e6, 2),
        'estimated_hours_remaining': round((total - stored) / rate / 3600, 2) if rate else None,
        'failures': failures or [],
    }
    atomic_json(root / 'status.json', status)
    return stored, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/mnt/bulk/openalex'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, help='Download first N files only, for a smoke test')
    args = parser.parse_args()
    # Prevent accidentally filling the root filesystem if the data mount is absent.
    if not os.path.ismount('/mnt/bulk'):
        raise RuntimeError('/mnt/bulk is not mounted')
    args.root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.root / '.download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())
    manifest_path = args.root / 'manifest.parquet.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        with request(MANIFEST_URL) as response:
            manifest = json.load(response)
        atomic_json(manifest_path, manifest)
    if manifest['format'] != 'parquet':
        raise ValueError('Not a Parquet manifest')
    entity = next(e for e in manifest['entities'] if e['entity'] == 'works')
    files = entity['files'][:args.limit] if args.limit else entity['files']
    if not args.limit and sum(f['meta']['content_length'] for f in files) != entity['content_length']:
        raise ValueError('Manifest total does not match listed file sizes')
    started = time.monotonic()
    initial, _ = progress(args.root, files, manifest['date'], started, 0, state='starting')
    print(f'START snapshot={manifest["date"]} files={len(files)} bytes={sum(f["meta"]["content_length"] for f in files)} workers={args.workers}', flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, args.root, entry): entry['url'] for entry in files}
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=20, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except Exception as error:
                    failures.append({'url': futures[future], 'error': str(error)})
                    print(f'FAILED {futures[future]}: {error}', flush=True)
                    STOP.set()
            _, status = progress(args.root, files, manifest['date'], started, initial, failures=failures)
            if not done:
                print('PROGRESS ' + json.dumps(status), flush=True)
    state = 'failed' if failures else 'paused' if STOP.is_set() else 'complete'
    _, status = progress(args.root, files, manifest['date'], started, initial, state=state, failures=failures)
    print('FINAL ' + json.dumps(status), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
