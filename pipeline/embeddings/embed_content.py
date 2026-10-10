"""Resumable DenseOn embeddings for a pinned content-table selection.

Use this directory's locked uv environment. Output is separate from the content
Parquet files. Source files are pinned by hard links and the model by commit SHA.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MODEL = 'lightonai/DenseOn'
DIM = 768
FORMAT = 1
STOP = False


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    sync_dir(path.parent)


@contextmanager
def locked(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.writer.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process holds {out}/.writer.lock') from None
        yield


def file_signature(path):
    s = path.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino, 'device': s.st_dev}


# Fields that identify a file's version. Not 'device': on btrfs, st_dev is an anonymous number given at
# mount time and changes across reboots for the very same file (2026-10-09: 103 -> 82 for all 594 inputs,
# size, mtime and inode unchanged), which stopped the backfill after a reboot.
STABLE = ('bytes', 'mtime_ns', 'inode')


def check_source(entry):
    path = Path(entry['snapshot_path'])
    now, pinned = file_signature(path), entry['signature']
    if any(now[k] != pinned[k] for k in STABLE):
        raise RuntimeError(f'Pinned source changed: {path}; refusing to mix input versions')
    return path


def footer_digest(path):
    with path.open('rb') as f:
        f.seek(-8, 2); trailer = f.read(8)
        if trailer[4:] != b'PAR1':
            raise ValueError(f'Not a Parquet file: {path}')
        length = int.from_bytes(trailer[:4], 'little')
        f.seek(-8-length, 2)
        return hashlib.sha256(f.read(length+8)).hexdigest()


SELECTIONS = {
    'complete': 'is_complete == true; no extra type/OA/title filter',
    'has_abstract': 'non-empty abstract after whitespace trimming; all languages and reference coverage; no type/OA/title filter',
    'has_english_abstract': "language == 'en' and non-empty abstract after whitespace trimming; references optional; no type/OA/title filter",
}


def selection_columns(policy):
    if policy not in SELECTIONS:
        raise ValueError(f'Unknown selection policy: {policy}')
    if policy == 'complete':
        return ['is_complete']
    return ['abstract', 'language'] if policy == 'has_english_abstract' else ['abstract']


def selection_mask(table, policy):
    if policy == 'complete':
        return pc.fill_null(table['is_complete'], False)
    if policy == 'has_abstract':
        return pc.fill_null(pc.not_equal(pc.utf8_trim_whitespace(table['abstract']), ''), False)
    if policy == 'has_english_abstract':
        return pc.and_(selection_mask(table, 'has_abstract'), pc.fill_null(pc.equal(table['language'], 'en'), False))
    raise ValueError(f'Unknown selection policy: {policy}')


def date_strings(column):
    # The production Parquet stores date32, while some inputs/tests use strings.
    # Normalize in Arrow before converting to Python (also handles out-of-range years).
    if pa.types.is_timestamp(column.type):
        return pc.strftime(column, format='%Y-%m-%d')
    if pa.types.is_date(column.type) or pa.types.is_null(column.type):
        return pc.cast(column, pa.string())
    return column


def canonical_date(value):
    if isinstance(value, datetime):
        value = value.date().isoformat()
    elif isinstance(value, date):
        value = value.isoformat()
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def plan(source, snapshot, out, revision=None, shard_rows=10000, selection='complete',
         order='source'):
    from huggingface_hub import HfApi, snapshot_download
    source, snapshot, out = source.resolve(), snapshot.resolve(), out.resolve()
    if order not in ('source', 'publication_date_desc'):
        raise ValueError('Unsupported ordering policy')
    count_columns = selection_columns(selection)
    if order == 'publication_date_desc':
        count_columns = list(dict.fromkeys(count_columns + ['date']))
    if shard_rows <= 0:
        raise ValueError('rows_per_shard must be positive')
    if (out / 'manifest.json').exists():
        raise ValueError('Manifest already exists. Resume with run; do not silently re-plan an existing output.')
    if not revision or not re.fullmatch(r'[a-f0-9]{40}', revision):
        revision = HfApi().model_info(MODEL, revision=revision or 'main').sha
    model_path = snapshot_download(MODEL, revision=revision, allow_patterns=['*.json', '*.safetensors', '*.txt'])
    with locked(out):
        files = sorted(source.glob('year=*/data.parquet'), key=lambda p: int(p.parent.name.split('=')[1]), reverse=True)
        if not files:
            raise ValueError(f'No content Parquet files in {source}')
        entries = []
        total = 0
        planned_source_rows = 0
        last_progress = 0.0
        cache_root = out / 'planning'
        cache_root.mkdir(exist_ok=True)
        policy_key = digest_json({'selection': selection, 'order': order, 'snapshot': str(snapshot),
                                  'planner_version': 3})
        def planning_status(i, src, group=None):
            atomic_json(out / 'status.json', {'state':'planning', 'updated_utc':now(),
                'completed_rows':0, 'expected_rows':None, 'eligible_rows_counted':total,
                'source_rows_scanned':planned_source_rows, 'files_completed':i, 'files_total':len(files),
                'current_file':str(src.relative_to(source)), 'current_row_group':group,
                'ordering':order, 'selection_policy':selection})
        for i, src in enumerate(files):
            original = file_signature(src)
            dest = snapshot / src.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                try:
                    os.link(src, dest)
                except OSError as e:
                    raise RuntimeError('Snapshot must be on the same filesystem as the content table (hard links, not a 145 GB copy)') from e
            if file_signature(dest) != original:
                raise RuntimeError(f'Snapshot/source mismatch: {src}. Choose a fresh snapshot path.')
            cache_file = cache_root / f'{src.parent.name}.json'
            if cache_file.exists():
                cached = json.loads(cache_file.read_text())
                if cached.get('policy_key') == policy_key and cached['entry']['signature'] == original:
                    entry = cached['entry']
                    entries.append(entry)
                    total += sum(g['selected_rows'] for g in entry['row_groups'])
                    planned_source_rows += sum(g['source_rows'] for g in entry['row_groups'])
                    planning_status(i+1,src)
                    continue
            pf = pq.ParquetFile(dest)
            for column in set(['id', 'title', 'abstract'] + count_columns):
                if column not in pf.schema_arrow.names:
                    raise ValueError(f'Missing {column} in {src}')
            if selection == 'complete' and pf.schema_arrow.field('is_complete').type != pa.bool_():
                raise TypeError('is_complete must be a Boolean column')
            groups = []
            for g in range(pf.metadata.num_row_groups):
                selected = 0
                date_counts = {}
                for batch in pf.iter_batches(batch_size=32768, row_groups=[g], columns=count_columns, use_threads=False):
                    table = pa.Table.from_batches([batch])
                    flags = selection_mask(table, selection)
                    selected += int(pc.sum(pc.cast(flags, pa.int64())).as_py() or 0)
                    if order == 'publication_date_desc':
                        for item in pc.value_counts(pc.filter(date_strings(table['date']),flags)).to_pylist():
                            day = canonical_date(item['values'])
                            date_counts[day] = date_counts.get(day,0) + item['counts']
                group = {'row_group':g, 'source_rows':pf.metadata.row_group(g).num_rows,
                         'selected_rows':selected}
                if order == 'publication_date_desc':
                    group['dates'] = [{'date':day,'rows':n} for day,n in date_counts.items()]
                    assert sum(date_counts.values()) == selected
                groups.append(group)
                total += selected
                planned_source_rows += group['source_rows']
                if time.monotonic() - last_progress > 5:
                    planning_status(i,src,g)
                    last_progress = time.monotonic()
                if STOP:
                    # Completed source files are reusable; the in-flight file is
                    # recounted after restart. No model work has started.
                    planning_status(i,src,g)
                    return None
            if file_signature(src) != original:
                raise RuntimeError(f'Source changed during planning: {src}')
            entry = {'relative_path':str(src.relative_to(source)), 'source_path':str(src),
                     'snapshot_path':str(dest), 'signature':original,
                     'footer_sha256':footer_digest(dest), 'row_groups':groups}
            entries.append(entry)
            atomic_json(cache_file, {'policy_key':policy_key,'entry':entry})
            planning_status(i+1,src)
            if (i+1) % 50 == 0:
                print(f'PLAN {i+1}/{len(files)} files; {total:,} eligible rows so far', flush=True)
        # Detect a concurrent publication spanning multiple year files.
        for entry in entries:
            if file_signature(Path(entry['source_path'])) != entry['signature']:
                raise RuntimeError('Content table changed while taking snapshot; choose a fresh snapshot and re-plan')
        embedding = {'model': MODEL, 'revision': revision, 'model_snapshot': model_path,
                     'dimensions': DIM, 'max_seq_length': 512, 'document_prompt': 'document: ',
                     'query_prompt': 'query: ', 'input_format': "((title or '') + ' ' + abstract).strip()",
                     'inference_dtype': 'bfloat16', 'normalize': 'float32 L2 normalization',
                     'storage_dtype': 'float16', 'text_hash': 'BLAKE2b-128 of full joined text UTF-8'}
        doc = {'format_version': FORMAT, 'created_utc': now(), 'source_root': str(source),
               'snapshot_root': str(snapshot), 'selection': SELECTIONS[selection], 'selection_policy': selection,
               'ordering': order, 'date_normalization':'arrow_date_v1', 'embedding': embedding, 'rows_per_shard': shard_rows, 'expected_rows': total,
               'expected_vector_bytes': total * DIM * 2, 'files': entries,
               'versions': {name: importlib.metadata.version(name) for name in
                            ('torch', 'sentence-transformers', 'transformers', 'pyarrow', 'numpy')}}
        doc['plan_id'] = digest_json(doc)
        atomic_json(out / 'manifest.json', doc)
        atomic_json(out / 'plan-summary.json', public_plan_summary(doc))
        atomic_json(out / 'status.json', {'state': 'planned', 'updated_utc': now(), 'plan_id': doc['plan_id'],
                                         'completed_rows': 0, 'expected_rows': total})
        print(json.dumps({k: doc[k] for k in ('plan_id', 'expected_rows', 'expected_vector_bytes')}, indent=2), flush=True)
        return doc


def public_plan_summary(doc):
    return {'plan_id':doc['plan_id'], 'expected_rows':doc['expected_rows'],
            'ordering':doc.get('ordering','source'), 'selection':doc['selection'],
            'model':doc['embedding']['model'], 'revision':doc['embedding']['revision'],
            'max_tokens':doc['embedding']['max_seq_length'], 'dimensions':doc['embedding']['dimensions'],
            'source_rows':sum(g['source_rows'] for f in doc['files'] for g in f['row_groups'])}


def read_manifest(out):
    doc = json.loads((out / 'manifest.json').read_text())
    checksum = doc.pop('plan_id')
    if digest_json(doc) != checksum or doc['format_version'] != FORMAT:
        raise ValueError('Manifest checksum/version mismatch')
    doc['plan_id'] = checksum
    # Missing policy denotes the legacy complete-only plan; never reinterpret it.
    selection_columns(doc.get('selection_policy', 'complete'))
    return doc


def shard_tasks(doc):
    size = doc['rows_per_shard']
    if doc.get('ordering') == 'publication_date_desc':
        dated = []
        for fi, entry in enumerate(doc['files']):
            year = Path(entry['relative_path']).parent.name
            for group in entry['row_groups']:
                for bucket in group['dates']:
                    for start in range(0,bucket['rows'],size):
                        tag = bucket['date'] or 'undated'
                        task = {'key':f'{year}/rg-{group["row_group"]:06d}/part-{tag}-{start:09d}',
                                'file_index':fi,'row_group':group['row_group'],
                                'publication_date':bucket['date'], 'date_rows':bucket['rows'],
                                'selected_start':start, 'rows':min(size,bucket['rows']-start)}
                        dated.append(task)
        # Global date order, not merely descending year folders. Even an inconsistent
        # source publication-year field cannot put an older date ahead of a newer one.
        dated.sort(key=lambda t:(t['publication_date'] or '', -t['file_index'],
                                 -t['row_group'], -t['selected_start']), reverse=True)
        yield from dated
        return
    for fi, entry in enumerate(doc['files']):
        year = Path(entry['relative_path']).parent.name
        for group in entry['row_groups']:
            count = group['selected_rows']
            for start in range(0, count, size):
                rows = min(size, count-start)
                key = f'{year}/rg-{group["row_group"]:06d}/part-{start:09d}'
                yield {'key': key, 'file_index': fi, 'row_group': group['row_group'],
                       'selected_start': start, 'rows': rows}


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*2**20), b''):
            h.update(block)
    return h.hexdigest()


def verify_shard(out, task, doc, checksums=True):
    path = out / 'shards' / task['key']
    marker = path / 'commit.json'
    if not path.exists():
        return None
    if not marker.exists():
        raise ValueError(f'Incomplete published shard: {path}')
    commit = json.loads(marker.read_text())
    if commit['plan_id'] != doc['plan_id'] or commit['task'] != task:
        raise ValueError(f'Shard belongs to a different plan/task: {path}')
    shapes = {'work_ids.npy': ((task['rows'],), np.dtype('uint64')),
              'vectors.npy': ((task['rows'], doc['embedding']['dimensions']), np.dtype('float16')),
              'text_hashes.npy': ((task['rows'], 16), np.dtype('uint8'))}
    for name, (shape, dtype) in shapes.items():
        f = path / name
        info = commit['files'][name]
        if f.stat().st_size != info['bytes']:
            raise ValueError(f'Shard file size changed: {f}')
        array = np.load(f, mmap_mode='r', allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f'Shard array schema changed: {f}')
        if checksums and sha_file(f) != info['sha256']:
            raise ValueError(f'Shard checksum failed: {f}')
    return commit


def normalized_f16(vectors, rows, dim):
    a = np.asarray(vectors, dtype=np.float32)
    if a.shape != (rows, dim) or not np.isfinite(a).all():
        raise FloatingPointError(f'Invalid embedding values/shape: {a.shape}; wanted {(rows, dim)}')
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise FloatingPointError('Zero or non-finite embedding norm')
    result = (a / norms).astype(np.float16)
    if not np.isfinite(result).all():
        raise FloatingPointError('Non-finite storage vector')
    return result


def records(table):
    ids, texts, hashes = [], [], []
    for item in table.to_pylist():
        match = re.fullmatch(r'(?:https://openalex\.org/)?W([0-9]+)', item['id'] or '')
        if not match:
            raise ValueError(f'Invalid OpenAlex work ID: {item["id"]!r}')
        wid = int(match[1])
        if not 0 < wid < 2**64:
            raise ValueError('OpenAlex ID outside uint64 range')
        abstract = item['abstract']
        if not isinstance(abstract, str) or not abstract.strip():
            raise ValueError(f'Selected row W{wid} has an empty abstract; refusing to silently skip it')
        title = item['title'] or ''
        text = f'{title} {abstract}'.strip()
        ids.append(wid); texts.append(text)
        hashes.append(hashlib.blake2b(text.encode(), digest_size=16).digest())
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate work IDs within source shard')
    return np.asarray(ids, dtype=np.uint64), texts, np.frombuffer(b''.join(hashes), dtype=np.uint8).reshape(-1,16)


def commit_shard(out, task, doc, ids, hashes, vectors, stats):
    n, dim = task['rows'], doc['embedding']['dimensions']
    if ids.shape != (n,) or ids.dtype != np.uint64 or hashes.shape != (n,16) or hashes.dtype != np.uint8:
        raise ValueError('Invalid shard identity arrays')
    if vectors.shape != (n,dim) or vectors.dtype != np.float16 or not np.isfinite(vectors).all():
        raise FloatingPointError('Invalid vectors before commit')
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    if not np.allclose(norms, 1.0, atol=0.002, rtol=0):
        raise FloatingPointError('Vectors are not normalized')
    final = out / 'shards' / task['key']
    if final.exists():
        raise FileExistsError(f'Refusing to overwrite committed shard: {final}')
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name('.' + final.name + '.tmp-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        files = {}
        for name, array in [('work_ids.npy', ids), ('text_hashes.npy', hashes), ('vectors.npy', vectors)]:
            f = temporary / name
            with f.open('wb') as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
            files[name] = {'bytes': f.stat().st_size, 'sha256': sha_file(f)}
        marker = {'plan_id': doc['plan_id'], 'task': task, 'files': files,
                  'committed_utc': now(), 'stats': stats}
        atomic_json(temporary / 'commit.json', marker)
        sync_dir(temporary)
        os.rename(temporary, final)
        sync_dir(final.parent)
        return marker
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def audit_unique_ids(out, doc):
    """Global ID audit uses ~8 bytes/row plus a Boolean comparison, not all vectors."""
    ids = np.empty(doc['expected_rows'], dtype=np.uint64)
    cursor = 0
    for task in shard_tasks(doc):
        part = np.load(out / 'shards' / task['key'] / 'work_ids.npy', allow_pickle=False)
        ids[cursor:cursor+len(part)] = part
        cursor += len(part)
    if cursor != doc['expected_rows']:
        raise ValueError('Global ID audit: row count mismatch')
    ids.sort()
    repeated = int(np.count_nonzero(ids[1:] == ids[:-1]))
    if repeated:
        raise ValueError(f'Global ID audit: {repeated} repeated work IDs across shards')
    return cursor


def gpu_candidates():
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    result=[]
    for line in p.stdout.splitlines():
        index, uid, free, total, utilization = [s.strip() for s in line.split(',')]
        result.append({'index': int(index), 'uuid': uid, 'free_mib': int(free),
                       'total_mib': int(total), 'utilization': int(utilization)})
    return result


class DenseEncoder:
    def __init__(self, config, device, batch_size, cpu_threads=6, allocation_gib=None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        torch.set_num_threads(cpu_threads)
        self.batch_size = batch_size
        self.device = device
        if device.startswith('cuda'):
            ix = int(device.split(':')[1])
            if allocation_gib is not None:
                total = torch.cuda.get_device_properties(ix).total_memory
                torch.cuda.set_per_process_memory_fraction(min(.8, allocation_gib * 2**30 / total), ix)
        dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
        self.model = SentenceTransformer(config['model_snapshot'], device=device,
                                         trust_remote_code=False, local_files_only=True,
                                         model_kwargs={'dtype': dtype})
        if self.model.get_sentence_embedding_dimension() != config['dimensions']:
            raise ValueError('Model dimension differs from manifest')
        if self.model.prompts.get('document') != config['document_prompt'] or self.model.prompts.get('query') != config['query_prompt']:
            raise ValueError('Model prompts differ from manifest')
        self.model.max_seq_length = config['max_seq_length']
        self.config = config

    def encode(self, texts):
        while True:
            try:
                result = self.model.encode(texts, batch_size=self.batch_size, prompt_name='document',
                                           show_progress_bar=False, normalize_embeddings=False,
                                           convert_to_tensor=True)
                array = result.float().cpu().numpy()
                break
            except self.torch.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if self.batch_size <= 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                print(f'GPU OOM: retrying uncommitted shard with batch size {self.batch_size}', flush=True)
        return normalized_f16(array, len(texts), self.config['dimensions'])

    def truncation_sample(self, texts, ids):
        # Stable 0.1% ID-hash sample, not a biased sample of short/long years.
        selected=[i for i,w in enumerate(ids) if int.from_bytes(hashlib.blake2b(int(w).to_bytes(8,'little'),digest_size=8).digest(),'little') % 1000 == 0]
        if not selected:
            return {'token_sample_rows': 0, 'token_sample_truncated': 0}
        strings=[self.config['document_prompt'] + texts[i] for i in selected]
        lengths=self.model.tokenizer(strings, truncation=True, max_length=self.config['max_seq_length']+1,
                                     return_length=True, return_attention_mask=False,
                                     return_token_type_ids=False)['length']
        return {'token_sample_rows': len(lengths),
                'token_sample_truncated': sum(n > self.config['max_seq_length'] for n in lengths)}


def handle_stop(*_):
    global STOP
    STOP = True


def run(out, device='auto', wait_gpu=True, min_free_gib=4, batch_size=64,
        cpu_threads=6, max_shards=None, encoder_factory=DenseEncoder):
    doc = read_manifest(out)
    if device not in ('auto','cpu') and not re.fullmatch(r'cuda:[0-9]+', device):
        raise ValueError('Device must be auto, cpu, or cuda:N')
    if batch_size < 1 or min_free_gib < 2:
        raise ValueError('batch_size must be positive; minimum GPU headroom must be at least 2 GiB')
    with locked(out):
        start = time.monotonic()
        active_started = None
        done = rows = new_rows = 0
        gpu_identity = None
        sample_n = sample_truncated = 0
        expected_by_year = {}
        done_by_year = {}
        for entry in doc['files']:
            for g in entry['row_groups']:
                buckets = g.get('dates', [{'date':None, 'rows':g['selected_rows']}])
                for b in buckets:
                    year = b['date'][:4] if b['date'] else 'undated'
                    expected_by_year[year] = expected_by_year.get(year,0) + b['rows']
        previous = {}
        if (out/'status.json').exists():
            previous = json.loads((out/'status.json').read_text())
            if previous.get('plan_id') != doc['plan_id']:
                previous = {}
        current_day = None
        recent_points = []
        def status(state, **extra):
            elapsed = time.monotonic() - start
            active_seconds = time.monotonic() - active_started if active_started is not None else 0
            rate = new_rows / active_seconds if new_rows and active_seconds else None
            verifying = state == 'verifying_committed_shards'
            display_rows = max(rows,previous.get('completed_rows',0)) if verifying else rows
            years = [{'year':y,'expected':n,'completed':done_by_year.get(y,0)}
                     for y,n in sorted(expected_by_year.items(),key=lambda x:x[0] if x[0]!='undated' else '',reverse=True) if n]
            if verifying and previous.get('years'):
                years = previous['years']
            value={'state': state, 'updated_utc': now(), 'plan_id': doc['plan_id'], 'pid': os.getpid(),
                   'expected_rows': doc['expected_rows'], 'completed_rows': display_rows, 'completed_shards': done,
                   'verified_rows': rows if verifying else None,
                   'percent': 100 * display_rows / doc['expected_rows'] if doc['expected_rows'] else 100,
                   'ordering':doc.get('ordering','source'), 'current_publication_date':current_day,
                   'years':years, 'recent_progress':recent_points[-40:],
                   'new_rows_this_run': new_rows, 'elapsed_seconds': elapsed, 'active_seconds': active_seconds, 'rows_per_second_this_run': rate,
                   'estimated_remaining_hours': (doc['expected_rows']-rows)/rate/3600 if rate and new_rows>=10000 else None,
                   'token_sample_rows': sample_n, 'token_sample_truncated': sample_truncated,
                   'gpu': gpu_identity, **extra}
            atomic_json(out / 'status.json', value)
        try:
            status('verifying_committed_shards')
            # Only these owned, uncommitted temporary shard directories are removed.
            for partial in (out / 'shards').glob('year=*/rg-*/.part-*.tmp-*'):
                shutil.rmtree(partial)
            for entry in doc['files']:
                check_source(entry)
            remaining=[]
            last_verify_update = 0.0
            for task in shard_tasks(doc):
                c = verify_shard(out, task, doc)
                if c:
                    done += 1; rows += task['rows']
                    sample_n += c['stats'].get('token_sample_rows', 0)
                    sample_truncated += c['stats'].get('token_sample_truncated', 0)
                    year = (task.get('publication_date') or '')[:4] or 'undated'
                    done_by_year[year] = done_by_year.get(year,0) + task['rows']
                else:
                    remaining.append(task)
                if time.monotonic()-last_verify_update > 3:
                    status('verifying_committed_shards')
                    last_verify_update = time.monotonic()
            if not remaining:
                assert rows == doc['expected_rows']
                status('verifying_unique_ids')
                audit_unique_ids(out, doc)
                status('complete', unique_ids_verified=True)
                return
            free_needed = (doc['expected_rows']-rows) * (doc['embedding']['dimensions']*2+24) + 5*2**30
            if shutil.disk_usage(out).free < free_needed:
                raise RuntimeError('Not enough output disk space for the remaining vectors plus headroom')
            allocation = None
            if device != 'cpu':
                while not STOP:
                    candidates = gpu_candidates()
                    acceptable = [g for g in candidates if g['free_mib'] >= min_free_gib*1024
                                  and g['utilization'] <= 30
                                  and (device == 'auto' or device == f"cuda:{g['index']}")]
                    if acceptable:
                        chosen = max(acceptable, key=lambda g: g['free_mib'])
                        gpu_identity = chosen
                        # Resolve physical NVIDIA index to UUID before importing torch;
                        # this avoids CUDA ordinal/order ambiguity on multi-GPU hosts.
                        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                        os.environ['CUDA_VISIBLE_DEVICES'] = chosen['uuid']
                        device = 'cuda:0'
                        allocation = min(8, chosen['free_mib']/1024 - 1.5)
                        break
                    status('waiting_for_gpu', gpu_candidates=candidates, minimum_free_gib=min_free_gib)
                    if not wait_gpu:
                        raise RuntimeError('No GPU has enough free memory; no other services were stopped')
                    for _ in range(30):
                        if STOP: break
                        time.sleep(1)
            if STOP:
                status('paused'); return
            status('loading_model', device=device, allocation_gib=allocation)
            encoder = encoder_factory(doc['embedding'], device, batch_size, cpu_threads, allocation)
            active_started = time.monotonic()
            loaded_key = None
            source_key = None
            source_table = table = None
            committed_this_run = 0
            for task in remaining:
                if STOP or (max_shards is not None and committed_this_run >= max_shards):
                    status('paused', reason='stop requested or shard limit reached', device=device)
                    return
                entry = doc['files'][task['file_index']]
                source = check_source(entry)
                base_key = (task['file_index'], task['row_group'])
                current_day = task.get('publication_date')
                key = (*base_key,current_day)
                group = entry['row_groups'][task['row_group']]
                dated = doc.get('ordering') == 'publication_date_desc'
                if source_key != base_key:
                    policy = doc.get('selection_policy', 'complete')
                    columns = list(dict.fromkeys(['id','title','abstract'] + selection_columns(policy) + (['date'] if dated else [])))
                    source_table = pq.ParquetFile(source).read_row_group(task['row_group'], columns=columns, use_threads=False)
                    source_table = source_table.filter(selection_mask(source_table, policy))
                    if dated:
                        source_table = source_table.set_column(source_table.schema.get_field_index('date'), 'date',
                                                               date_strings(source_table['date']))
                    if len(source_table) != group['selected_rows']:
                        raise ValueError(f'Eligible-row count changed: {source}, group {base_key[1]}')
                    source_key = base_key
                if loaded_key != key:
                    table = source_table
                    if dated:
                        if current_day is not None:
                            mask = pc.fill_null(pc.equal(table['date'],current_day),False)
                        else:
                            valid_days = [b['date'] for b in group['dates'] if b['date'] is not None]
                            mask = pc.invert(pc.is_in(table['date'],value_set=pa.array(valid_days,type=table['date'].type)))
                        table = table.filter(mask).sort_by([('id','descending')])
                        if len(table) != task['date_rows']:
                            raise ValueError('Date-bucket row count differs from pinned plan')
                    table = table.select(['id','title','abstract'])
                    loaded_key = key
                ids, texts, hashes = records(table.slice(task['selected_start'], task['rows']))
                status('encoding', device=device, current_shard=task['key'], batch_size=encoder.batch_size)
                t = time.monotonic()
                vectors = encoder.encode(texts)
                if STOP:
                    status('paused', reason='stop requested; uncommitted shard will be retried', device=device)
                    return
                stats = {'device': device, 'batch_size': encoder.batch_size,
                         'inference_dtype': 'float32' if device=='cpu' else 'bfloat16',
                         'gpu': gpu_identity,
                         'encode_seconds': time.monotonic()-t, **encoder.truncation_sample(texts,ids)}
                check_source(entry)
                commit_shard(out, task, doc, ids, hashes, vectors, stats)
                done += 1; rows += task['rows']; new_rows += task['rows']; committed_this_run += 1
                sample_n += stats['token_sample_rows']; sample_truncated += stats['token_sample_truncated']
                year = current_day[:4] if current_day else 'undated'
                done_by_year[year] = done_by_year.get(year,0) + task['rows']
                if not recent_points or time.monotonic()-recent_points[-1]['monotonic']>=10:
                    recent_points.append({'utc':now(),'monotonic':time.monotonic(),'completed_rows':rows,
                                          'rows_per_second':new_rows/max(time.monotonic()-active_started,.001)})
                    recent_points[:] = recent_points[-40:]
                status('running', device=device, last_committed_shard=task['key'], batch_size=encoder.batch_size)
                print(f'COMMITTED {task["key"]}: {rows:,}/{doc["expected_rows"]:,} ({100*rows/doc["expected_rows"]:.3f}%)', flush=True)
            if rows != doc['expected_rows']:
                raise AssertionError('Final row count differs from manifest')
            status('verifying_unique_ids', device=device)
            audit_unique_ids(out, doc)
            status('complete', device=device, unique_ids_verified=True)
        except BaseException as e:
            status('failed', error=f'{type(e).__name__}: {e}')
            raise


def verify(out):
    doc = read_manifest(out)
    committed = rows = 0
    missing=[]
    for entry in doc['files']:
        check_source(entry)
    for task in shard_tasks(doc):
        if verify_shard(out, task, doc):
            committed += 1; rows += task['rows']
        else: missing.append(task['key'])
    complete = rows == doc['expected_rows'] and not missing
    if complete:
        audit_unique_ids(out, doc)
    result = {'committed_shards':committed,'completed_rows':rows,'expected_rows':doc['expected_rows'],
              'missing_shards':len(missing),'complete':complete, 'unique_ids_verified':complete}
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('plan')
    prepare.add_argument('--source',type=Path,default=Path('/mnt/bulk/openalex/content'))
    prepare.add_argument('--snapshot',type=Path,default=Path('/mnt/bulk/openalex/embedding-inputs/denseon-complete'))
    prepare.add_argument('--output',type=Path,default=Path('/mnt/fast/openalex-embeddings/denseon-complete'))
    prepare.add_argument('--revision')
    prepare.add_argument('--shard-rows',type=int,default=10000)
    prepare.add_argument('--selection',choices=SELECTIONS,default='complete',
                         help='Use a fresh snapshot/output when changing selection; old plans stay immutable')
    prepare.add_argument('--order',choices=['publication_date_desc','source'],default='publication_date_desc',
                         help='Newest actual publication dates first; missing/invalid dates last')
    execute=sub.add_parser('run')
    execute.add_argument('--output',type=Path,default=Path('/mnt/fast/openalex-embeddings/denseon-complete'))
    execute.add_argument('--device',default='auto')
    execute.add_argument('--min-free-gib',type=float,default=4)
    execute.add_argument('--batch-size',type=int,default=64)
    execute.add_argument('--cpu-threads',type=int,default=6)
    execute.add_argument('--max-shards',type=int)
    execute.add_argument('--no-wait-gpu',action='store_true')
    for verb in ('verify','status'):
        s=sub.add_parser(verb);s.add_argument('--output',type=Path,default=Path('/mnt/fast/openalex-embeddings/denseon-complete'))
    a=p.parse_args()
    signal.signal(signal.SIGTERM, handle_stop); signal.signal(signal.SIGINT, handle_stop)
    if a.command=='plan':plan(a.source,a.snapshot,a.output,a.revision,a.shard_rows,a.selection,a.order)
    elif a.command=='run':run(a.output,a.device,not a.no_wait_gpu,a.min_free_gib,a.batch_size,a.cpu_threads,a.max_shards)
    elif a.command=='verify':verify(a.output)
    else:print((a.output/'status.json').read_text())

if __name__=='__main__':main()
