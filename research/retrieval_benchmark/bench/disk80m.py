"""Storage/compute prototype, NOT a retrieval-quality benchmark.

Physically materialize 80M SQ8 entries using repeated codes from the real 1.2M
DenseOn index. Use 65,536 synthetic child centroids and a RAM-resident HNSW router.
This measures 80M-sized I/O/scanning, NOT recall on 80M distinct abstracts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import faiss
import numpy as np

from .common import DATA, DATASETS, write_json

ROOT = DATA / 'disk80m'
SOURCE = DATA / 'index2/faiss/denseon/beir4+openalex1m.ivf4096_sq8.index'
DIM = 768


def ondisk_layout(path, counts, code_size):
    """Attach an already-written compact [codes, IDs] file without copying it."""
    inv = faiss.OnDiskInvertedLists(len(counts), code_size, str(path))
    entries = faiss.OnDiskOneListVector()
    offset = 0
    for n in counts:
        item = faiss.OnDiskOneList()
        item.size = item.capacity = int(n)
        item.offset = offset
        entries.push_back(item)
        offset += int(n) * (code_size + 8)
    if path.stat().st_size != offset:
        raise ValueError('Inverted-list file size does not match its layout')
    inv.lists = entries
    inv.totsize = offset
    inv.read_only = True
    inv.do_mmap()
    inv.prefetch_nthread = 4
    return inv


def build(root=ROOT, total=80_000_000, nlist=65536, threads=16):
    import subprocess
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['chattr', '+m', str(root)], check=True)
    manifest = root / 'manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text())
        assert m['entries'] == total and m['lists'] == nlist
        print('Reusing completed prototype', manifest, flush=True)
        return
    faiss.omp_set_num_threads(threads)
    source = faiss.read_index(str(SOURCE), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    parents = source.quantizer.reconstruct_n(0, source.nlist)
    rng = np.random.default_rng(20261005)
    parent_ids = np.arange(nlist) % source.nlist
    # These are NOT trained 80M centroids. Small perturbations avoid identical router
    # nodes while retaining rough scientific-topic locality for the I/O experiment.
    centroids = parents[parent_ids] + rng.normal(0, 0.02 / np.sqrt(DIM), (nlist, DIM)).astype('float32')
    np.save(root / 'centroids.npy', centroids)
    router = faiss.IndexHNSWFlat(DIM, 16, faiss.METRIC_INNER_PRODUCT)
    router.hnsw.efConstruction = 80
    t0 = time.monotonic()
    router.add(centroids)
    print(f'Router built: {nlist:,} nodes in {time.monotonic()-t0:.1f}s', flush=True)
    counts = np.full(nlist, total // nlist, dtype=np.int64)
    counts[:total % nlist] += 1
    temp = root / 'postings.data.part'
    next_id = 0
    nonempty = [i for i in range(source.nlist) if source.invlists.list_size(i)]
    t0 = time.monotonic()
    with temp.open('wb', buffering=8 * 2**20) as f:
        for cell, n in enumerate(counts):
            parent = int(parent_ids[cell])
            if not source.invlists.list_size(parent):
                parent = nonempty[parent % len(nonempty)]
            nr = source.invlists.list_size(parent)
            ptr = source.invlists.get_codes(parent)
            try:
                original = faiss.rev_swig_ptr(ptr, nr * source.code_size).reshape(nr, source.code_size)
                # Reused real SQ8 codes, synthetic unique IDs. This is deliberate
                # capacity simulation; repeated content makes accuracy meaningless.
                select = (np.arange(n) + cell * 17) % nr
                f.write(original[select].tobytes())
            finally:
                source.invlists.release_codes(parent, ptr)
            f.write(np.arange(next_id, next_id + n, dtype='<i8').tobytes())
            next_id += int(n)
            if (cell + 1) % 4096 == 0:
                print(f'Written {next_id:,}/{total:,} entries ({f.tell()/1e9:.1f} GB)', flush=True)
        f.flush()
        os.fsync(f.fileno())
    postings = root / 'postings.data'
    temp.replace(postings)
    inv = ondisk_layout(postings, counts, source.code_size)
    index = faiss.IndexIVFScalarQuantizer(router, DIM, nlist, source.sq.qtype, faiss.METRIC_INNER_PRODUCT)
    index.sq = source.sq
    index.by_residual = source.by_residual
    index.is_trained = True
    index.replace_invlists(inv, False)
    index.ntotal = total
    faiss.write_index(index, str(root / 'index.faiss'))
    assert inv.compute_ntotal() == total
    # Cross-check that metadata can be loaded and a real scanner can consume the file.
    loaded = faiss.read_index(str(root / 'index.faiss'), faiss.IO_FLAG_READ_ONLY)
    loaded.nprobe = min(4, nlist)
    faiss.downcast_index(loaded.quantizer).hnsw.efSearch = 64
    scores, ids = loaded.search(parents[:1], 10)
    assert np.all((ids >= 0) & (ids < total)) and np.isfinite(scores).all()
    stat = postings.stat()
    write_json(manifest, {
        'experiment': '80M physical-entry capacity simulation, not 80M distinct papers',
        'entries': total, 'lists': nlist, 'dimension': DIM, 'code_bytes_per_vector': source.code_size,
        'source_entries': int(source.ntotal), 'source_index': str(SOURCE),
        'postings_bytes': stat.st_size, 'allocated_postings_bytes': stat.st_blocks * 512,
        'metadata_bytes': (root/'index.faiss').stat().st_size,
        'build_write_seconds': time.monotonic()-t0, 'seed': 20261005,
        'synthetic_centroids': 'existing 4096 centroids repeated, with seeded 0.02/sqrt(d) Gaussian perturbation; not retrained',
        'limitations': ['repeated vector codes; unique synthetic IDs', 'no accuracy inference',
                        'list lengths are balanced; real lists may be skewed',
                        'topic diversity remains that of the 1.2M source, not 80M papers']})
    print('BUILD COMPLETE', manifest, flush=True)


def prepare_queries(root=ROOT, per_dataset=32):
    from .beir_data import ROOT as BEIR_ROOT
    rng = np.random.default_rng(135)
    rows, vectors = [], []
    for ds in DATASETS:
        folder = DATA / 'emb/denseon' / ds
        qids = json.loads((folder / 'query_ids.json').read_text())
        q = np.load(folder / 'queries.npy').astype('float32')
        text = {}
        with (BEIR_ROOT / ds / 'queries.jsonl').open() as f:
            for line in f:
                item = json.loads(line)
                text[str(item['_id'])] = item['text']
        for i in rng.choice(len(qids), min(per_dataset, len(qids)), replace=False):
            rows.append({'dataset': ds, 'qid': qids[i], 'text': text[qids[i]]})
            vectors.append(q[i])
    order = rng.permutation(len(rows))
    np.save(root / 'queries.npy', np.asarray(vectors)[order])
    write_json(root / 'queries.json', [rows[i] for i in order])


def cg_path():
    return Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')


def io_bytes():
    return sum(int(x.split('=')[1]) for line in (cg_path()/'io.stat').read_text().splitlines()
               for x in line.split()[1:] if x.startswith('rbytes='))


def stats(values):
    s = sorted(values)
    return {'n':len(s), 'median_ms':statistics.median(s), 'p95_ms':s[round(.95*(len(s)-1))],
            'mean_ms':statistics.fmean(s)}


def worker(root, probes, workload, encode_query=False, prefetch=False):
    faiss.omp_set_num_threads(4)
    rows = json.loads((root/'queries.json').read_text())
    Q = np.load(root/'queries.npy')
    t = time.perf_counter()
    index = faiss.read_index(str(root/'index.faiss'), faiss.IO_FLAG_READ_ONLY)
    index.nprobe = probes
    index.parallel_mode = 1                 # parallelize list scanning, not query batches
    router = faiss.downcast_index(index.quantizer)
    router.hnsw.efSearch = max(128, probes * 2)
    faiss.downcast_InvertedLists(index.invlists).prefetch_nthread = 4
    load_ms = 1000*(time.perf_counter()-t)
    if encode_query:
        import torch
        torch.set_num_threads(4)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('lightonai/DenseOn', device='cpu')
        def embed(text):
            return model.encode([text], prompt_name='query', normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype('float32')
        embed('initialization query')
    centroids = np.load(root/'centroids.npy', mmap_mode='r') if workload=='dispersed' else None
    rng = np.random.default_rng(9123)
    times, out = [], []
    before = io_bytes()
    for j, row in enumerate(rows):
        t0 = time.perf_counter()
        q = embed(row['text']) if encode_query else np.ascontiguousarray(Q[j:j+1])
        t1 = time.perf_counter()
        distances, cells = router.search(q, probes)
        if workload == 'dispersed':
            # Control experiment to avoid optimistic routing/cache locality inherited
            # from duplicated source data. No relevance interpretation whatsoever.
            cells = rng.choice(index.nlist, probes, replace=False).reshape(1,-1).astype('int64')
            distances = (np.asarray(centroids[cells[0]]) @ q[0]).reshape(1,-1).astype('float32')
        t2 = time.perf_counter()
        if prefetch:
            # IndexIVF.search normally does this before search_preassigned. Keep the
            # operation inside the timed scan stage when splitting router timings.
            index.invlists.prefetch_lists(faiss.swig_ptr(cells), cells.size)
        scores, labels = index.search_preassigned(q, 100, cells, distances)
        t3 = time.perf_counter()
        assert np.isfinite(scores).all() and (labels >= 0).all()
        record = {'dataset':row['dataset'], 'qid':row['qid'], 'encoding_ms':1000*(t1-t0),
                  'routing_ms':1000*(t2-t1), 'scan_ms':1000*(t3-t2),
                  'search_ms':1000*(t3-t1), 'total_ms':1000*(t3-t0)}
        out.append(record)
        times.append(record['total_ms'])
        if (j+1)%32==0:print(f'query {j+1}/{len(rows)}', flush=True)
    after = io_bytes()
    m=json.loads((root/'manifest.json').read_text())
    result={'workload':workload, 'includes_query_encoding':encode_query, 'nprobe':probes,
            'prefetch':prefetch,
            'entries':int(index.ntotal), 'lists':index.nlist, 'load_ms':load_ms,
            'total_latency':stats(times), 'search_latency':stats([r['search_ms'] for r in out]),
            'encoding_latency':stats([r['encoding_ms'] for r in out]),
            'routing_latency':stats([r['routing_ms'] for r in out]),
            'mean_disk_MB_per_query':(after-before)/1e6/len(out),
            'memory_peak_bytes':int((cg_path()/'memory.peak').read_text()),
            'memory_limit':(cg_path()/'memory.max').read_text().strip(),
            'memory_events':(cg_path()/'memory.events').read_text(),
            'postings_bytes':m['postings_bytes'], 'index_metadata_bytes':m['metadata_bytes'],
            'threads':4,'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'storage':'WD SN850X NVMe, uncompressed btrfs files', 'query_timings':out,
            'accuracy':'NOT MEASURED: synthetic capacity simulation with repeated codes',
            'faiss_version':faiss.__version__}
    print(json.dumps(result), flush=True)
    return result


def evict(root):
    for name in ('postings.data', 'index.faiss', 'centroids.npy'):
        with (root/name).open('rb') as f:
            os.fsync(f.fileno())
            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)


def sweep(root=ROOT, prefetch=False):
    import subprocess,sys,uuid
    out=root/'results'
    out.mkdir(exist_ok=True)
    experiments=[(n,'routed',False) for n in (16,64,256)] + [(64,'dispersed',False),(64,'routed',True)]
    if prefetch:
        experiments=[(64,'routed',False),(64,'routed',True)]
    failures=[]
    for n,workload,with_encoding in experiments:
        name=f'np{n}-{workload}-'+('end-to-end' if with_encoding else 'search-only')
        if prefetch:
            name += '-prefetch'
        evict(root)
        cmd=[sys.executable,'-m','bench.disk80m','worker','--root',str(root),'--probes',str(n),'--workload',workload]
        if with_encoding:cmd.append('--encode-query')
        if prefetch:cmd.append('--prefetch')
        unit=f'srl-disk80m-query-{uuid.uuid4().hex[:10]}.scope'
        cmd=['systemd-run','--user','--scope','--quiet',f'--unit={unit}',
             '-p','MemoryMax=4G','-p','MemorySwapMax=0','--','taskset','-c','12-15',*cmd]
        env=os.environ|{'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                        'CUDA_VISIBLE_DEVICES':'','TOKENIZERS_PARALLELISM':'false'}
        print('START',name,flush=True)
        try:
            p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=600)
            lines=[s for s in p.stdout.splitlines() if s.startswith('{')]
            if p.returncode or not lines:
                r={'error':f'exit {p.returncode}','stderr':p.stderr[-3000:],'stdout':p.stdout[-1000:]}
                failures.append(name)
            else:r=json.loads(lines[-1])
        except subprocess.TimeoutExpired:
            subprocess.run(['systemctl','--user','stop',unit],capture_output=True,timeout=30)
            r={'error':'timeout after 10 minutes'};failures.append(name)
        write_json(out/(name+'.json'),r)
        if 'error' in r:print('FAILED',name,r,flush=True)
        else:print(name,r['total_latency'],'disk MB/query',round(r['mean_disk_MB_per_query'],1),'memory GB',round(r['memory_peak_bytes']/1e9,2),flush=True)
        write_json(out/'status.json',{'state':'running','last_finished':name,'failures':failures})
    write_json(out/'status.json',{'state':'partial' if failures else 'complete','failures':failures})
    if failures:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('step',choices=['build','queries','worker','sweep'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--entries',type=int,default=80_000_000)
    p.add_argument('--lists',type=int,default=65536)
    p.add_argument('--probes',type=int,default=64)
    p.add_argument('--workload',choices=['routed','dispersed'],default='routed')
    p.add_argument('--encode-query',action='store_true')
    p.add_argument('--prefetch',action='store_true',help='Use the prefetch stage normally invoked by IndexIVF.search')
    a=p.parse_args()
    if a.step=='build':build(a.root,a.entries,a.lists)
    elif a.step=='queries':prepare_queries(a.root)
    elif a.step=='worker':worker(a.root,a.probes,a.workload,a.encode_query,a.prefetch)
    else:sweep(a.root,a.prefetch)

if __name__=='__main__':main()
