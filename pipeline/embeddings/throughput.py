"""Exclusive-GPU DenseOn throughput test, separate from production embeddings.

Prepare a deterministic, cluster-sampled English-abstract workload before stopping
services. Benchmark BF16 batches, then validate sustained production-format writes.
Always restore precisely the services/containers that were running beforehand.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import embed_content as e

ROOT = Path('/mnt/fast/openalex-embeddings/denseon-throughput')
MASTER = Path('/mnt/fast/openalex-embeddings/denseon-complete')
SYSTEM_UNITS = ['chattering-semantic', 'kokoro-tts', 'parakeet-server']
USER_UNITS = ['qwen3-embed-4b']
CONTAINERS = ['demo-08b']
GPU_UUID = 'GPU-73bb8110-4d4b-f645-dce4-724b0baf767f'
TOTAL = 203486035


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'sample.json').exists():
        print('Reusing pinned throughput sample', flush=True)
        return
    doc=e.read_manifest(MASTER)
    rng=np.random.default_rng(20261005)
    groups=[(fi,g['row_group'],g['source_rows']) for fi,f in enumerate(doc['files']) for g in f['row_groups']]
    weights=np.array([x[2] for x in groups],dtype=np.float64)
    # Draw row groups proportional to raw row count, sample the same number of raw
    # rows within each, THEN filter English abstracts. Do not oversample sparse groups
    # by taking a fixed number of eligible rows from each.
    draws=rng.choice(len(groups),size=40,replace=True,p=weights/weights.sum())
    selected=[];provenance=[];source_rows=eligible_rows=0
    t0=time.perf_counter()
    for draw in draws:
        fi,rg,n=groups[int(draw)]
        entry=doc['files'][fi];path=e.check_source(entry)
        t=pq.ParquetFile(path).read_row_group(rg,columns=['id','title','abstract','language'],use_threads=False)
        mask=e.selection_mask(t,'has_english_abstract')
        eligible=int(pc.sum(pc.cast(mask,pa.int64())).as_py() or 0)
        source_rows+=len(t);eligible_rows+=eligible
        sample_rows=np.sort(rng.choice(len(t),size=min(3000,len(t)),replace=False))
        small=t.take(pa.array(sample_rows)).filter(pc.take(mask,pa.array(sample_rows)))
        selected.extend(small.select(['id','title','abstract']).to_pylist())
        provenance.append({'file_index':fi,'row_group':rg,'source_rows':len(t),'eligible_rows':eligible,
                           'sampled_raw_rows':len(sample_rows),'sampled_eligible_rows':len(small)})
        print(f'SAMPLE {len(provenance)}/40 groups; {len(selected):,} eligible samples',flush=True)
    seen=set();unique=[]
    for r in selected:
        if r['id'] not in seen:
            unique.append(r);seen.add(r['id'])
    rng.shuffle(unique)
    if len(unique)<20000:
        raise RuntimeError(f'Only {len(unique)} samples; increase group draws explicitly')
    table=pa.Table.from_pylist(unique[:20000])
    pq.write_table(table,ROOT/'sample.parquet',row_group_size=4096)
    ids,texts,_=e.records(table)
    # CPU-only tokenization: no GPU service needs to be stopped for preparation.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(doc['embedding']['model_snapshot'],local_files_only=True)
    token_lens=tokenizer([doc['embedding']['document_prompt']+t for t in texts],
                         truncation=True,max_length=513,return_length=True,
                         return_attention_mask=False,return_token_type_ids=False)['length']
    lens=np.array(token_lens)
    # A separate actual-HDD row-group validation, bounded at 32,768 eligible records.
    validation=next(p for p in provenance if p['eligible_rows']>=10000)
    e.atomic_json(ROOT/'sample.json',{'created_utc':e.now(),'seed':20261005,'rows':len(table),
                   'model':doc['embedding'],'input_plan_id':doc['plan_id'],'groups':provenance,
                   'source_rows_read':source_rows,'eligible_rows_in_read_groups':eligible_rows,
                   'preparation_seconds':time.perf_counter()-t0,
                   'mean_capped_tokens':float(np.minimum(lens,512).mean()),
                   'p50_capped_tokens':float(np.median(np.minimum(lens,512))),
                   'truncated_fraction':float(np.mean(lens>512)),
                   'validation_group':validation,
                   'sampling_note':'40 PPS row-group draws; uniform raw-row sample then English/nonempty-abstract filter; not a perfectly independent row sample'})
    print('Sample ready',ROOT/'sample.json',flush=True)


def command(args, **kwargs):
    return subprocess.run(args,check=True,**kwargs)


def active(unit,user=False):
    return subprocess.run(['systemctl',*(['--user'] if user else []),'is-active','--quiet',unit]).returncode==0


def capture(state):
    command(['sudo','-n','true'])
    data={'captured_utc':e.now(),'gpu_uuid':GPU_UUID,
          'system_units':[u for u in SYSTEM_UNITS if active(u)],
          'user_units':[u for u in USER_UNITS if active(u,True)],'containers':[], 'restored':False}
    for name in CONTAINERS:
        p=subprocess.run(['podman','inspect',name],capture_output=True,text=True)
        if p.returncode==0 and json.loads(p.stdout)[0]['State']['Running']:
            data['containers'].append(name)
    e.atomic_json(state,data)
    return data


def pause(state):
    data=capture(state)
    for u in data['system_units']:command(['sudo','-n','systemctl','stop',u])
    for u in data['user_units']:command(['systemctl','--user','stop',u])
    for name in data['containers']:
        stopped=subprocess.run(['podman','stop','--time','30',name],capture_output=True,text=True)
        if stopped.returncode:
            inspect=subprocess.run(['podman','inspect',name],capture_output=True,text=True)
            if inspect.returncode or json.loads(inspect.stdout)[0]['State']['Running']:
                raise RuntimeError(f'Container {name} did not stop: {stopped.stderr}')
            print(f'{name}: stopped; container manager reported a conmon-exit warning',flush=True)
    for _ in range(30):
        gpu=next(g for g in e.gpu_candidates() if g['uuid']==GPU_UUID)
        if gpu['free_mib']>=22000:
            print('GPU 1 ready:',gpu,flush=True);return
        time.sleep(1)
    raise RuntimeError('GPU 1 still occupied after stopping the approved services; no other process will be killed')


def wait_port(port,timeout=150):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=1):return True
        except OSError:time.sleep(2)
    return False


def restore(state):
    if not state.exists():return
    data=json.loads(state.read_text())
    if data.get('restored'):return
    errors=[]
    # Establish fixed vLLM reservations before loading elastic PyTorch speech/search
    # caches. Otherwise Parakeet can retain enough startup workspace to prevent the
    # 0.8B server's fixed memory reservation, despite the steady-state set fitting.
    # Also recovers an earlier partial restore, using only originally active units.
    for u in data['system_units']:
        try:command(['sudo','-n','systemctl','stop',u])
        except Exception as exc:errors.append(f'{u}: {exc}')
    for u in data['user_units']:
        try:
            command(['systemctl','--user','start',u])
            if not wait_port(8100,180):errors.append(f'{u}: port 8100 not ready')
        except Exception as exc:errors.append(f'{u}: {exc}')
    for name in data['containers']:
        try:
            # Podman's conmon can inherit its caller's cgroup. A separate scope must
            # outlive this temporary benchmark service, or cleanup kills the demo.
            scope=f'{name}-runtime-{uuid.uuid4().hex[:8]}'
            command(['systemd-run','--user','--scope','--quiet',f'--unit={scope}',
                     'podman','start',name])
            if not wait_port(8014,180):errors.append(f'{name}: port 8014 not ready')
        except Exception as exc:errors.append(f'{name}: {exc}')
    ports={'chattering-semantic':8090,'kokoro-tts':8880,'parakeet-server':8765}
    for u in data['system_units']:
        try:
            command(['sudo','-n','systemctl','start',u])
            if not wait_port(ports[u],90):errors.append(f'{u}: port {ports[u]} not ready')
        except Exception as exc:errors.append(f'{u}: {exc}')
    data.update(restored=not errors,restoration_checked_utc=e.now(),restore_errors=errors)
    e.atomic_json(state,data)
    print('RESTORE',json.dumps({'ok':not errors,'errors':errors}),flush=True)
    if errors:raise RuntimeError('Some services failed to restore; see '+str(state))


class Monitor:
    def __init__(self):self.stop=threading.Event();self.samples=[]
    def __enter__(self):
        def collect():
            while not self.stop.is_set():
                p=subprocess.run(['nvidia-smi','-i',GPU_UUID,'--query-gpu=utilization.gpu,memory.used,power.draw',
                                  '--format=csv,noheader,nounits'],capture_output=True,text=True)
                if p.returncode==0:
                    try:self.samples.append([float(x.strip()) for x in p.stdout.strip().split(',')])
                    except ValueError:pass
                self.stop.wait(.5)
        self.thread=threading.Thread(target=collect,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=3)
    def result(self):
        if not self.samples:return {}
        a=np.array(self.samples)
        return {'gpu_utilization_mean_percent':float(a[:,0].mean()),'gpu_utilization_max_percent':float(a[:,0].max()),
                'device_memory_peak_mib':float(a[:,1].max()),'power_mean_watts':float(a[:,2].mean()),
                'gpu_samples':len(a)}


def vectors_similarity(a,b):
    a=a.astype('float32');b=b.astype('float32')
    sims=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    return float(sims.min()),float(sims.mean())


def measure(runroot):
    import torch
    config=json.loads((ROOT/'sample.json').read_text())['model']
    encoder=e.DenseEncoder(config,'cuda:0',64,cpu_threads=8,allocation_gib=20)
    metadata={'created_utc':e.now(),'model':config,'gpu_uuid':GPU_UUID,'physical_gpu':1,
              'torch':torch.__version__,'attention_implementation':getattr(encoder.model[0].auto_model.config,'_attn_implementation',None),
              'population_rows':TOTAL,'trials':[]}
    target=runroot/'results.json'
    runroot.mkdir(parents=True,exist_ok=True)
    baseline=None
    # Repeat 64 at the end to expose warm-up/thermal/order effects.
    for trial,batch in enumerate((64,128,256,512,64)):
        encoder.batch_size=batch
        warm=pq.read_table(ROOT/'sample.parquet').slice(0,min(batch,512))
        _,warm_text,_=e.records(warm)
        encoder.encode(warm_text);torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        with Monitor() as monitor:
            sample=pq.read_table(ROOT/'sample.parquet').slice(0,4096)
            ids,texts,hashes=e.records(sample)
            read_s=time.perf_counter()-started
            t=time.perf_counter();vectors=encoder.encode(texts);torch.cuda.synchronize();encode_s=time.perf_counter()-t
            t=time.perf_counter()
            task={'key':f'trial-{trial:02d}-batch-{batch}','rows':len(texts)}
            doc={'plan_id':'throughput-'+runroot.name,'embedding':config}
            e.commit_shard(runroot,task,doc,ids,hashes,vectors,{'benchmark_only':True})
            write_s=time.perf_counter()-t
            seconds=time.perf_counter()-started
        if baseline is None:baseline=vectors.copy()
        cosine_min,cosine_mean=vectors_similarity(baseline,vectors)
        if cosine_min<0.995:raise ValueError('Batch-size change produced unexpectedly different embeddings')
        row={'trial':trial,'requested_batch':batch,'actual_batch':encoder.batch_size,'rows':len(texts),
             'read_prepare_seconds':read_s,'encode_seconds':encode_s,'write_commit_seconds':write_s,
             'seconds':seconds,'rows_per_second':len(texts)/seconds,'encode_rows_per_second':len(texts)/encode_s,
             'tensor_memory_peak_bytes':torch.cuda.max_memory_allocated(),
             'cosine_min_vs_batch64':cosine_min,'cosine_mean_vs_batch64':cosine_mean,**monitor.result()}
        metadata['trials'].append(row);e.atomic_json(target,metadata)
        print('TRIAL',json.dumps(row),flush=True)
    grouped={}
    for row in metadata['trials']:grouped.setdefault(row['actual_batch'],[]).append(row['rows_per_second'])
    best=max(grouped,key=lambda b:statistics.median(grouped[b]))
    encoder.batch_size=best
    # Sustained validation at production's 10,000-record shard size, not one huge
    # globally sorted call; include normalized vectors, IDs, hashes and fsync commits.
    started=time.perf_counter()
    with Monitor() as monitor:
        table=pq.read_table(ROOT/'sample.parquet')
        for start in range(0,len(table),10000):
            ids,texts,hashes=e.records(table.slice(start,10000))
            vectors=encoder.encode(texts)
            e.commit_shard(runroot,{'key':f'validation-{start:06d}','rows':len(texts)},
                           {'plan_id':'throughput-'+runroot.name,'embedding':config},ids,hashes,vectors,{'benchmark_only':True})
        torch.cuda.synchronize();seconds=time.perf_counter()-started
    validation={'requested_batch':best,'actual_batch':encoder.batch_size,'rows':len(table),'seconds':seconds,
                'rows_per_second':len(table)/seconds,'estimated_population_days':TOTAL/(len(table)/seconds)/86400,
                'input':'representative staged sample on NVMe; 10k-record production-sized commits',**monitor.result()}
    metadata['validation']=validation;e.atomic_json(target,metadata);print('VALIDATION',json.dumps(validation),flush=True)
    # A separate source-to-output check actually reads an original HDD row group.
    group=json.loads((ROOT/'sample.json').read_text())['validation_group']
    master=e.read_manifest(MASTER);entry=master['files'][group['file_index']]
    path=e.check_source(entry)
    started=time.perf_counter()
    with Monitor() as monitor:
        table=pq.ParquetFile(path).read_row_group(group['row_group'],columns=['id','title','abstract','language'],use_threads=False)
        table=table.filter(e.selection_mask(table,'has_english_abstract')).select(['id','title','abstract']).slice(0,32768)
        read_s=time.perf_counter()-started
        for start in range(0,len(table),10000):
            ids,texts,hashes=e.records(table.slice(start,10000));vectors=encoder.encode(texts)
            e.commit_shard(runroot,{'key':f'hdd-validation-{start:06d}','rows':len(texts)},
                           {'plan_id':'throughput-'+runroot.name,'embedding':config},ids,hashes,vectors,{'benchmark_only':True})
        torch.cuda.synchronize();seconds=time.perf_counter()-started
    metadata['hdd_validation']={'rows':len(table),'seconds':seconds,'source_read_filter_seconds':read_s,
          'rows_per_second':len(table)/seconds,'source':str(path),'row_group':group['row_group'],
          'note':'One real source group; may benefit from OS cache after sample preparation; not representative alone',**monitor.result()}
    metadata['state']='complete';e.atomic_json(target,metadata)
    print('HDD VALIDATION',json.dumps(metadata['hdd_validation']),flush=True)


def exclusive(runroot):
    state=runroot/'services.json';runroot.mkdir(parents=True,exist_ok=True)
    def interrupted(*_):raise KeyboardInterrupt('Exclusive test interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    process=None
    try:
        pause(state)
        env=os.environ|{'CUDA_VISIBLE_DEVICES':GPU_UUID,'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
                        'TRITON_LIBCUDA_PATH':'/run/opengl-driver/lib','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
                        'TOKENIZERS_PARALLELISM':'false'}
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'measure','--run',str(runroot)],
                                  env=env,start_new_session=True)
        try:code=process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Measurement exceeded its 15-minute time budget')
        if code:raise RuntimeError(f'Benchmark exited with status {code}')
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
        restore(state)


def main():
    p=argparse.ArgumentParser();p.add_argument('step',choices=['prepare','exclusive','measure','restore'])
    p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.step=='prepare':prepare()
    else:
        if a.run is None:p.error('--run is required')
        if a.step=='exclusive':exclusive(a.run)
        elif a.step=='measure':measure(a.run)
        else:restore(a.run/'services.json')

if __name__=='__main__':main()
