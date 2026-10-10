"""Boot-resumable embedding supervisor. No GPU services are stopped.

The persistent systemd unit retries temporary failures. Exit 78 marks input/model
integrity errors that need a person, rather than retrying them forever.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time

import embed_content as e

OUTPUT=Path(os.environ.get('OPENALEX_EMBEDDING_OUTPUT','/mnt/fast/openalex-embeddings/denseon-english-abstracts'))
SOURCE=Path('/mnt/bulk/openalex/embedding-inputs/denseon-complete')
SNAPSHOT=Path('/mnt/bulk/openalex/embedding-inputs/denseon-english-abstracts')
REVISION='cb9947ebccb33862d24e3c7ca2edb25e51acd887'
EXPECTED=203486035


def update(state, **extra):
    if not Path('/mnt/fast').is_mount():
        return
    old={}
    if (OUTPUT/'status.json').exists():
        old=json.loads((OUTPUT/'status.json').read_text())
    e.atomic_json(OUTPUT/'status.json',{**old,'state':state,'updated_utc':e.now(),**extra})


def main():
    signal.signal(signal.SIGTERM,e.handle_stop)
    signal.signal(signal.SIGINT,e.handle_stop)
    # Never write to the underlying root disk when the data mount is absent.
    while not all(Path(p).is_mount() for p in ('/mnt/fast','/mnt/bulk')):
        print('Waiting for data mounts',flush=True)
        if e.STOP:return
        time.sleep(5)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'PAUSED').exists():
        update('paused',pause_persists_across_reboot=True)
        return
    # Give the existing model/voice services first chance to establish their GPU
    # reservations during boot. Free memory is still checked before model loading.
    while float(Path('/proc/uptime').read_text().split()[0]) < 180:
        update('waiting_for_boot_services',message='Waiting for existing GPU services to settle after boot')
        if e.STOP:return
        time.sleep(5)
    while not e.STOP:
        try:
            if not (OUTPUT/'manifest.json').exists():
                e.plan(SOURCE,SNAPSHOT,OUTPUT,REVISION,10000,'has_english_abstract','publication_date_desc')
            if e.STOP:return
            doc=e.read_manifest(OUTPUT)
            if (doc.get('selection_policy')!='has_english_abstract' or
                    doc.get('ordering')!='publication_date_desc' or
                    doc.get('date_normalization')!='arrow_date_v1' or doc['expected_rows']!=EXPECTED):
                raise ValueError('Manifest does not match the approved English-abstract, newest-first population')
            e.atomic_json(OUTPUT/'plan-summary.json',e.public_plan_summary(doc))
            # Keep GPU 0 available for the other model workloads, including after boot.
            e.run(OUTPUT,device='cuda:1',min_free_gib=4,batch_size=64,cpu_threads=8)
            return
        except RuntimeError as exc:
            if 'Another process holds' not in str(exc):
                raise
            # An already-running planner or worker owns the lock. Leave its progress
            # untouched and retry; never create a second writer.
            print('Existing planner/worker holds the lock; waiting',flush=True)
            for _ in range(10):
                if e.STOP:return
                time.sleep(1)


if __name__=='__main__':
    try:
        main()
    except (ValueError,FloatingPointError,AssertionError) as exc:
        update('failed',error=str(exc),requires_attention=True)
        print(f'Permanent validation failure: {exc}',file=sys.stderr,flush=True)
        sys.exit(78)
    except Exception as exc:
        if 'Pinned source changed' in str(exc) or 'Source changed' in str(exc):
            update('failed',error=str(exc),requires_attention=True)
            sys.exit(78)
        update('failed',error=f'{type(exc).__name__}: {exc}',requires_attention=False)
        print(f'Temporary failure, systemd will retry: {exc}',file=sys.stderr,flush=True)
        sys.exit(75)
