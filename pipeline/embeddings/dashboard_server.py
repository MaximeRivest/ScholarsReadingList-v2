"""Read-only, aggregate progress API and dashboard. Bind localhost; Tailscale HTTPS
provides private remote access. No arbitrary file access, logs, secrets or write API.
Uses only the Python standard library so it is available before the model venv loads.
"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit

OUTPUT=Path(os.environ.get('OPENALEX_EMBEDDING_OUTPUT','/mnt/fast/openalex-embeddings/denseon-english-abstracts'))
INVENTORY=Path('/mnt/fast/openalex-embeddings/abstract-inventory.json')
UNIT='openalex-embeddings.service'
WEB=Path(__file__).parent/'dashboard'
_manifest_cache={}
_lock=threading.Lock()


def read_json(path):
    try:
        if path.stat().st_size>32*2**20:
            return {}
        return json.loads(path.read_text())
    except (OSError,ValueError):
        return {}


def service_state():
    try:
        p=subprocess.run(['systemctl','--user','show',UNIT,'--property=ActiveState,SubState,MainPID,NRestarts,UnitFileState,ExecMainStatus'],
                         capture_output=True,text=True,timeout=2)
        return dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
    except (OSError,subprocess.TimeoutExpired):
        return {}


def manifest_summary():
    small=OUTPUT/'plan-summary.json'
    if small.exists():
        return read_json(small)
    path=OUTPUT/'manifest.json'
    try:mtime=path.stat().st_mtime_ns
    except OSError:return {}
    with _lock:
        if _manifest_cache.get('mtime')!=mtime:
            doc=read_json(path)
            _manifest_cache.update(mtime=mtime,value={
                'plan_id':doc.get('plan_id'),'expected_rows':doc.get('expected_rows'),
                'ordering':doc.get('ordering'),'selection':doc.get('selection'),
                'model':doc.get('embedding',{}).get('model'),
                'revision':doc.get('embedding',{}).get('revision'),
                'max_tokens':doc.get('embedding',{}).get('max_seq_length'),
                'dimensions':doc.get('embedding',{}).get('dimensions'),
                'source_rows':sum(g['source_rows'] for f in doc.get('files',[]) for g in f['row_groups'])})
        return dict(_manifest_cache['value'])


def payload():
    mounted=all(Path(p).is_mount() for p in ('/mnt/fast','/mnt/bulk'))
    status=read_json(OUTPUT/'status.json') if mounted else {}
    plan=manifest_summary() if mounted else {}
    inventory=read_json(INVENTORY) if mounted else {}
    service=service_state()
    expected=plan.get('expected_rows') or inventory.get('languages',{}).get('en')
    current_state=status.get('state','not_started')
    paused=mounted and (OUTPUT/'PAUSED').exists()
    if not mounted:current_state='waiting_for_disks'
    elif paused:current_state='paused'
    elif service.get('SubState')=='auto-restart':current_state='retrying'
    elif service.get('ActiveState')=='failed':current_state='failed'
    elif service.get('ActiveState') in ('inactive','deactivating') and current_state not in ('complete','planned','paused'):
        # Do not keep showing "encoding" from a stale file after a killed worker.
        current_state='stopped'
    updated=status.get('updated_utc')
    age=None
    if updated:
        try:age=max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(updated)).total_seconds())
        except ValueError:pass
    disk={}
    if mounted:
        usage=shutil.disk_usage('/mnt/fast')
        disk={'free_bytes':usage.free,'total_bytes':usage.total}
    public_keys=('completed_rows','completed_shards','verified_rows','percent','current_publication_date',
                 'years','recent_progress','rows_per_second_this_run','estimated_remaining_hours','active_seconds',
                 'token_sample_rows','token_sample_truncated','gpu','minimum_free_gib','eligible_rows_counted',
                 'source_rows_scanned','files_completed','files_total','current_row_group','requires_attention','batch_size')
    progress={k:status[k] for k in public_keys if k in status}
    gpu=progress.get('gpu')
    if gpu:
        progress['gpu']={k:gpu[k] for k in ('index','free_mib','total_mib') if k in gpu}
    # Errors are displayed as text, never HTML. No journal/config/environment export.
    progress['error']=str(status.get('error',''))[:500]
    return {'server_utc':datetime.now(timezone.utc).isoformat(),'state':current_state,
            'worker_state':status.get('state'),'updated_utc':updated,'update_age_seconds':age,
            'expected_rows':expected,'total_source_rows':plan.get('source_rows') or inventory.get('rows_scanned'),
            'progress':progress,'plan':plan,'service':service,'disk':disk,'pause_persists':paused,
            'restart_policy':'enabled at boot; temporary failures retry after 60 seconds; checkpoints verified on resume'}


class Handler(BaseHTTPRequestHandler):
    def response_headers(self,status,content_type,length):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(length))
        self.send_header('Cache-Control','no-store')
        # Only nonsensitive aggregate progress is exposed, read-only, on the tailnet.
        # Artifact iframes can have an opaque/null origin; no credentials are needed.
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/api/status':
            body=json.dumps(payload(),allow_nan=False).encode()
            self.response_headers(200,'application/json; charset=utf-8',len(body));self.wfile.write(body)
        elif path in ('/','/index.html'):
            body=(WEB/'index.html').read_bytes()
            self.response_headers(200,'text/html; charset=utf-8',len(body));self.wfile.write(body)
        else:
            body=b'Not found';self.response_headers(404,'text/plain',len(body));self.wfile.write(body)
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Allow-Methods','GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Content-Type')
        self.send_header('Content-Length','0');self.end_headers()
    def log_message(self,*_):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8797)
    a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    server.daemon_threads=True
    print(f'Read-only embedding progress: http://127.0.0.1:{a.port}',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
