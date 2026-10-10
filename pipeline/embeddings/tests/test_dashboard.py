import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import dashboard_server as d


def test_stopped_worker_is_not_shown_as_encoding(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'OUTPUT',tmp_path)
    monkeypatch.setattr(d,'manifest_summary',lambda:{'expected_rows':1000})
    monkeypatch.setattr(d,'service_state',lambda:{'ActiveState':'inactive','UnitFileState':'enabled'})
    (tmp_path/'status.json').write_text(json.dumps({'state':'encoding','completed_rows':123,'updated_utc':'2026-01-01T00:00:00+00:00'}))
    result=d.payload()
    assert result['state']=='stopped' and result['progress']['completed_rows']==123
    (tmp_path/'PAUSED').write_text('{}')
    assert d.payload()['state']=='paused'


def test_read_only_http_and_no_arbitrary_file_serving(monkeypatch):
    monkeypatch.setattr(d,'payload',lambda:{'state':'running','expected_rows':1000})
    server=ThreadingHTTPServer(('127.0.0.1',0),d.Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/status') as r:
            assert json.load(r)['expected_rows']==1000
            assert r.headers['Access-Control-Allow-Origin']=='*'
            assert r.headers['Cache-Control']=='no-store'
        for request,code in [(Request(base+'/../manifest.json'),404),
                             (Request(base+'/api/status',data=b'{}',method='POST'),501)]:
            try:urlopen(request)
            except HTTPError as exc:assert exc.code==code
            else:raise AssertionError('request unexpectedly allowed')
    finally:
        server.shutdown();server.server_close();thread.join()
