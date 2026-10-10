import json
from types import SimpleNamespace

import numpy as np

import throughput as t


def test_similarity_reports_float16_batch_drift():
    a=np.array([[.6,.8],[1.,0.]],dtype='float16')
    low,mean=t.vectors_similarity(a,a)
    assert low>.9999 and mean>.9999


def test_restore_only_previously_running_services(tmp_path,monkeypatch):
    state=tmp_path/'services.json'
    state.write_text(json.dumps({'system_units':['kokoro-tts'],'user_units':[],
                                 'containers':['demo-08b'],'restored':False}))
    calls=[]
    monkeypatch.setattr(t,'command',lambda args,**kw:calls.append(args))
    monkeypatch.setattr(t,'wait_port',lambda *a,**kw:True)
    t.restore(state)
    assert calls[0]==['sudo','-n','systemctl','stop','kokoro-tts']
    assert calls[1][:3]==['systemd-run','--user','--scope']
    assert calls[1][-3:]==['podman','start','demo-08b']
    assert calls[2]==['sudo','-n','systemctl','start','kokoro-tts']
    assert json.loads(state.read_text())['restored']
    t.restore(state)
    assert len(calls)==3


def test_exclusive_failure_still_restores(tmp_path,monkeypatch):
    import pytest
    restored=[]
    def pause(_):raise RuntimeError('stop failed halfway')
    monkeypatch.setattr(t,'pause',pause)
    monkeypatch.setattr(t,'restore',lambda state:restored.append(state))
    monkeypatch.setattr(t.signal,'signal',lambda *a:None)
    with pytest.raises(RuntimeError,match='stop failed'):
        t.exclusive(tmp_path)
    assert restored==[tmp_path/'services.json']
