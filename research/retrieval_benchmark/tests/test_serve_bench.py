import json

from bench import serve_bench as s


def test_unsupported_one_bit_is_recorded_without_running(monkeypatch, tmp_path):
    monkeypatch.setattr(s, 'OUT', tmp_path)
    monkeypatch.setattr(s, 'run_worker', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('must not run')))
    s.orchestrate(['lateon-pool4-1bit-np32'], ['ram', 'disk'])
    for regime in ('ram', 'disk'):
        data = json.loads((tmp_path / regime / 'lateon-pool4-1bit-np32.json').read_text())
        assert 'unsupported' in data


def test_disk_failure_does_not_abort_later_configs(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(s, 'OUT', tmp_path)
    monkeypatch.setattr(s, 'configs', lambda: {'a': {'engine': 'faiss'}, 'b': {'engine': 'faiss'}})
    monkeypatch.setattr(s, 'index_files', lambda _: ['file'])
    monkeypatch.setattr(s, 'search_bytes', lambda _: 10)
    monkeypatch.setattr(s, 'evict', lambda _: None)
    called = []
    def fail(name, regime, limit):
        called.append(name)
        return {'name': name, 'regime': regime, 'error': 'exit -9', 'stderr_tail': 'loading index'}
    monkeypatch.setattr(s, 'run_worker', fail)
    with pytest.raises(SystemExit):
        s.orchestrate(['a', 'b'], ['disk'])
    assert called == ['a', 'b']
    assert json.loads((tmp_path / 'status.json').read_text())['state'] == 'partial'


def test_latency_units():
    r = s.latency_stats([0.001, 0.003, 0.002])
    assert r['p50_ms'] == 2
    assert r['p95_ms'] == 3
