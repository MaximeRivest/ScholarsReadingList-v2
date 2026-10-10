from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import embed_content as e


@pytest.fixture
def planned(tmp_path, monkeypatch):
    import huggingface_hub as hf
    monkeypatch.setattr(hf, 'HfApi', lambda: SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha='a'*40)))
    monkeypatch.setattr(hf, 'snapshot_download', lambda *a, **kw: '/fake/pinned/model')
    source=tmp_path/'source'; snapshot=tmp_path/'snapshot'; out=tmp_path/'out'
    f=source/'year=2024/data.parquet'; f.parent.mkdir(parents=True)
    table=pa.table({'id':[f'https://openalex.org/W{i}' for i in range(1,6)],
                    'title':['First','Ignore','Unknown',None,'Last'],
                    'abstract':['A real abstract.']*5,
                    'is_complete':[True,False,None,True,True]})
    pq.write_table(table,f,row_group_size=3)
    doc=e.plan(source,snapshot,out,shard_rows=2)
    return source,snapshot,out,doc


class FakeEncoder:
    calls=0
    def __init__(self, config, device, batch_size, *args):
        self.dim=config['dimensions']; self.batch_size=batch_size
    def encode(self, texts):
        type(self).calls+=1
        return e.normalized_f16(np.ones((len(texts),self.dim)),len(texts),self.dim)
    def truncation_sample(self, texts, ids):
        return {'token_sample_rows':0,'token_sample_truncated':0}


def test_flag_selection_resume_and_exact_id_mapping(planned):
    source,snapshot,out,doc=planned
    assert doc['expected_rows']==3
    FakeEncoder.calls=0
    e.run(out,device='cpu',max_shards=1,encoder_factory=FakeEncoder)
    first=json.loads((out/'status.json').read_text())
    assert first['state']=='paused' and first['completed_rows']==1
    assert FakeEncoder.calls==1
    e.run(out,device='cpu',encoder_factory=FakeEncoder)
    assert FakeEncoder.calls==2
    assert e.verify(out)['complete']
    tasks=list(e.shard_tasks(doc))
    ids=np.concatenate([np.load(out/'shards'/t['key']/'work_ids.npy') for t in tasks])
    assert ids.tolist()==[1,4,5]
    e.run(out,device='cpu',encoder_factory=FakeEncoder)
    assert FakeEncoder.calls==2, 'Completed outputs must not be re-encoded'


def test_hardlink_snapshot_survives_atomic_source_replacement(planned):
    source,snapshot,out,doc=planned
    src=source/'year=2024/data.parquet'
    original=e.check_source(doc['files'][0])
    tmp=src.with_suffix('.new');tmp.write_bytes(b'new publication');os.replace(tmp,src)
    assert original.read_bytes()[:4]==b'PAR1'
    assert e.check_source(doc['files'][0])==original
    e.run(out,device='cpu',encoder_factory=FakeEncoder)
    assert e.verify(out)['completed_rows']==3


def test_modified_snapshot_refused(planned):
    _,snapshot,out,doc=planned
    f=snapshot/'year=2024/data.parquet'
    with f.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(RuntimeError,match='Pinned source changed'):
        e.run(out,device='cpu',encoder_factory=FakeEncoder)
    assert json.loads((out/'status.json').read_text())['state']=='failed'


def test_corrupted_committed_vector_refused(planned):
    _,_,out,doc=planned
    e.run(out,device='cpu',encoder_factory=FakeEncoder)
    t=next(e.shard_tasks(doc)); p=out/'shards'/t['key']/'vectors.npy'
    with p.open('r+b') as f:
        f.seek(-2,2);f.write(b'\x00\x00')
    with pytest.raises(ValueError,match='checksum'):
        e.run(out,device='cpu',encoder_factory=FakeEncoder)


def test_lock_excludes_second_writer(tmp_path):
    with e.locked(tmp_path):
        with pytest.raises(RuntimeError,match='holds'):
            with e.locked(tmp_path):pass


def test_manifest_tampering_refused(planned):
    _,_,out,doc=planned
    doc['embedding']['max_seq_length']=8192
    (out/'manifest.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='checksum'):
        e.read_manifest(out)


def test_nonfinite_and_zero_vectors_refused():
    with pytest.raises(FloatingPointError):e.normalized_f16([[float('nan'),1]],1,2)
    with pytest.raises(FloatingPointError):e.normalized_f16([[0,0]],1,2)
    with pytest.raises(FloatingPointError):e.normalized_f16([[1,2,3]],1,2)
    v=e.normalized_f16([[3,4]],1,2)
    np.testing.assert_allclose(v,[[.6,.8]],atol=.001)


def test_input_format_ids_and_hash():
    table=pa.table({'id':['https://openalex.org/W7'],'title':[None],'abstract':['Some text.']})
    ids,texts,hashes=e.records(table)
    assert ids.tolist()==[7] and texts==['Some text.']
    assert hashes[0].tobytes()==hashlib.blake2b(b'Some text.',digest_size=16).digest()


def test_abstract_selection_keeps_all_languages_without_references():
    table=pa.table({'abstract':['English text','Texte français','未知语言',None,'',' \t\n','\u2003'],
                    'language':['en','fr',None,'en','en','en','en'],
                    'references':[None]*7, 'is_complete':[False]*7})
    assert e.selection_mask(table,'has_abstract').to_pylist()==[True,True,True,False,False,False,False]
    assert e.selection_mask(table,'complete').to_pylist()==[False]*7
    assert e.selection_mask(table,'has_english_abstract').to_pylist()==[True,False,False,False,False,False,False]


def test_broader_plan_uses_same_selection_during_encoding(planned,tmp_path):
    source,_,_,_=planned
    out=tmp_path/'all-output'
    doc=e.plan(source,tmp_path/'all-snapshot',out,shard_rows=2,selection='has_abstract')
    assert doc['expected_rows']==5 and doc['selection_policy']=='has_abstract'
    e.run(out,device='cpu',encoder_factory=FakeEncoder)
    assert e.verify(out)['completed_rows']==5


def test_english_abstract_selection_does_not_require_complete_flag():
    table=pa.table({'abstract':['An abstract.','Another abstract.'], 'language':['en','en'],
                    'is_complete':[False,False]})
    assert e.selection_mask(table,'has_english_abstract').to_pylist()==[True,True]


def test_unknown_selection_rejected():
    with pytest.raises(ValueError,match='Unknown selection'):
        e.selection_columns('invented')


def test_invalid_complete_row_not_silently_skipped():
    for row in [{'id':'https://openalex.org/W1','title':'Title','abstract':''},
                {'id':'bad','title':'Title','abstract':'Content'}]:
        with pytest.raises(ValueError):e.records(pa.Table.from_pylist([row]))


def test_global_duplicate_id_audit(planned):
    _,_,out,doc=planned
    e.run(out,device='cpu',encoder_factory=FakeEncoder)
    second=list(e.shard_tasks(doc))[1]
    np.save(out/'shards'/second['key']/'work_ids.npy',np.array([1,5],dtype=np.uint64))
    with pytest.raises(ValueError,match='repeated work IDs'):
        e.audit_unique_ids(out,doc)


def test_crash_after_commit_resumes_in_fresh_process(planned):
    import subprocess,sys
    _,_,out,doc=planned
    code='''
import os,sys
from pathlib import Path
import embed_content as e
from test_pipeline import FakeEncoder
if sys.argv[2] == 'crash':
    original=e.commit_shard
    def crash(*args,**kwargs):
        original(*args,**kwargs)
        os._exit(99)
    e.commit_shard=crash
e.run(Path(sys.argv[1]),device='cpu',encoder_factory=FakeEncoder)
'''
    project=Path(e.__file__).parent
    env=os.environ | {'PYTHONPATH':str(project)+os.pathsep+str(project/'tests')}
    p=subprocess.run([sys.executable,'-c',code,str(out),'crash'],env=env,capture_output=True,text=True)
    assert p.returncode==99
    committed=list((out/'shards').glob('**/commit.json'))
    assert len(committed)==1
    vector=committed[0].parent/'vectors.npy'
    before=(vector.stat().st_mtime_ns,e.sha_file(vector))
    subprocess.run([sys.executable,'-c',code,str(out),'resume'],env=env,check=True,capture_output=True,text=True)
    assert before==(vector.stat().st_mtime_ns,e.sha_file(vector))
    assert json.loads((out/'status.json').read_text())['completed_rows']==3
    assert e.verify(out)['complete']


def test_failed_encoding_has_no_committed_shard(planned):
    _,_,out,doc=planned
    class Failing(FakeEncoder):
        def encode(self, texts):raise FloatingPointError('bad vectors')
    with pytest.raises(FloatingPointError):e.run(out,device='cpu',encoder_factory=Failing)
    assert not list((out/'shards').glob('**/commit.json'))
    e.run(out,device='cpu',encoder_factory=FakeEncoder)
    assert e.verify(out)['complete']
