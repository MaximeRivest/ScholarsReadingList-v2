import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datetime import date

import embed_content as e
from test_pipeline import FakeEncoder


@pytest.mark.parametrize('date_storage',['string','date32'])
def test_global_publication_order_and_resume(tmp_path,monkeypatch,date_storage):
    import huggingface_hub as hf
    monkeypatch.setattr(hf,'HfApi',lambda:SimpleNamespace(model_info=lambda *a,**k:SimpleNamespace(sha='a'*40)))
    monkeypatch.setattr(hf,'snapshot_download',lambda *a,**k:'/fake/model')
    src=tmp_path/'source';out=tmp_path/'out'
    # Deliberately inconsistent year partition and unsorted dates inside row groups.
    for year,ids,dates in [(2025,[10,11,12,13],['2025-12-31','2025-01-01',None,'not-a-date']),
                           (2024,[14,15],['2025-06-01','2024-05-03'])]:
        f=src/f'year={year}/data.parquet';f.parent.mkdir(parents=True)
        values=pa.array(dates,type=pa.string()) if date_storage=='string' else pa.array(
            [date.fromisoformat(s) if e.canonical_date(s) else None for s in dates],type=pa.date32())
        pq.write_table(pa.table({'id':[f'W{i}' for i in ids], 'title':['Title']*len(ids),
                                'abstract':['Actual text']*len(ids),'language':['en']*len(ids),
                                'date':values}),f,row_group_size=2)
    doc=e.plan(src,tmp_path/'snapshot',out,shard_rows=2,selection='has_english_abstract',order='publication_date_desc')
    tasks=list(e.shard_tasks(doc))
    assert [t['publication_date'] for t in tasks]==['2025-12-31','2025-06-01','2025-01-01','2024-05-03',None]
    FakeEncoder.calls=0
    e.run(out,device='cpu',max_shards=2,encoder_factory=FakeEncoder)
    assert FakeEncoder.calls==2
    e.run(out,device='cpu',encoder_factory=FakeEncoder)
    ids=np.concatenate([np.load(out/'shards'/t['key']/'work_ids.npy') for t in tasks]).tolist()
    assert ids[:4]==[10,14,11,15] and set(ids[4:])=={12,13}
    assert FakeEncoder.calls==5
    status=json.loads((out/'status.json').read_text())
    assert status['state']=='complete' and status['completed_rows']==6
    assert {y['year']:y['completed'] for y in status['years']}=={'2025':3,'2024':1,'undated':2}
    assert e.verify(out)['complete']


def test_planning_resumes_from_completed_file_cache(tmp_path,monkeypatch):
    import huggingface_hub as hf
    monkeypatch.setattr(hf,'HfApi',lambda:SimpleNamespace(model_info=lambda *a,**k:SimpleNamespace(sha='a'*40)))
    monkeypatch.setattr(hf,'snapshot_download',lambda *a,**k:'/fake/model')
    src=tmp_path/'source';out=tmp_path/'out'
    for year,wid in [(2025,1),(2024,2)]:
        f=src/f'year={year}/data.parquet';f.parent.mkdir(parents=True)
        pq.write_table(pa.table({'id':[f'W{wid}'],'title':['Title'],'abstract':['Abstract'],
                                'language':['en'],'date':[f'{year}-04-03']}),f)
    original=e.footer_digest
    def stop_after_first_file(path):
        value=original(path);e.STOP=True;return value
    monkeypatch.setattr(e,'STOP',False)
    monkeypatch.setattr(e,'footer_digest',stop_after_first_file)
    assert e.plan(src,tmp_path/'snapshot',out,selection='has_english_abstract',order='publication_date_desc') is None
    assert (out/'planning/year=2025.json').exists()
    monkeypatch.setattr(e,'STOP',False)
    monkeypatch.setattr(e,'footer_digest',original)
    counted=[]
    mask=e.selection_mask
    def count_calls(table,policy):
        if policy=='has_english_abstract':counted.append(1)
        return mask(table,policy)
    monkeypatch.setattr(e,'selection_mask',count_calls)
    doc=e.plan(src,tmp_path/'snapshot',out,selection='has_english_abstract',order='publication_date_desc')
    assert doc['expected_rows']==2
    assert len(counted)==1, 'Completed file was needlessly rescanned'


def test_native_parquet_date32_is_not_treated_as_undated():
    from datetime import date,datetime
    values=pa.array([date(2026,9,1),None,date(2025,1,1)],type=pa.date32())
    assert e.date_strings(values).to_pylist()==['2026-09-01',None,'2025-01-01']
    assert e.canonical_date(date(2026,9,1))=='2026-09-01'
    assert e.canonical_date(datetime(2026,9,1,12,30))=='2026-09-01'


def test_invalid_dates_go_last():
    assert e.canonical_date('2024-02-29')=='2024-02-29'
    assert e.canonical_date('2023-02-29') is None
    assert e.canonical_date('2024') is None
    assert e.canonical_date('') is None
    assert e.canonical_date(None) is None
