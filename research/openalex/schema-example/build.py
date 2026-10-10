"""Build a small, auditable example; never scans or modifies the production corpus.

Run from the repository root: uv run python research/openalex/schema-example/build.py
Source: the two real OpenAlex records saved during the earlier schema review.
"""
from pathlib import Path
import csv
import json
import polars as pl

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'schema-review-examples.parquet'
WORKS = sorted(pl.read_parquet(SOURCE).to_dicts(), key=lambda w: w['publication_year'])
EXPECTED = {'https://openalex.org/W2083027281', 'https://openalex.org/W2609272741'}
assert {w['id'] for w in WORKS} == EXPECTED
SCOPE = 'two-paper-fixture; source citation counts, not induced-graph counts'


def source_id(url):
    return url.replace('https://openalex.org/', 'openalex:', 1) if url else None


def scalar_check(rows):
    for row in rows:
        assert all(v is None or isinstance(v, (str, int, float, bool)) for v in row.values()), row


content = []
signatures = []
sc_by_source = {}
cc_by_source = {}
sm = []
cm = []
references = []


def cluster(registry, prefix, key, kind, name, scheme):
    if key not in registry:
        registry[key] = dict(cluster_id=f'{prefix}{len(registry)+1}', kind=kind,
                             name=name, source_id=key, scheme=scheme)
    return registry[key]['cluster_id']


def membership(rows, member, cid, relation, evidence, method, score=None, primary=None):
    rows.append(dict(member_id=member, cluster_id=cid, relation=relation,
                     evidence=evidence, method=method, score=score, is_primary=primary))


for w in WORKS:
    wid = f'P{len(content)+1}'
    content.append(dict(content_id=wid, source_id=source_id(w['id']), title=w['title'],
                        type=w['type'], year=w['publication_year'], language=w['language'],
                        citations=w['cited_by_count'], references_count=w['referenced_works_count']))
    for i, a in enumerate(w['authorships']):
        sid = f'S{len(signatures)+1}'
        affs = a.get('affiliations') or []
        # Exact strings in original order. LF is a display/export separator, not
        # a promised lossless serialization of arbitrary lists. The raw fixture
        # and per-field source paths preserve the original boundaries.
        signatures.append(dict(signature_id=sid, content_id=wid, kind='author_entry',
                               raw_text=a.get('raw_author_name'),
                               affiliation_text='\n'.join(f['raw_affiliation_string'] for f in affs),
                               order=i+1, position=a['author_position'],
                               is_corresponding=a['is_corresponding'],
                               source_path=f"{source_id(w['id'])}#authorships[{i}]"))
        au=a['author']
        if au.get('id'):
            cid=cluster(sc_by_source,'SC',source_id(au['id']),'person',au['display_name'],'openalex')
            membership(sm,sid,cid,'identifies_person',f"authorships[{i}].raw_author_name",'openalex imported assignment')
        insts={x['id']:x for x in a.get('institutions') or []}
        evidence={}
        for j, aff in enumerate(affs):
            for iid in aff.get('institution_ids') or []:
                evidence.setdefault(iid,[]).append(f"authorships[{i}].affiliations[{j}].raw_affiliation_string")
        for iid, paths in evidence.items():
            assert iid in insts, iid
            cid=cluster(sc_by_source,'SC',source_id(iid),'institution',insts[iid]['display_name'],'openalex')
            membership(sm,sid,cid,'identifies_institution','\n'.join(paths),'openalex imported assignment')
    loc=w['primary_location']
    sid=f'S{len(signatures)+1}'
    signatures.append(dict(signature_id=sid,content_id=wid,kind='venue_entry',
                           raw_text=loc['raw_source_name'],affiliation_text=None,
                           order=None,position=None,is_corresponding=None,
                           source_path=f"{source_id(w['id'])}#primary_location"))
    src=loc['source']
    cid=cluster(sc_by_source,'SC',source_id(src['id']),'venue',src['display_name'],'openalex')
    membership(sm,sid,cid,'identifies_venue','primary_location.raw_source_name','openalex imported assignment')
    pt=w['primary_topic']
    cid=cluster(cc_by_source,'CC',source_id(pt['id']),'topic',pt['display_name'],'openalex topics')
    membership(cm,wid,cid,'about','primary_topic','openalex imported assignment',pt['score'],True)
    # Deliberate small selection: one primary topic and two shared keywords.
    for j,k in enumerate(w['keywords']):
        if k['display_name'] in ('Neuroscience','Endocrinology'):
            cid=cluster(cc_by_source,'CC',source_id(k['id']),'keyword',k['display_name'],'openalex keywords')
            membership(cm,wid,cid,'about',f'keywords[{j}]','openalex imported assignment',k['score'],None)
    for target in w['referenced_works'] or []:
        references.append(dict(from_content_id=wid,to_source_id=source_id(target),relation='cites',method='openalex referenced_works'))

signature_clusters=list(sc_by_source.values())
content_clusters=list(cc_by_source.values())
content_by_id={x['content_id']:x for x in content}
sig_by_id={x['signature_id']:x for x in signatures}
metrics=[]
metric_members=[]


def add_metrics(family, clusters, links):
    for c in clusters:
        memberships=[x for x in links if x['cluster_id']==c['cluster_id']]
        member_ids={x['member_id'] for x in memberships}
        ids=sorted({sig_by_id[x]['content_id'] for x in member_ids} if family=='signature' else member_ids)
        rows=[content_by_id[x] for x in ids]
        citations=sorted((r['citations'] for r in rows),reverse=True)
        h=max((i for i,n in enumerate(citations,1) if n>=i),default=0)
        metrics.append(dict(cluster_family=family,cluster_id=c['cluster_id'],name=c['name'],
                            member_count=len(member_ids),content_count=len(ids),
                            citation_sum=sum(citations),mean_citations=sum(citations)/len(ids),h_index=h,
                            first_year=min(r['year'] for r in rows),last_year=max(r['year'] for r in rows),scope=SCOPE))
        for wid in ids:
            metric_members.append(dict(cluster_family=family,cluster_id=c['cluster_id'],content_id=wid))


add_metrics('signature',signature_clusters,sm)
add_metrics('content',content_clusters,cm)
tables=dict(content=content,signatures=signatures,signature_clusters=signature_clusters,
            signature_memberships=sm,content_clusters=content_clusters,content_memberships=cm,
            metric_content=metric_members,metrics=metrics,content_references=references)

# Check foreign keys, distinct membership, and deduplication before exporting.
assert len(content)==2 and len(signatures)==7
assert len(signature_clusters)==9 and len(content_clusters)==4
assert len(sm)==13 and len(cm)==6
assert set(sm[0])==set(cm[0])
for rows in tables.values(): scalar_check(rows)
for rows, members, groups in [(sm,set(sig_by_id),{c['cluster_id'] for c in signature_clusters}),
                              (cm,set(content_by_id),{c['cluster_id'] for c in content_clusters})]:
    assert all(x['member_id'] in members and x['cluster_id'] in groups for x in rows)
    keys=[(x['member_id'],x['cluster_id'],x['relation']) for x in rows]
    assert len(keys)==len(set(keys))
for m in metrics:
    if m['name']=='Yvette Taché': assert (m['member_count'],m['content_count'],m['citation_sum'],m['h_index'])==(2,2,110,2)
    if m['name']=='University of California, Los Angeles':
        assert (m['member_count'],m['content_count'],m['citation_sum'],m['h_index'])==(4,2,110,2)
assert len(references)==248
assert not ({r['to_source_id'] for r in references} & {c['source_id'] for c in content})
for name, rows in tables.items():
    with (HERE/f'{name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

notes = [
    'Real source: two saved OpenAlex work records, not live API values. Short IDs P/S/SC/CC are local display aliases.',
    'Content shows selected metadata columns; it is not a replacement for the full production content schema.',
    'Example choice: one author signature contains a raw name plus its affiliation text. Venue attribution is another signature kind.',
    'Affiliations are joined with line breaks only for display. Original field boundaries remain in the source fixture and membership evidence paths.',
    'Institution assignments use only the exact raw-affiliation mappings supplied by OpenAlex. Unmatched lines remain in the signatures, without invented links.',
    'Only the primary venue, primary topic, and two shared keywords are shown. Other locations/classifications and funding metadata are outside this example.',
    'Missing identity confidence is shown as unknown, not 1.0. Topic and keyword scores are imported source scores, not comparable identity probabilities.',
    'All metrics use distinct content IDs within these two papers. They are not lifetime profile metrics.',
    'Citation totals use the papers’ incoming citation counts in the saved snapshot. The two example papers do not cite each other.',
    'The full reference list is exported separately: 248 edges, with endpoints outside this two-paper example.',
]
payload=dict(tables=tables,notes=notes,scope=SCOPE)
(HERE/'data.js').write_text('window.EXAMPLE = '+json.dumps(payload,ensure_ascii=False,default=str)+';\n')
print(json.dumps({name:len(rows) for name,rows in tables.items()},indent=2))
print('All assertions passed. Only the two-paper fixture was read.')
