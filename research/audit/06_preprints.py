"""How often does a preprint in the corpus have a published twin under a different record? 150 random preprints (2016-2022), exact-title keyword search on the local API."""
import duckdb, json, re, time, urllib.request, urllib.parse
con=duckdb.connect()
rows=con.execute("""SELECT id, title, publication_year, raw_source_name, source_name FROM read_parquet('research/audit/sample/*.parquet')
 WHERE type='preprint' AND language='en' AND has_abstract AND publication_year BETWEEN 2016 AND 2022 AND length(title)>40 ORDER BY random() LIMIT 150""").fetchall()
norm=lambda t: re.sub(r'[^a-z0-9]','',t.lower())
B='http://127.0.0.1:8810'
res=[]; twins=0; checked=0
for wid,title,yr,raw,src in rows:
    q='title:"'+title.replace('"','')+'"'
    try:
        r=json.load(urllib.request.urlopen(f"{B}/search?"+urllib.parse.urlencode({"q":q,"k":20,"mode":"keyword","abstracts":"false"}),timeout=60))
    except Exception as e:
        print('err',e); continue
    checked+=1
    hits=[h for h in r.get('results',[]) if norm(h['title'])==norm(title) and h['id']!=wid.rsplit('/',1)[-1]]
    pub=[h for h in hits if h['type']!='preprint']
    res.append({'preprint':wid,'year':yr,'source':src or raw,'title':title[:100],'same_title_other_records':len(hits),'published_twin':[(h['id'],h['type'],h['venue'],h['year']) for h in pub][:3]})
    twins+= bool(pub)
    time.sleep(0.05)
print(f'{checked} preprints checked; {twins} have a non-preprint record with the identical title ({twins/checked:.0%})')
import collections; print(collections.Counter(r['source'] for r in res).most_common(8))
json.dump({'checked':checked,'with_published_twin':twins,'rows':res},open('research/audit/results_preprints.json','w'),indent=1)
