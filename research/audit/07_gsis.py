"""'Greater South Information System' (DOI prefix 10.60692): are these records copies of papers already in OpenAlex? 120 random ones, exact-title keyword search on the local API (which only holds works with an English abstract: a twin found here is a lower bound)."""
import duckdb, json, re, urllib.request, urllib.parse, collections
con=duckdb.connect()
rows=con.execute("""SELECT id, title, publication_year FROM read_parquet('research/audit/sample/*.parquet') WHERE source_name='Greater South Information System' AND language='en' AND length(title)>40 ORDER BY random() LIMIT 120""").fetchall()
norm=lambda t: re.sub(r'[^a-z0-9]','',t.lower())
B='http://127.0.0.1:8810'; res=[]; twin=0; pref=collections.Counter()
for wid,title,yr in rows:
    q='title:"'+title.replace('"','')+'"'
    try: r=json.load(urllib.request.urlopen(f"{B}/search?"+urllib.parse.urlencode({"q":q,"k":20,"mode":"keyword","abstracts":"false"}),timeout=60))
    except Exception as e: print('err',e); continue
    hits=[h for h in r.get('results',[]) if norm(h['title'])==norm(title) and h['id']!=wid.rsplit('/',1)[-1] and h['venue']!='Greater South Information System']
    twin+=bool(hits)
    for h in hits[:1]: pref[(h['doi'] or 'nodoi').split('/')[0]]+=1
    res.append({'gsis':wid,'year':yr,'title':title[:100],'twins':[(h['id'],h['venue'],h['doi'],h['year']) for h in hits][:3]})
print(f'{len(res)} GSIS records; {twin} have a same-title record in the English-abstract corpus ({twin/len(res):.0%}); twin DOI prefixes: {pref.most_common(8)}')
json.dump({'checked':len(res),'with_twin_in_corpus':twin,'twin_prefixes':pref.most_common(),'rows':res},open('research/audit/results_gsis.json','w'),indent=1)
