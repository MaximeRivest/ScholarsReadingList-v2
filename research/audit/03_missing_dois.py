"""Are the 49 expert-test DOIs absent from the site's corpus also absent from the raw snapshot, or present without an English abstract?"""
import json, csv, duckdb, collections
rows=json.load(open('research/audit/topics_expert200.json'))
arts={r['scopus_id']:r for r in csv.DictReader(open('/home/maxime/Projects/jevarticle/data/articles_resolved.csv'))}
miss=[(r['doi'].lower(), int(arts[r['scopus_id']]['year'])) for r in rows if not r['W']]
by=collections.defaultdict(list)
for d,y in miss: by[y].append(d)
con=duckdb.connect(); con.execute("SET threads=6")
found=[]
alld=[d for d,_ in miss]
for y in sorted(by):
    f=f'/mnt/bulk/openalex/content/year={y}/data.parquet'
    r=con.execute("SELECT id, lower(replace(doi,'https://doi.org/','')) d, language, length(abstract) alen, type, year, is_complete, venue FROM read_parquet(?) WHERE lower(replace(doi,'https://doi.org/','')) IN (SELECT unnest(?))",[f,alld]).fetchall()
    found+=r; print(y, len(r), flush=True)
print(len(miss),'missing;',len(found),'found in raw snapshot')
for r in found: print(r)
json.dump({'missing':miss,'found_in_raw':found},open('research/audit/expert200_missing_dois.json','w'),indent=1,default=str)
