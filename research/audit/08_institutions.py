"""Institution linking: does the linked institution's name appear in the raw affiliation string? (rough signal; translated names and acronyms count as misses, so read the examples)"""
import duckdb, json, re, os, unicodedata
HERE=os.path.dirname(os.path.abspath(__file__))
con=duckdb.connect(); con.execute("SET threads=8")
rows=con.execute(f"""SELECT a.affiliations[1].raw_affiliation_string AS raw, i.display_name AS inst, i.country_code AS cc, i.type AS itype, w.id
 FROM read_parquet('{HERE}/sample/*.parquet') w, UNNEST(w.authorships) t(a), UNNEST(a.institutions) u(i)
 WHERE w.type IN ('article','review','conference-paper') AND len(a.affiliations)=1 AND len(a.affiliations[1].institution_ids)=1 AND len(a.institutions)=1 AND a.affiliations[1].raw_affiliation_string IS NOT NULL
 ORDER BY random() LIMIT 20000""").fetchall()
STOP=set('university universite universitat universidad universita institute institut instituto hospital school college department center centre national research laboratory laboratories medical medicine sciences science technology state faculty academy general city county clinic clinical foundation health system group company inc ltd gmbh'.split())
def toks(s):
    s=unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()
    return {t for t in re.findall(r'[a-z]{4,}',s) if t not in STOP}
miss=[]; n=0; nomatch=0
for raw,inst,cc,itype,wid in rows:
    it=toks(inst)
    if not it: continue
    n+=1
    rt=toks(raw)
    if not (it & rt):
        nomatch+=1; miss.append((raw[:120],inst,cc,itype))
print(f'{n} single-institution authorships; {nomatch} where no distinctive word of the institution name appears in the affiliation string ({nomatch/n*100:.1f}%)')
import random; random.seed(3); ex=random.sample(miss,40)
for m in ex: print('   ',m)
# country consistency: raw string mentions a country word different from the institution's country
json.dump({'checked':n,'no_name_overlap':nomatch,'pct':round(nomatch/n*100,2),'examples':ex},open(f'{HERE}/results_institutions.json','w'),indent=1)
