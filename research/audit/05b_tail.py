"""Remaining work-level checks (05 crashed on a keyword); plus a cheap English check on 'en' titles."""
import duckdb, json, os, re
HERE=os.path.dirname(os.path.abspath(__file__))
con=duckdb.connect(); con.execute("SET threads=8")
PUB="type IN ('article','review','conference-paper','conference-abstract','preprint','book','book-chapter','dissertation','report','letter','editorial','data-paper','software-paper') AND NOT is_paratext"
con.execute(f"CREATE TABLE w AS SELECT id,title,type,is_paratext,publication_year,language,has_abstract,is_oa,has_fulltext, ({PUB}) AS is_pub FROM read_parquet('{HERE}/sample/*.parquet')")
R={}
def q(name, sql):
    r=con.execute(sql).fetchall(); R[name]=r; print(name); [print('   ',x) for x in r[:40]]; return r
q('title_norm_dups_in_sample', """WITH t AS (SELECT lower(regexp_replace(title,'[^a-zA-Z0-9]','','g')) nt, count(*) c FROM w WHERE is_pub AND title IS NOT NULL AND length(title)>30 GROUP BY 1 HAVING count(*)>1) SELECT count(*) AS n_groups, sum(c-1) AS extra FROM t""")
q('title_dup_examples', """WITH t AS (SELECT lower(regexp_replace(title,'[^a-zA-Z0-9]','','g')) nt, list(type) AS tys, list(publication_year) AS yrs, count(*) c FROM w WHERE is_pub AND title IS NOT NULL AND length(title)>30 GROUP BY 1 HAVING count(*)>1 AND count(*)<20) SELECT nt[1:60], tys, yrs FROM t ORDER BY random() LIMIT 12""")
q('preprint_share', "SELECT round(100.0*sum((type='preprint')::int)/count(*),2) FROM w WHERE is_pub")
q('oa', "SELECT round(100.0*sum(is_oa::int)/count(*),1) pct_oa, round(100.0*sum(has_fulltext::int)/count(*),1) pct_fulltext FROM w WHERE is_pub")
q('title_missing', "SELECT round(100.0*sum((title IS NULL OR length(title)<3)::int)/count(*),2) FROM w WHERE is_pub")
# English check: a title tagged 'en' that has >=4 words and no English stopword and not mostly ASCII-uppercase acronyms
rows=con.execute("SELECT id,title FROM w WHERE is_pub AND language='en' AND has_abstract AND title IS NOT NULL AND length(title)>25 ORDER BY random() LIMIT 20000").fetchall()
SW=set('the of and in a to for on with by from an at as is are between during after under using into or its their new study analysis effect effects based case review model role via versus among'.split())
NONEN=set('de la le les des du der die und das el los las en y del een van het di il della delle para por com uma um na no do dos das zur für mit auf ein eine dans sur pour au aux'.split())
bad=[]; n=0
for wid,t in rows:
    words=re.findall(r"[A-Za-zÀ-ÿ']+",t.lower())
    if len(words)<4: continue
    n+=1
    if not (set(words)&SW) and (set(words)&NONEN or re.search(r'[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0600-\u06FF]',t)): bad.append(t)
print(f'english_check: {n} en-tagged titles with >=4 words; {len(bad)} look non-English ({len(bad)/n*100:.2f}%)'); R['english_check']=[[n,len(bad),round(len(bad)/n*100,2)]]; R['english_check_examples']=bad[:25]
for b in bad[:12]: print('   ',b)
json.dump(R, open(f'{HERE}/results_works_tail.json','w'), indent=1, default=str)
