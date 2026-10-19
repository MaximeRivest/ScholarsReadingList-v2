"""Author-identity audit on the 1M-work sample (authorships exploded)."""
import duckdb, json, os
HERE=os.path.dirname(os.path.abspath(__file__))
con=duckdb.connect(); con.execute("SET threads=8")
S=f"read_parquet('{HERE}/sample/*.parquet')"
PUB="type IN ('article','review','conference-paper','conference-abstract','preprint','book','book-chapter','dissertation','report','letter','editorial','data-paper','software-paper') AND NOT is_paratext"
con.execute(f"""CREATE TABLE au AS
SELECT w.id AS work, w.publication_year AS year, w.type, ({PUB}) AS is_pub, w.is_xpac,
       a.author.id AS aid, a.author.display_name AS aname, a.author.orcid AS aorcid,
       a.raw_author_name AS raw_name, a.raw_orcid, a.author_position AS apos,
       len(a.raw_affiliation_strings) AS n_raw_aff, len(a.institutions) AS n_inst,
       a.affiliations AS affs, a.countries AS countries, a.is_corresponding
FROM {S} w, UNNEST(w.authorships) AS t(a)""")
R={}
def q(name, sql):
    r=con.execute(sql).fetchall(); R[name]=r; print(name, r[:25]); return r
q('authorships_total', "SELECT count(*), count(DISTINCT work), sum(is_pub::int) FROM au")
q('author_id_missing', "SELECT is_pub, count(*) n, sum((aid IS NULL)::int) no_id, round(100.0*sum((aid IS NULL)::int)/count(*),2) pct FROM au GROUP BY 1")
q('orcid_coverage', "SELECT is_pub, round(100.0*sum((aorcid IS NOT NULL)::int)/count(*),2) author_record_has_orcid, round(100.0*sum((raw_orcid IS NOT NULL)::int)/count(*),2) publisher_gave_orcid FROM au GROUP BY 1")
q('orcid_by_year', "SELECT year//5*5 y5, count(*) n, round(100.0*sum((raw_orcid IS NOT NULL)::int)/count(*),1) raw_orcid_pct, round(100.0*sum((aorcid IS NOT NULL)::int)/count(*),1) rec_orcid_pct FROM au WHERE is_pub AND year BETWEEN 1990 AND 2026 GROUP BY 1 ORDER BY 1")
# Direct error signal: publisher gave an ORCID, OpenAlex record carries a *different* ORCID
q('orcid_contradiction', """SELECT count(*) both_present,
   sum((lower(replace(raw_orcid,'https://orcid.org/','')) <> lower(replace(aorcid,'https://orcid.org/','')))::int) differ,
   round(100.0*sum((lower(replace(raw_orcid,'https://orcid.org/','')) <> lower(replace(aorcid,'https://orcid.org/','')))::int)/count(*),3) pct
   FROM au WHERE raw_orcid IS NOT NULL AND aorcid IS NOT NULL AND aid IS NOT NULL""")
q('orcid_given_but_record_has_none', """SELECT count(*) n, round(100.0*sum((aorcid IS NULL)::int)/count(*),2) pct_record_without_orcid FROM au WHERE raw_orcid IS NOT NULL AND aid IS NOT NULL""")
# SPLIT: one ORCID (publisher-given, normalised) -> several OpenAlex author ids, among ORCIDs seen >= 2 times
q('split_by_raw_orcid', """WITH g AS (SELECT lower(replace(raw_orcid,'https://orcid.org/','')) o, count(*) n, count(DISTINCT aid) k FROM au WHERE raw_orcid IS NOT NULL AND aid IS NOT NULL GROUP BY 1 HAVING count(*)>=2)
  SELECT count(*) orcids_seen_twice_plus, sum((k>1)::int) split_into_several_records, round(100.0*sum((k>1)::int)/count(*),2) pct, sum(k-1) extra_records FROM g""")
q('split_by_raw_orcid_by_n', """WITH g AS (SELECT lower(replace(raw_orcid,'https://orcid.org/','')) o, count(*) n, count(DISTINCT aid) k FROM au WHERE raw_orcid IS NOT NULL AND aid IS NOT NULL GROUP BY 1)
  SELECT CASE WHEN n>=10 THEN '10+' WHEN n>=5 THEN '5-9' WHEN n>=3 THEN '3-4' ELSE '2' END bucket, count(*) orcids, round(100.0*sum((k>1)::int)/count(*),2) pct_split FROM g WHERE n>=2 GROUP BY 1 ORDER BY 1""")
# same using OpenAlex's own author.orcid: two author records carrying the same ORCID = certain split OpenAlex could fix itself
q('split_by_record_orcid', """WITH g AS (SELECT aorcid, count(*) n, count(DISTINCT aid) k FROM au WHERE aorcid IS NOT NULL AND aid IS NOT NULL GROUP BY 1 HAVING count(*)>=2)
  SELECT count(*) orcids_seen_twice_plus, sum((k>1)::int) with_several_records, round(100.0*sum((k>1)::int)/count(*),3) pct FROM g""")
# MERGE: one OpenAlex author id -> several distinct publisher ORCIDs
q('merge_by_raw_orcid', """WITH g AS (SELECT aid, count(*) n, count(DISTINCT lower(replace(raw_orcid,'https://orcid.org/',''))) k FROM au WHERE raw_orcid IS NOT NULL AND aid IS NOT NULL GROUP BY 1 HAVING count(*)>=2)
  SELECT count(*) records_with_2plus_orcid_authorships, sum((k>1)::int) carrying_several_orcids, round(100.0*sum((k>1)::int)/count(*),2) pct FROM g""")
q('merge_by_raw_orcid_by_n', """WITH g AS (SELECT aid, count(*) n, count(DISTINCT lower(replace(raw_orcid,'https://orcid.org/',''))) k FROM au WHERE raw_orcid IS NOT NULL AND aid IS NOT NULL GROUP BY 1)
  SELECT CASE WHEN n>=10 THEN '10+' WHEN n>=5 THEN '5-9' WHEN n>=3 THEN '3-4' ELSE '2' END bucket, count(*) records, round(100.0*sum((k>1)::int)/count(*),2) pct_merged FROM g WHERE n>=2 GROUP BY 1 ORDER BY 1""")
# MERGE by surname: one record, raw names whose last token differs (loose; transliteration/marriage inflate it)
con.execute("""CREATE TABLE nm AS SELECT aid, raw_name, lower(regexp_replace(strip_accents(raw_name), '[^A-Za-z ]', '', 'g')) norm FROM au WHERE aid IS NOT NULL AND raw_name IS NOT NULL""")
q('merge_by_surname', """WITH g AS (SELECT aid, count(*) n, count(DISTINCT CASE WHEN norm LIKE '% %' THEN split_part(norm,' ',-1) ELSE norm END) k FROM nm GROUP BY 1 HAVING count(*)>=2)
  SELECT count(*) records_seen_twice_plus, sum((k>1)::int) with_different_surnames, round(100.0*sum((k>1)::int)/count(*),2) pct FROM g""")
q('merge_by_surname_examples', """WITH g AS (SELECT aid, list(DISTINCT raw_name) nms, count(*) n, count(DISTINCT CASE WHEN norm LIKE '% %' THEN split_part(norm,' ',-1) ELSE norm END) k FROM nm GROUP BY 1 HAVING count(*)>=3)
  SELECT aid, nms[1:6], n FROM g WHERE k>1 USING SAMPLE 25""")
# authorship share where name on the paper has a different first initial than the record's display name
q('first_initial_mismatch', """SELECT count(*) n, round(100.0*sum((lower(left(strip_accents(raw_name),1)) <> lower(left(strip_accents(aname),1)) AND lower(split_part(strip_accents(raw_name),' ',-1)) <> lower(split_part(strip_accents(aname),' ',-1)))::int)/count(*),2) pct_neither_initial_nor_surname_match
  FROM au WHERE aid IS NOT NULL AND raw_name IS NOT NULL AND aname IS NOT NULL AND raw_name LIKE '% %' AND aname LIKE '% %'""")
q('mismatch_examples', """SELECT raw_name, aname, aid, work FROM au WHERE aid IS NOT NULL AND raw_name IS NOT NULL AND aname IS NOT NULL AND raw_name LIKE '% %' AND aname LIKE '% %'
  AND lower(left(strip_accents(raw_name),1)) <> lower(left(strip_accents(aname),1)) AND lower(split_part(strip_accents(raw_name),' ',-1)) <> lower(split_part(strip_accents(aname),' ',-1)) USING SAMPLE 20""")
# Affiliation matching
q('affiliation_matching', """SELECT is_pub, count(*) n,
   round(100.0*sum((n_raw_aff>0)::int)/count(*),2) has_raw_affiliation,
   round(100.0*sum((n_inst>0)::int)/count(*),2) has_institution,
   round(100.0*sum((n_raw_aff>0 AND n_inst=0)::int)/count(*),2) raw_but_unmatched,
   round(100.0*sum((n_raw_aff=0 AND n_inst>0)::int)/count(*),2) institution_without_raw
   FROM au GROUP BY 1""")
q('affiliation_by_year', """SELECT year//5*5 y5, count(*) n, round(100.0*sum((n_raw_aff>0)::int)/count(*),1) has_raw, round(100.0*sum((n_inst>0)::int)/count(*),1) has_inst, round(100.0*sum((n_raw_aff>0 AND n_inst=0)::int)/sum((n_raw_aff>0)::int),1) unmatched_of_raw FROM au WHERE is_pub AND year BETWEEN 1990 AND 2026 GROUP BY 1 ORDER BY 1""")
q('unmatched_affiliation_examples', """SELECT a.raw_affiliation_string FROM au, UNNEST(affs) t(a) WHERE len(a.institution_ids)=0 AND a.raw_affiliation_string IS NOT NULL USING SAMPLE 40""")
q('authors_per_paper_zero', "SELECT round(100.0*sum((authors_count=0 OR authors_count IS NULL)::int)/count(*),2) FROM "+S+" WHERE "+PUB)
json.dump({k:[list(map(lambda x: x if not hasattr(x,'isoformat') else str(x), row)) for row in v] for k,v in R.items()}, open(f'{HERE}/results_authors.json','w'), indent=1, default=str)
con.execute(f"COPY (SELECT * EXCLUDE (affs) FROM au) TO '{HERE}/sample_authorships.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)")
q('split_record_orcid_examples', """WITH g AS (SELECT aorcid, list(DISTINCT aid) ids, list(DISTINCT aname) nms, count(*) n FROM au WHERE aorcid IS NOT NULL AND aid IS NOT NULL GROUP BY 1 HAVING count(DISTINCT aid)>1) SELECT * FROM g USING SAMPLE 12""")
q('orcid_contradiction_examples', """SELECT raw_name, aname, raw_orcid, aorcid, work FROM au WHERE raw_orcid IS NOT NULL AND aorcid IS NOT NULL AND lower(replace(raw_orcid,'https://orcid.org/','')) <> lower(replace(aorcid,'https://orcid.org/','')) USING SAMPLE 12""")
q('no_author_id_by_year', """SELECT year//5*5 y5, count(*) n, round(100.0*sum((aid IS NULL)::int)/count(*),1) pct_no_id FROM au WHERE is_pub AND year BETWEEN 1950 AND 2026 GROUP BY 1 ORDER BY 1""")
q('no_author_id_by_type', """SELECT type, count(*) n, round(100.0*sum((aid IS NULL)::int)/count(*),1) pct_no_id FROM au GROUP BY 1 ORDER BY 2 DESC LIMIT 12""")
q('no_author_id_examples', """SELECT raw_name, work, year, type FROM au WHERE aid IS NULL AND is_pub USING SAMPLE 15""")
json.dump({k:[list(map(lambda x: x if not hasattr(x,'isoformat') else str(x), row)) for row in v] for k,v in R.items()}, open(f'{HERE}/results_authors.json','w'), indent=1, default=str)
