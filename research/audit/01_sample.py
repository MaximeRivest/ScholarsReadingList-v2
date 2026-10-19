"""Draw a ~1M-work random sample from the raw OpenAlex works snapshot (2026-09-23).

Files are chosen weighted by size (see sample_files.json); within each file we keep a
random 1-in-N of rows so the sample is close to uniform over works. Output: one parquet
with a flat projection plus the nested authorships, in research/audit/sample/.
Read-only on the source; CPU only; 8 threads at nice 10.
"""
import json, os, duckdb, time
HERE=os.path.dirname(os.path.abspath(__file__))
files=json.load(open(f'{HERE}/sample_files.json'))
out=f'{HERE}/sample'; os.makedirs(out,exist_ok=True)
con=duckdb.connect(); con.execute("SET threads=8; SET memory_limit='24GB'; SET preserve_insertion_order=false;")
Q="""
SELECT id, doi, title, type, language, publication_year, publication_date, created_date, updated_date,
       is_paratext, is_retracted, is_xpac, cited_by_count, referenced_works_count, authors_count,
       institutions_distinct_count, countries_distinct_count, locations_count, has_fulltext,
       abstract_inverted_index IS NOT NULL AS has_abstract,
       length(abstract_inverted_index) AS abstract_len,
       ids['pmid'] AS pmid, ids['pmcid'] AS pmcid, ids['mag'] AS mag,
       indexed_in,
       primary_topic.id AS topic_id, primary_topic.display_name AS topic_name, primary_topic.score AS topic_score,
       primary_topic.subfield.id AS subfield_id, primary_topic.subfield.display_name AS subfield_name,
       primary_topic.field.id AS field_id, primary_topic.field.display_name AS field_name,
       primary_topic.domain.display_name AS domain_name,
       len(topics) AS n_topics,
       [t.field.id FOR t IN topics] AS topic_fields,
       [t.score FOR t IN topics] AS topic_scores,
       len(keywords) AS n_keywords, len(concepts) AS n_concepts, len(mesh) AS n_mesh,
       primary_location.source.id AS source_id, primary_location.source.display_name AS source_name,
       primary_location.source.type AS source_type, primary_location.raw_source_name AS raw_source_name,
       primary_location.raw_type AS raw_type, primary_location.version AS version,
       primary_location.landing_page_url AS landing_url,
       [l.source.id FOR l IN locations] AS location_source_ids,
       [l.raw_source_name FOR l IN locations] AS location_raw_sources,
       open_access.is_oa AS is_oa, open_access.oa_status AS oa_status,
       biblio.volume AS volume, biblio.issue AS issue, biblio.first_page AS first_page, biblio.last_page AS last_page,
       authorships, related_works IS NOT NULL AS has_related,
       fwci
FROM read_parquet(?) 
WHERE hash(id) % 7 = 0
"""
t0=time.time()
for i,f in enumerate(files):
    dst=f'{out}/part_{i:02d}.parquet'
    if os.path.exists(dst): continue
    con.execute(f"COPY ({Q}) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)", [f])
    n=con.execute("SELECT count(*) FROM read_parquet(?)",[dst]).fetchone()[0]
    print(i, os.path.basename(os.path.dirname(f)), os.path.basename(f), n, f'{time.time()-t0:.0f}s', flush=True)
