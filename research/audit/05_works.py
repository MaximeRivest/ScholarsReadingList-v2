"""Work-level audit on the 1M sample: topics, abstracts, dates, venues, languages, retractions, references."""
import duckdb, json, os
HERE=os.path.dirname(os.path.abspath(__file__))
con=duckdb.connect(); con.execute("SET threads=8")
PUB="type IN ('article','review','conference-paper','conference-abstract','preprint','book','book-chapter','dissertation','report','letter','editorial','data-paper','software-paper') AND NOT is_paratext"
con.execute(f"CREATE TABLE w AS SELECT *, ({PUB}) AS is_pub, (CAST(substr(id,23) AS BIGINT) >= 6000000000) AS new_wid FROM read_parquet('{HERE}/sample/*.parquet')")
R={}
def q(name, sql):
    r=con.execute(sql).fetchall(); R[name]=r; print(name); [print('   ',x) for x in r[:40]]
    json.dump({k:[[str(x) if hasattr(x,'isoformat') else x for x in row] for row in v] for k,v in R.items()}, open(f'{HERE}/results_works.json','w'), indent=1, default=str); return r
q('scope', "SELECT is_pub, is_xpac, new_wid, count(*) FROM w GROUP BY 1,2,3 ORDER BY 1,2,3")
# ---- topics
q('topic_missing', "SELECT is_pub, has_abstract, count(*) n, round(100.0*sum((topic_id IS NULL)::int)/count(*),2) pct_no_topic FROM w GROUP BY 1,2 ORDER BY 1,2")
q('topic_missing_pub_by_new', "SELECT new_wid, is_xpac, count(*) n, round(100.0*sum((topic_id IS NULL)::int)/count(*),2) pct_no_topic FROM w WHERE is_pub GROUP BY 1,2 ORDER BY 1,2")
q('topic_missing_pub_by_year', "SELECT publication_year//10*10 AS decade, count(*) n, round(100.0*sum((topic_id IS NULL)::int)/count(*),1) pct_no_topic, round(100.0*sum((NOT has_abstract)::int)/count(*),1) pct_no_abstract FROM w WHERE is_pub AND publication_year BETWEEN 1900 AND 2026 GROUP BY 1 ORDER BY 1")
q('topic_score_dist', "SELECT has_abstract, count(*) n, round(quantile_cont(topic_score,0.1),3) p10, round(quantile_cont(topic_score,0.5),3) p50, round(quantile_cont(topic_score,0.9),3) p90, round(100.0*sum((topic_score<0.5)::int)/count(*),1) pct_below_half FROM w WHERE is_pub AND topic_id IS NOT NULL GROUP BY 1")
q('topic_fields_disagree', """SELECT count(*) n, round(100.0*sum((len(list_distinct(topic_fields))>1)::int)/count(*),1) pct_topics_span_several_fields,
   round(100.0*sum((len(list_distinct(topic_fields))>2)::int)/count(*),1) pct_three_fields FROM w WHERE is_pub AND n_topics>=2""")
q('topic_second_close', "SELECT round(100.0*sum((topic_scores[2] >= 0.9*topic_scores[1])::int)/count(*),1) pct_second_topic_within_10pct, round(100.0*sum((topic_scores[2] >= 0.9*topic_scores[1] AND topic_fields[2]<>topic_fields[1])::int)/count(*),1) pct_and_other_field FROM w WHERE is_pub AND n_topics>=2")
q('topic_name_examples_no_abstract', "SELECT title, topic_name, subfield_name FROM w WHERE is_pub AND NOT has_abstract AND topic_id IS NOT NULL AND title IS NOT NULL ORDER BY random() LIMIT 15")
q('top_topics', "SELECT topic_name, subfield_name, count(*) n FROM w WHERE is_pub GROUP BY 1,2 ORDER BY 3 DESC LIMIT 15")
q('attractor_topic', "SELECT has_abstract, language IS NULL OR language<>'en' AS non_english, count(*) n, round(100.0*sum((topic_name='Military Technology and Strategies')::int)/count(*),2) pct_military, round(100.0*sum((topic_name='Magnetic confinement fusion research')::int)/count(*),2) pct_fusion FROM w WHERE is_pub GROUP BY 1,2 ORDER BY 1,2")
q('attractor_topic_examples', "SELECT title, language, has_abstract, round(topic_score,2) FROM w WHERE is_pub AND topic_name='Military Technology and Strategies' ORDER BY random() LIMIT 20")
q('attractor_topic_en_abstract_examples', "SELECT title, round(topic_score,2) FROM w WHERE is_pub AND topic_name='Military Technology and Strategies' AND language='en' AND has_abstract ORDER BY random() LIMIT 15")
q('fusion_examples', "SELECT title, language, has_abstract, round(topic_score,2) FROM w WHERE is_pub AND topic_name='Magnetic confinement fusion research' ORDER BY random() LIMIT 12")
q('topic_concentration', "WITH t AS (SELECT topic_name, count(*) c FROM w WHERE is_pub AND topic_id IS NOT NULL GROUP BY 1) SELECT count(*) topics_used, round(100.0*max(c)/sum(c),2) pct_in_largest, round(100.0*sum(CASE WHEN c>=2000 THEN c END)/sum(c),1) pct_in_topics_over_2000 FROM t")
q('keywords_mesh_concepts', "SELECT round(100.0*sum((n_keywords>0)::int)/count(*),1) kw, round(100.0*sum((n_mesh>0)::int)/count(*),1) mesh, round(100.0*sum((n_concepts>0)::int)/count(*),1) concepts FROM w WHERE is_pub")
# ---- abstracts
q('abstract_by_type', "SELECT type, count(*) n, round(100.0*sum(has_abstract::int)/count(*),1) pct_abstract FROM w GROUP BY 1 ORDER BY 2 DESC LIMIT 14")
q('abstract_by_doi_prefix', """SELECT split_part(replace(doi,'https://doi.org/',''),'/',1) prefix, count(*) n, round(100.0*sum(has_abstract::int)/count(*),1) pct_abstract FROM w WHERE is_pub AND type='article' AND doi IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 20""")
q('abstract_by_year', "SELECT publication_year//5*5 y5, count(*) n, round(100.0*sum(has_abstract::int)/count(*),1) pct_abstract FROM w WHERE is_pub AND type IN ('article','review') AND publication_year BETWEEN 1950 AND 2026 GROUP BY 1 ORDER BY 1")
q('abstract_short', "SELECT round(100.0*sum((abstract_len<200)::int)/count(*),2) pct_inverted_index_under_200_chars FROM w WHERE has_abstract AND is_pub")
q('abstract_language', "SELECT language, count(*) n, round(100.0*sum(has_abstract::int)/count(*),1) pct_abstract FROM w WHERE is_pub GROUP BY 1 ORDER BY 2 DESC LIMIT 12")
# ---- language
q('language_missing', "SELECT is_pub, count(*) n, round(100.0*sum((language IS NULL)::int)/count(*),2) pct_no_language FROM w GROUP BY 1")
q('language_en_nonlatin_title', """SELECT count(*) n, round(100.0*sum((regexp_matches(title,'[\\x{0400}-\\x{04FF}\\x{4E00}-\\x{9FFF}\\x{3040}-\\x{30FF}\\x{AC00}-\\x{D7AF}\\x{0600}-\\x{06FF}\\x{0E00}-\\x{0E7F}]'))::int)/count(*),2) pct_en_with_nonlatin_title FROM w WHERE is_pub AND language='en' AND title IS NOT NULL""")
q('language_en_nonlatin_examples', "SELECT title, language FROM w WHERE is_pub AND language='en' AND regexp_matches(title,'[\\x{0400}-\\x{04FF}\\x{4E00}-\\x{9FFF}\\x{3040}-\\x{30FF}\\x{AC00}-\\x{D7AF}\\x{0600}-\\x{06FF}]') ORDER BY random() LIMIT 8")
q('language_en_sample_for_manual_check', "SELECT id, title FROM w WHERE is_pub AND language='en' AND has_abstract AND title IS NOT NULL ORDER BY random() LIMIT 300")
# ---- dates
q('year_missing_or_odd', """SELECT round(100.0*sum((publication_year IS NULL)::int)/count(*),3) pct_no_year, round(100.0*sum((publication_year>2026)::int)/count(*),3) pct_future,
   round(100.0*sum((publication_year<1800)::int)/count(*),3) pct_before_1800, round(100.0*sum((publication_date IS NULL)::int)/count(*),3) pct_no_date,
   round(100.0*sum((month(publication_date)=1 AND day(publication_date)=1)::int)/count(*),1) pct_jan_1 FROM w WHERE is_pub""")
q('year_vs_date_mismatch', "SELECT round(100.0*sum((year(publication_date)<>publication_year)::int)/count(*),3) FROM w WHERE is_pub AND publication_date IS NOT NULL AND publication_year IS NOT NULL")
q('created_vs_pub', "SELECT round(100.0*sum((publication_year > year(created_date))::int)/count(*),2) pct_pubyear_after_record_created FROM w WHERE is_pub")
# ---- venues
q('source_missing', "SELECT is_pub, type, count(*) n, round(100.0*sum((source_id IS NULL)::int)/count(*),1) pct_no_source, round(100.0*sum((source_id IS NULL AND raw_source_name IS NOT NULL)::int)/count(*),1) pct_raw_name_unmatched FROM w WHERE type IN ('article','review','conference-paper','book-chapter','preprint','dissertation') GROUP BY 1,2 ORDER BY 3 DESC")
q('source_type', "SELECT source_type, count(*) n FROM w WHERE is_pub GROUP BY 1 ORDER BY 2 DESC")
q('unmatched_raw_source_examples', "SELECT raw_source_name, type FROM w WHERE is_pub AND source_id IS NULL AND raw_source_name IS NOT NULL ORDER BY random() LIMIT 20")
q('biblio_missing_articles', "SELECT round(100.0*sum((volume IS NULL)::int)/count(*),1) no_volume, round(100.0*sum((first_page IS NULL)::int)/count(*),1) no_first_page FROM w WHERE type='article' AND source_type='journal'")
# ---- retractions
q('retracted', "SELECT round(100.0*sum(is_retracted::int)/count(*),3) pct_flag, sum(is_retracted::int) n_flag, sum((regexp_matches(lower(title),'^(retracted|withdrawn|retraction( note| notice)?:|\\[retracted\\]|retracted:)'))::int) n_title_says_retracted, sum((regexp_matches(lower(title),'^(retracted|withdrawn|\\[retracted\\]|retracted:)') AND NOT is_retracted)::int) title_says_but_not_flagged FROM w WHERE is_pub")
q('retracted_title_examples', "SELECT title, is_retracted, type FROM w WHERE is_pub AND regexp_matches(lower(title),'^(retracted|withdrawn|\\[retracted\\]|retracted:)') AND NOT is_retracted ORDER BY random() LIMIT 10")
# ---- references
q('refs_zero', "SELECT type, count(*) n, round(100.0*sum((referenced_works_count=0)::int)/count(*),1) pct_zero_refs FROM w WHERE is_pub AND publication_year>=2000 GROUP BY 1 ORDER BY 2 DESC LIMIT 8")
q('refs_zero_by_year', "SELECT publication_year//5*5 y5, round(100.0*sum((referenced_works_count=0)::int)/count(*),1) pct_zero_refs FROM w WHERE type IN ('article','review') AND publication_year BETWEEN 1980 AND 2026 GROUP BY 1 ORDER BY 1")
# ---- duplicates inside the sample (1/466 of the corpus: only an upper-bound hint; global checks elsewhere)
q('dup_doi', "SELECT count(*) - count(DISTINCT lower(doi)) FROM w WHERE doi IS NOT NULL")
q('dup_pmid', "SELECT count(*) - count(DISTINCT pmid) FROM w WHERE pmid IS NOT NULL")
q('title_norm_dups_in_sample', """WITH t AS (SELECT lower(regexp_replace(title,'[^a-zA-Z0-9]','','g')) nt, count(*) c FROM w WHERE is_pub AND title IS NOT NULL AND length(title)>30 GROUP BY 1 HAVING count(*)>1) SELECT count(*) AS n_groups, sum(c-1) AS extra FROM t""")
q('title_dup_examples', """WITH t AS (SELECT lower(regexp_replace(title,'[^a-zA-Z0-9]','','g')) nt, list(id) ids, list(type) types, list(publication_year) yrs, count(*) c FROM w WHERE is_pub AND title IS NOT NULL AND length(title)>30 GROUP BY 1 HAVING count(*)>1) SELECT nt[1:60], types, yrs FROM t ORDER BY random() LIMIT 12""")
q('preprint_vs_published', "SELECT round(100.0*sum((type='preprint')::int)/count(*),2) pct_preprints FROM w WHERE is_pub")
q('oa', "SELECT round(100.0*sum(is_oa::int)/count(*),1) pct_oa, round(100.0*sum(has_fulltext::int)/count(*),1) pct_fulltext FROM w WHERE is_pub")
json.dump({k:[[str(x) if hasattr(x,'isoformat') else x for x in row] for row in v] for k,v in R.items()}, open(f'{HERE}/results_works.json','w'), indent=1, default=str)
