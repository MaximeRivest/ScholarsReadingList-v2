"""Parsed references -> OpenAlex works, in order: arXiv id, DOI, title (+ year / first author), volume + first page
+ year (+ first author). Lookup tables: /mnt/bulk/arxiv/match/*.parquet (build_lookup.sql there).
    python match.py REFS.jsonl OUT.parquet CITING.json      (CITING: {aid: citing W number}, to drop self links)"""
import json, sys
import duckdb
refs, out, citing = sys.argv[1:4]
M = "/mnt/bulk/arxiv/match"
con = duckdb.connect(config={"threads": 16, "memory_limit": "40GB"})
con.execute("""CREATE MACRO tkey(t) AS md5_number_lower(left(regexp_replace(lower(coalesce(t,'')), '[^a-z0-9]', '', 'g'), 200));
CREATE MACRO norm_sur(s) AS regexp_replace(lower(strip_accents(coalesce(s,''))), '[^a-z]', '', 'g');""")
con.execute("CREATE TABLE citing AS SELECT key aid, CAST(value AS BIGINT) cw FROM json_each(?)", [json.dumps(json.load(open(citing)))])
con.execute(f"""CREATE TABLE r AS SELECT aid, n, title, year AS y, norm_sur(surnames[1]) au,
    CASE WHEN length(regexp_replace(lower(coalesce(title,'')), '[^a-z0-9]', '', 'g')) >= 12 THEN tkey(title) END tk,
    nullif(rtrim(lower(doi), '.,;)'), '') doi, nullif(lower(arxiv), '') ax,
    nullif(lower(regexp_replace(volume, '[^0-9A-Za-z]', '', 'g')), '') vol, nullif(lower(regexp_extract(page, '^([0-9A-Za-z]+)', 1)), '') p1
  FROM read_json_auto('{refs}', format='newline_delimited')""")
con.execute(f"""CREATE TABLE m AS
WITH ax AS (SELECT r.aid, r.n, CAST(regexp_extract(o.id, 'W(\\d+)', 1) AS BIGINT) w FROM r
            JOIN read_parquet('/mnt/bulk/arxiv/openalex_vs_arxiv.parquet') o ON o.aid = r.ax),
doi AS (SELECT r.aid, r.n, min(d.w) w FROM r JOIN read_parquet('{M}/doi.parquet') d ON d.doi = r.doi GROUP BY ALL),
ti AS (SELECT aid, n, w FROM (
         SELECT r.aid, r.n, t.w, (t.au = r.au AND r.au <> '')::INT * 2 + (abs(t.year - r.y) <= 1)::INT AS s,
                row_number() OVER (PARTITION BY r.aid, r.n ORDER BY (t.au = r.au AND r.au <> '')::INT * 2 + (abs(t.year - r.y) <= 1)::INT DESC, t.citations DESC) k,
                r.y, r.au FROM r JOIN read_parquet('{M}/title.parquet') t ON t.tk = r.tk)
       WHERE k = 1 AND (s >= 1 OR (y IS NULL AND au = ''))),
vp AS (SELECT aid, n, w FROM (
         SELECT r.aid, r.n, v.w, (v.au = r.au AND r.au <> '') am,
                count(*) OVER (PARTITION BY r.aid, r.n) c,
                row_number() OVER (PARTITION BY r.aid, r.n ORDER BY (v.au = r.au AND r.au <> '') DESC, v.citations DESC) k
         FROM r JOIN read_parquet('{M}/volpage.parquet') v ON v.year = r.y AND v.vol = r.vol AND v.p1 = r.p1 WHERE r.tk IS NULL)
       WHERE k = 1 AND (am OR c = 1))
SELECT r.aid, r.n, coalesce(ax.w, doi.w, ti.w, vp.w) w,
       CASE WHEN ax.w IS NOT NULL THEN 'arxiv' WHEN doi.w IS NOT NULL THEN 'doi' WHEN ti.w IS NOT NULL THEN 'title'
            WHEN vp.w IS NOT NULL THEN 'volpage' END how
FROM r LEFT JOIN ax USING (aid, n) LEFT JOIN doi USING (aid, n) LEFT JOIN ti USING (aid, n) LEFT JOIN vp USING (aid, n)""")
con.execute("""DELETE FROM m USING citing WHERE m.aid = citing.aid AND m.w = citing.cw""")
con.execute(f"COPY m TO '{out}' (FORMAT parquet)")
print(con.execute("SELECT how, count(*), round(100.0*count(*)/sum(count(*)) OVER (), 1) FROM m GROUP BY how ORDER BY 2 DESC").fetchall())
