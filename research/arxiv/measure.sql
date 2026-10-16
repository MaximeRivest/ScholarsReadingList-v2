-- duckdb (nix run nixpkgs#duckdb -- < research/arxiv/measure.sql): the numbers in coverage.md
SET threads=24; SET memory_limit='60GB';
-- 1. per year: arXiv-DOI works, with references, with abstracts (our tidy OpenAlex table)
COPY (SELECT year,
  count(*) FILTER (WHERE doi ILIKE '%10.48550/arxiv%') arxiv_doi,
  count(*) FILTER (WHERE doi ILIKE '%10.48550/arxiv%' AND references_count>0) arxiv_doi_refs,
  count(*) FILTER (WHERE doi ILIKE '%10.48550/arxiv%' AND abstract IS NOT NULL AND length(abstract)>50) arxiv_doi_abs
FROM read_parquet('/mnt/bulk/openalex/content/year=*/data.parquet') WHERE year BETWEEN 1991 AND 2026 GROUP BY year ORDER BY year)
TO '/mnt/bulk/arxiv/coverage_by_year.csv' (HEADER);
-- 2. OpenAlex's cached full text (raw works snapshot)
COPY (SELECT publication_year y, count(DISTINCT id) n, count(DISTINCT id) FILTER (WHERE has_content.grobid_xml) grobid
FROM read_parquet('/mnt/bulk/openalex/works/*/*.parquet', union_by_name=true) WHERE doi ILIKE '%10.48550/arxiv.%' GROUP BY 1 ORDER BY 1)
TO '/mnt/bulk/arxiv/content_by_year.csv' (HEADER);
-- 3. OpenAlex vs arXiv's own metadata (titles compared in Python: word-set overlap < 0.3)
COPY (
WITH oa AS (SELECT id, lower(regexp_extract(doi, '10\.48550/arxiv\.(.+)$', 1)) aid, year, title, citations, references_count
            FROM read_parquet('/mnt/bulk/openalex/content/year=*/data.parquet') WHERE doi ILIKE '%10.48550/arxiv.%'),
ax AS (SELECT lower(id) aid, title ax_title, categories, versions[1].created v1, "journal-ref" journal_ref, doi ax_doi, license
       FROM read_parquet('/mnt/bulk/arxiv/metadata-snapshot/*.parquet'))
SELECT oa.*, ax.* EXCLUDE (aid) FROM oa LEFT JOIN ax USING (aid)) TO '/mnt/bulk/arxiv/openalex_vs_arxiv.parquet';
