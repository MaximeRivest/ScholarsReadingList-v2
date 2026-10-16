-- Lookup tables to link a parsed reference to an OpenAlex work (every work in the snapshot, any language/type).
-- tkey: the title's letters and digits, lower case, first 200; md5_number_lower: a stable 64-bit key.
SET threads=20; SET memory_limit='70GB'; SET preserve_insertion_order=false; SET temp_directory='/mnt/bulk/arxiv/match/tmp';
CREATE OR REPLACE MACRO tkey(t) AS md5_number_lower(left(regexp_replace(lower(coalesce(t,'')), '[^a-z0-9]', '', 'g'), 200));
CREATE OR REPLACE MACRO surname(a) AS regexp_replace(lower(strip_accents(regexp_extract(split_part(coalesce(a,''), ' | ', 1), '([^ ]+)$', 1))), '[^a-z]', '', 'g');
COPY (SELECT CAST(substr(id, 23) AS BIGINT) w, tkey(title) tk, year, surname(authors) au, citations
      FROM read_parquet('/mnt/bulk/openalex/content/year=*/data.parquet')
      WHERE title IS NOT NULL AND length(regexp_replace(lower(title), '[^a-z0-9]', '', 'g')) >= 12)
  TO 'title.parquet' (FORMAT parquet, COMPRESSION zstd);
COPY (SELECT lower(regexp_replace(doi, '^https?://(dx\.)?doi\.org/', '')) doi, CAST(substr(id, 23) AS BIGINT) w
      FROM read_parquet('/mnt/bulk/openalex/content/year=*/data.parquet') WHERE doi IS NOT NULL)
  TO 'doi.parquet' (FORMAT parquet, COMPRESSION zstd);
COPY (SELECT year, lower(regexp_replace(volume, '[^0-9A-Za-z]', '', 'g')) vol, lower(regexp_extract(pages, '^([0-9A-Za-z]+)', 1)) p1,
             surname(authors) au, CAST(substr(id, 23) AS BIGINT) w, citations
      FROM read_parquet('/mnt/bulk/openalex/content/year=*/data.parquet')
      WHERE volume IS NOT NULL AND pages IS NOT NULL AND year IS NOT NULL)
  TO 'volpage.parquet' (FORMAT parquet, COMPRESSION zstd);
