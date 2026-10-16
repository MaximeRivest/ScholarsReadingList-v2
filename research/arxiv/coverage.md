# arXiv in OpenAlex: what is there, what is missing, where the rest can come from

Measured 2026-10-16 on lambda (OpenAlex snapshot 2026-09-23 in `/mnt/bulk/openalex`, arXiv's metadata snapshot of
2026-10-12 in `/mnt/bulk/arxiv/metadata-snapshot`). Join of the two: `/mnt/bulk/arxiv/openalex_vs_arxiv.parquet`
(one row per OpenAlex work with an arXiv DOI, with arXiv's own title, categories, first-version date, licence).

## What OpenAlex has

| | |
|---|---|
| arXiv papers (arXiv's snapshot) | 3,205,432 |
| OpenAlex works with an arXiv DOI (10.48550/arxiv.*) | 2,925,997 (91%); the rest are mostly merged into a journal version's record (not measured) |
| Abstract present | ~99.9% every year |
| Reference list present (referenced_works) | ~20-42% for 1991-2021, then **3.5% (2022), 8.8% (2023), 0.5% (2024), 0.05% (2025), 0.03% (2026)** |
| Median references when present (2019) | ~26 (Semantic Scholar: ~40-45 for the same kind of papers) |
| Title differs from arXiv's (word overlap < 0.3) | 1,601 works (0.055%), rising: 0.01% of 2017 papers, 0.23% of 2026 |
| Full text cached by OpenAlex (has_content.grobid_xml) | 12% (2021), 15% (2022), **85% (2023), 82% (2024)**, 25% (2025), 14% (2026) |

Reference lists: the cliff at 2022 matches the end of Microsoft Academic Graph (which parsed PDFs) at the end of 2021;
since then OpenAlex takes references from Crossref/PubMed records, and arXiv's DOIs are DataCite DOIs with no
reference lists. OpenAlex says it also extracts references from open-access PDFs, and it holds Grobid XML for ~83% of
2023-24 arXiv papers, but those references are not in referenced_works (the cause is our inference, not documented).
Queries: `measure.sql` beside this file; per-year results in `/mnt/bulk/arxiv/coverage_by_year.csv` and `content_by_year.csv`.

Titles: some mismatches are honest (the authors renamed the paper in a later version), but the most cited are
hijackings: a junk record (often a Zenodo upload carrying the arXiv DOI) has replaced the title, and sometimes the
abstract, of a famous paper. Examples: 2201.11903 Chain-of-Thought ("BNAI, NO-TOKEN, and MIND-UNITY…"), 1807.06209
Planck 2018 VI ("Dark Matter as a Topological Vacuum Condensate"), 2005.11401 RAG, 2307.09288 Llama 2, math/0211159
Perelman's Ricci flow, 2106.09685 LoRA, 2304.02643 Segment Anything, 2212.08073 Constitutional AI, 2210.03629 ReAct,
2205.01833 OpenAlex itself ("OpenAlex Snapshot"). The 1,601 carry 30,651 citations (0.48% of arXiv's). Abstracts:
hijacked for Chain-of-Thought, RAG, Planck, Segment Anything; intact for LoRA, Constitutional AI. Our search vectors
and map positions for these papers are computed from the junk text.

## What can come in bulk, without parsing

| Source | What | Access | Licence / terms |
|---|---|---|---|
| arXiv metadata (OAI-PMH, daily; weekly Kaggle snapshot, mirrored on Hugging Face as librarian-bots/arxiv-metadata-snapshot, 2.7 GB parquet) | id, title, authors, abstract, categories, versions + dates, DOI and journal-ref of the published version, licence, comments | free | metadata CC0 |
| arXiv PDFs, Kaggle's Google Cloud bucket `gs://arxiv-dataset/arxiv/arxiv/pdf/YYMM/` | every PDF, one file per version, current to 2026-09 | free, anonymous HTTP | per paper: most under arXiv's non-exclusive licence: we may index and extract, must link to arXiv for the paper |
| arXiv PDFs + LaTeX source, S3 `s3://arxiv/` (requester pays) | ~9.2 TB (April 2025), +~100 GB/month; source only here | AWS account; transfer ~$0.09/GB (~$500 for source alone, estimate) | as above |
| Semantic Scholar datasets (weekly) | citations: 2.4B links with the citing sentence, intent and "influential" flag; papers with arXiv ids and DOIs | free API key (form filled 2026-10-09; no key on lambda yet) | citation graph ODC-BY (credit Semantic Scholar); some other datasets CC BY-NC: check per dataset |
| OpenAlex content API | Grobid XML / PDF per work | $10 per 1,000 files; free $1/day ≈ 100 files | |

Semantic Scholar sample (80 random arXiv papers, 2026-10-16): references for 58 of 60 papers from 2023-25 (median 40)
and 19 of 20 from 2018-19; OpenAlex had them for 3 of the 60 and 10 of the 20.

## What needs parsing

- **References with our own extraction**: from the LaTeX source for ~88% of papers (53 of 60 sampled had source;
  the .bbl/.bib gives clean, structured entries, often with arXiv ids and DOIs), from the PDF (Grobid) for the
  PDF-only rest. Then matching each entry to an OpenAlex work (DOI / arXiv id when present, else title + year +
  first author against our 203M-work index).
- **Full text** (sections, figures, equations), **citation contexts** (the sentence around each citation; Semantic
  Scholar has these for what it parses).
- **Metadata does not need parsing**: titles, abstracts, authors, dates, published-version DOIs are all in arXiv's
  metadata.

## Next steps (proposed, not started)

1. Titles and abstracts from arXiv's metadata for every work with an arXiv DOI whose text differs (1,601), re-embed
   them (minutes). Fixes Chain-of-Thought, Llama 2, LoRA… on the site, in search and on the map.
2. References from Semantic Scholar's citation dataset where OpenAlex has none (needs the key): the cheapest large win.
3. Our own extraction from LaTeX source as the check on (2) and for what it misses: start with one month of 2025
   from the free PDFs + a source sample, measure against Semantic Scholar before scaling.
