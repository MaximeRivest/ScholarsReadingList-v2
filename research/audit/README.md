# Audit of the OpenAlex-based paper database (2026-10-18)

Read-only audit of the data behind Science Made Readable: the raw OpenAlex works snapshot
(2026-09-23, 476.2M records, `/mnt/bulk/openalex/works/`), the citewalk author index built from it
(`/mnt/fast/citewalk/20261007T003120Z-50ce3865/`), the site's corpus (`corpus-2026-10-05/works.sqlite`,
203.5M works with an English abstract) and the live local search API (127.0.0.1:8810). Nothing was
changed. CPU only, `nice 10`, no GPU.

Scripts are numbered in this folder; every number below is in a `results_*.json` next to them.
Scopes used throughout:

- **all**: every record in the snapshot (includes datasets, paratext, the "xpac" expansion collection).
- **publications**: the site's scope (`citewalk/science/scope.py` types, not paratext).
- **site corpus**: publications with `language = 'en'` and an abstract: what semantic search, maps and
  reading lists actually see.

## 1. Sample

`01_sample.py`: 24 parquet files chosen with probability proportional to size, 1-in-7 of their rows by
hash of the id: **1,022,009 works** (0.21 % of the snapshot), 3,728,592 authorships. Because the snapshot
is partitioned by `updated_date` and most records were re-touched on 2026-09-22, this is close to a
uniform sample; the only visible skew is towards recently updated records (which is what the site serves).

Composition: 874,792 publications (85.6 %), of which 100,510 are xpac and 169,810 carry a W-number
≥ 6,000,000,000 (records OpenAlex created in 2025–26). Site corpus: 415,135 (40.6 % of the sample).

## 2. Authors (`02_authors.py`, `results_authors.json`, `results_author_index.json`, `results_merge_surname.json`)

### 2.1 Authorships without any author record

| scope | authorships | without `author.id` |
|---|---:|---:|
| publications | 3,238,859 | **14.3 %** |
| site corpus | 1,854,782 | **16.5 %** |
| legacy records (W < 6e9, not xpac) | 2,657,526 | 2.4 % |
| new W-numbers, not xpac | 195,288 | 21.0 % |
| xpac | 386,045 | 92.7 % |

By year of publication the share rises from 4–6 % (1960–2000) to **22.7 % for 2020–24**. By index:
Crossref-only works 2.4 %, DataCite-only 87 %, `indexed_in = []` 31 %. So one authorship in six on the
site has no person behind it: it cannot appear on any author page, in Experts lists, or in co-author
clouds. The cause is OpenAlex not yet running disambiguation on records it added in 2025–26, not name
ambiguity.

### 2.2 ORCID coverage (the only ground truth available at scale)

Publisher-supplied ORCID (`raw_orcid`) on 15.3 % of publication authorships (27.8 % for 2020–24);
the matched author record carries an ORCID on 51.8 % of authorships (59.7 % in the site corpus, because
prolific authors are more often ORCID-holders).

### 2.3 Splits (one person, several records)

Among 39,943 publisher ORCIDs seen on ≥ 2 authorships in the sample, **4.05 % point to more than one
OpenAlex author record** (2,036 surplus records). The rate grows with productivity:

| authorships of that ORCID in the 1M sample | ORCIDs | split |
|---|---:|---:|
| 2 | 28,017 | 3.1 % |
| 3–4 | 8,300 | 4.8 % |
| 5–9 | 2,977 | 3.8 % |
| 10+ (≈ 4,700+ works in the full data) | 649 | **36.1 %** |

Even using OpenAlex's own `author.orcid`, 3.7 % of ORCIDs (10,778 of 288,435) sit on two or more author
records in this 1M sample alone — e.g. ORCID 0000-0002-9951-4583 "Matthew Jones" has 8 records. These
are splits OpenAlex could close mechanically.

Direct contradictions: on 469,793 authorships where both the publisher ORCID and the record ORCID are
present, **0.36 % differ** (1,695: the authorship was attached to a different person's record), and on
2.9 % the record has no ORCID although the publisher gave one.

### 2.4 Merges (several people in one record)

- By ORCID: among 39,565 records with ≥ 2 ORCID-bearing authorships, **0.20 %** carry two different
  ORCIDs (0 % among the most prolific).
- By surname (`results_merge_surname.json`, surname taken before the comma or as the last token):
  2.77 % of records seen twice show two surnames; 0.93 % after discounting surname/given-name inversions.
  Reading 30 of those: most are still Chinese-name order problems; about 4 in 30 are real merges
  ("Ruiqing Liu" + "Xiujing Lin"; "Alastair D Hay" + "Alastair Hay Leyland"). Estimated true merge rate
  ≈ 0.1–0.2 % of records, 6.5 % of prolific records (10+ authorships in the sample) show a second surname.

So **splitting is 20× more common than merging**, and both concentrate on prolific authors — exactly the
authors whose pages people open.

### 2.5 The whole author index (`results_author_index.json`, from `a_works/a_year0/a_year1.npy`)

127.7M author records; 124.0M have ≥ 1 publication. 60.0 % have exactly one, 90 % of authorships belong
to the 40 % with ≥ 2. Career spans: 1.42 % of records span ≥ 50 years, **626,400 records (0.51 %) span
≥ 70 years**, 246,632 ≥ 100 years. Among the 370,249 records with ≥ 200 works, **19 % span ≥ 60 years**
(reprints, merges, or both; e.g. "George Smith" 627 works 1828–2026, "John D. Scott" 420 works 1845–2026).

The largest records are not people: "Anonymous" 102k works, "叶兴阳双语音标有声读物" 95k, ":unav" 87k
(and three more ":unav" records), "Massachusetts." 60k, "Carey Hand Funeral Home" 28k, "CERTIFICATION
EXAM" 18k, "nasa" 14k. Only 8,361 records have ≥ 1,000 works and 297 ≥ 3,000; a large share of those
are organisations, placeholders or merges.

### 2.6 Names on paper vs name on record

11.5 % of authorships have a raw name whose first initial *and* last token both differ from the record's
display name; sampling shows these are almost all format differences ("Yanagi, Hidekatsu" vs "Hidekatsu
Yanagi", "Ersanlı, C.C." vs "Cem Cüneyt Ersanlı"). A name-matcher must normalise "Surname, Given",
initials, diacritics and East-Asian order before anything else.

## 3. Institutions (`02_authors.py`, `08_institutions.py`)

- Publication authorships with a raw affiliation string: 65.6 %; with a linked institution: 60.9 %;
  **with a string but no institution: 4.7 %** (11 % of strings in 1990–99, 5.4 % in 2020–24, 3.9 % in 2025).
  Unmatched strings are departments, field stations, small companies, cities ("New Delhi"), ships.
- Of 19,800 single-institution authorships, 5.0 % have no distinctive word in common between the string and
  the institution's name. Reading 40 of them: ~34 are translations/acronyms (ENEA, CSIRO, "Pädagogische
  Hochschule Luzern" → correct), ~6 are wrong links: "MDACC, Houston" → *Mississippi Delta Community
  College*; "SVTC, Austin" → *SVT Associates*; "MicroSilicon Inc" → *MicroVision*; "Semenenko Institute of
  Geochemistry" → *Palladin Institute of Biochemistry*. Estimated **≈ 0.7 % of linked affiliations point to
  the wrong organisation**, plus the 4.7 % unlinked; 34 % of authorships have no affiliation at all.
- Literature agrees: missing-institution rates above 60 % for journal articles in early OpenAlex [7];
  systematically lower institutional counts than Scopus for German institutions [8].

## 4. Topics and fields (`04_topics_vs_experts.py`, `05_works.py`)

### 4.1 Against the 200-article expert gold standard (Rivest et al. 2021; the set Comparison uses)

181 of the 200 articles have DOIs; **only 132 are in the site corpus** (see §5). For those, OpenAlex's
primary topic, read up to its subfield and compared with the experts' modal first choice through a
hand-written equivalence map (in the script; lenient, one-to-many):

| journals | n | OpenAlex subfield = expert modal choice | in any expert's list | coarse field right |
|---|---:|---:|---:|---:|
| disciplinary | 53 | **47 %** | 74 % | 79 % |
| multidisciplinary | 79 | **24 %** | 68 % | 65 % |
| all | 132 | 33 % | 70 % | 70 % |

Same test, same scoring idea, from the Comparison conversation: one human expert 63 % / 57 %, Jev 64–68 % /
61–65 %, our 2B 62–68 % / 61–62 %. OpenAlex's own label is therefore **roughly 20–35 points below a human
or our own classifier**, and at the coarse field level it is wrong for about one paper in four to three.
Glaring misses: quantum-optics papers labelled *Artificial Intelligence*; a renal-toxicity paper labelled
*Plant Science*; a health-systems paper labelled *Finance*. Caveat: the two taxonomies differ, so ± 5
points on the field numbers; the subfield numbers favour OpenAlex because the map is lenient.

### 4.2 Structural problems in the topic assignment (1M sample)

- **A garbage-attractor topic.** "Military Technology and Strategies" (subfield Aerospace Engineering) is the
  single most frequent primary topic among publications: **40,233 of 874,792 = 4.6 %**, all with score 0.02.
  It takes 29.9 % of non-English publications without an abstract and 7.9 % of non-English ones with an
  abstract; in English it is 0.04–0.05 %. Titles sent there: Japanese veterinary abstracts, Ukrainian law,
  Arabic paediatrics, a Chinese poetry paper. "Magnetic confinement fusion research" (0.8 %) collects
  Spanish education and art papers with scores 0.9+. The site's English-only corpus is mostly spared
  (0.04 %), but the map's counts per subfield and any multilingual expansion are not.
- **Low confidence is common.** Primary-topic score < 0.5 for 20 % of publications with an abstract, 30 %
  without; **12.5 % in the site corpus**. 68 % of publications have their three topics spread over more than
  one field; for 33 % the second topic scores within 10 % of the first *and* lies in another field.
- Missing topic: 3.1 % of publications without abstract, 0.6 % with; 0.31 % in the site corpus; 5.8 % for
  new non-xpac records.
- Keywords present on 98 %, concepts 98 %, MeSH 16 %.
- Literature: the first independent evaluation of OpenAlex Topics (837 hand-coded works, 2026) finds label
  precision between **54 % and 83 % depending on field**, with scope-inflated labels, single-topic subfields
  and cross-domain misplacements [1]. Publication-level clustering (CWTS-style, which OpenAlex's topics
  descend from) is known to shift rankings materially versus journal-based schemes [9].

## 5. Abstracts and corpus coverage (`05_works.py`, `03_missing_dois.py`)

- Articles with an abstract: **61.5 %**; by publisher DOI prefix: Elsevier 10.1016 **14.1 %** (67,651
  articles in the sample, the biggest prefix), Springer 10.1007 **11.9 %**, Nature 10.1038 45 %, JSTOR 33.5 %;
  Wiley 84–85 %, T&F 92 %, OUP 92 %, IEEE 95 %, ACS/RSC/APS 97–99 %, MDPI 100 %.
- By year (articles+reviews): 25 % in 1950, 50 % in 1995, 66 % in 2010, 83 % in 2020–24.
- By language of the record: en 64.5 %, ja 10.7 %, de 39 %, fr 49 %.
- Consequence measured on the expert set: **49 of 181 DOIs (27 %) are not in the site corpus; 44 of the 49
  exist in OpenAlex with no abstract** (Tetrahedron Letters, Virology, J. Power Sources, Nature, Springer
  proceedings…). The semantic index silently lacks about a quarter of mainstream Scopus-indexed articles,
  skewed against Elsevier and Springer.

## 6. Duplicates and mirrors (`06_preprints.py`, `07_gsis.py`, `results_doi_dups.json`, `results_title_dups.json`)

- Same DOI on two W-numbers: 477,764 surplus records over 318.4M DOIs (0.15 %): small.
- **"Greater South Information System" (DOI prefix 10.60692)** is the third-largest publication source in the
  snapshot: 21,728 articles in the sample = **3.3 % of all articles, 4.9 % of the site corpus**, 100 %
  with abstract, 0.02 citations on average, 0 % with references, years 1937–2024. 120 random ones searched
  by exact title: **62 % duplicate a paper already in the English corpus** (twins at BMC, Nature, Elsevier,
  PLOS, Frontiers…). The other 38 % were not found by that search; whether they are twins of papers
  without an abstract was not established here (an earlier draft gave the Vaccine 2023 paper
  10.1016/j.vaccine.2023.02.054 as such a case: wrong, it is in the corpus with its abstract). See
  `../../pipeline/identity/` for the full measurement.
- Preprints are 3.3 % of publications; of 150 random 2016–22 preprints, **39 % have a non-preprint record
  with the identical title** (lower bound: titles change). OpenAlex does not link them.
- Within the 1M sample alone, 1,987 pairs of long titles (>50 chars) share title and year: 1,871 with two
  distinct DOIs — mostly GSIS-vs-original and journal-vs-arXiv. Scaled to the corpus this is a few hundred
  thousand visible pairs beyond GSIS and preprints.
- Release 20261009 already dropped 3.6M byte-identical title+abstract texts (4.7 % of 76M publications).

## 7. Dates, venues, languages, retractions, references (`05_works.py`, `05b_tail.py`)

- **Dates**: 3.0 % of publications have no year/date; 30.6 % of dates are 1 January (year precision only);
  year vs date never disagree; 0.08 % have a publication year after the record's creation year.
- **Venues**: articles without a source 7.3 % (2.4 % have a raw venue name that was not matched);
  conference papers **39 %**, book chapters **36 %**, dissertations 16 %. Unmatched names are IEEE/SPE
  proceedings, book titles, Korean journals, repository URLs. Journal articles lack volume 7.8 %, first page
  11.8 %. Second- and third-largest sources are "PubMed" (as a source) and GSIS; "CyberLeninka",
  "Institutional Digital Library (Federal Senate)" and "Medical Entomology and Zoology" (6,610 "articles",
  0.8 % with abstract) are catalogue noise.
- **Languages**: 4.9 % of publications have no language. Of 'en'-tagged titles, 0.28 % are in a non-Latin
  script; a stop-word check on 19,837 'en'-tagged titles with an abstract finds **1.0 % plainly not
  English** (Indonesian, Spanish, Portuguese, Polish). Literature: OpenAlex language detection is weakest
  for non-English and multilingual records [10].
- **Retractions**: 0.035 % of publications are flagged; 97 further titles begin with
  RETRACTED/WITHDRAWN without the flag (**≈ 24 % of the retracted-looking set is unflagged**; 40 of them in
  the site corpus against 221 flagged). The API already hides "[RETRACTED]"/"Retraction Note:" titles, not
  "WITHDRAWN:". Literature confirms the single boolean loses Crossref's update types [11].
- **References**: 47.5 % of articles since 2000 have zero references in OpenAlex (34.5 % in the site
  corpus; 38 % for 2020–24, 55–63 % before 2005); book chapters 64 %, preprints 62 %. Reference coverage
  in OpenAlex is known to trail Scopus and WoS [12]; the Comparison conversation already measured 60 % of
  pre-2022 arXiv papers without reference lists and started filling them.
- Open access 43.4 %; full text known 26.9 %.
- Document types: OpenAlex promotes editorial matter to "article" [13]; 2.9 % of publications have zero
  authors.

## 8. What the field does (literature, via the Scholars API)

- **Author disambiguation.** S2AND (Semantic Scholar/AI2): a pairwise model over name, affiliation,
  co-authors, venues, titles, embeddings, then agglomerative clustering; trained on the union of eight
  datasets it cut the production error by more than half (B³ F1) and generalised to unseen datasets [2].
  WhoIsWho (AMiner): 1M+ papers, interactive annotation, a leaderboard, and a deployed model doing daily
  arXiv assignment; names with 100+ homonyms are the hard case [3]. ORCID as an authority source gives
  labelled data at the scale of millions of mentions, with the caveat that ORCID-holders are not
  representative of all mentions [4]. LAGOS-AND is a large ORCID-based gold standard [5]. A 2025 benchmark
  built with search-augmented LLM annotation measures OpenAlex and Clarivate author IDs by precision and
  recall per country and discipline [6].
- **Subject classification.** Expert agreement itself is only ~60 % at subfield level (Rivest et al.),
  which bounds what any classifier can show; Comparison's 2B already sits at that bound. The weak point of
  OpenAlex Topics is the LLM-labelled citation clusters and their deep-learning assignment [1].
- **Metadata quality.** Missing institutions [7][8], document types [13], language [10], retractions [11],
  references [12].

## References

- [1] https://doi.org/10.5281/zenodo.22656225 — Assessing the quality of OpenAlex Topics as a classification system, 2026
- [2] https://doi.org/10.1109/jcdl52503.2021.00029 — Subramanian et al. 2021, S2AND, JCDL
- [3] https://doi.org/10.1145/3580305.3599930 — Chen et al. 2023, WhoIsWho, KDD
- [4] https://doi.org/10.1007/s11192-020-03826-6 — Kim 2021, ORCID-linked labeled data, Scientometrics
- [5] https://doi.org/10.1002/asi.24720 — Zhang et al. 2022, LAGOS-AND, JASIST
- [6] https://doi.org/10.48550/arxiv.2502.11610 — Accuracy assessment of OpenAlex and Clarivate scholar IDs, 2025
- [7] https://doi.org/10.5281/zenodo.8305858 — Missing institutions in OpenAlex, 2023
- [8] https://doi.org/10.48550/arxiv.2605.01337 — OpenAlex vs Scopus coverage of German institutions, 2026
- [9] https://doi.org/10.1002/asi.24017 — Perianes-Rodriguez & Ruiz-Castillo 2018, classification systems and the Leiden Ranking, JASIST
- [10] https://doi.org/10.1002/asi.24979 — Céspedes et al. 2025, linguistic coverage of OpenAlex, JASIST
- [11] https://doi.org/10.34657/39383 — (Non-)retracted academic papers in OpenAlex, 2024
- [12] https://doi.org/10.48550/arxiv.2401.16359 — Culbert et al. 2024, reference coverage of OpenAlex vs WoS and Scopus
- [13] https://doi.org/10.1007/s11192-025-05524-7 — a classifier for research journal publications in OpenAlex, Scientometrics 2026

Also used, not cited above: 10.1371/journal.pone.0251493 (Rivest et al. 2021, the expert gold standard).
