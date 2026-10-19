# What to do about the database, in order (2026-10-18)

Companion to `README.md` (the measurements). Costs are my estimates: days of one person's work plus
compute on the two RTX 3090s or at ~$0.25/h rented. Nothing here spends money without a decision.

## 0. Principles a scholarly-data expert would insist on

- **Keep OpenAlex as the raw layer and build our own resolved layer on top** (persons, papers, venues as
  clusters of OpenAlex records), with every decision reversible and re-appliable on the next snapshot.
  Never edit OpenAlex ids in place.
- **Measure before and after with the same held-out sets** (the ORCID pairs, the 200 expert articles, the
  1M sample). Make this audit a script that reruns on every snapshot.
- **Fix mechanical errors before training anything.** Most of what is wrong today is not an AI problem.

## 1. Paper identity: remove the mirror copies and link preprints — start here

Measured: 4.9 % of the site corpus is "Greater South Information System" copies, 62 % of which duplicate
a record already in the corpus; 39 % of preprints have a published twin under the identical title; the
semantic index already hides exact text duplicates but not these.

Do: cluster works by (normalised title, year ± 1, first-author surname) over the 203M-row corpus with
DuckDB; inside a cluster pick the canonical record (real publisher DOI > repository > GSIS), carry the
abstract from a copy to a canonical that has none, keep the cluster as `paper_versions`. Search, author
pages and citation walk show one entry with "also as preprint / N versions". Guard: titles > 40 chars
only, author overlap required when years differ.

Cost: 2–3 days; CPU hours; no GPU. Gain: cleaner results for every query; correct author counts; and,
where a copy carries an abstract its original lacks, a free abstract (how many: measured in pipeline/identity).

## 2. Abstract coverage: the quiet 27 % hole

Measured: Elsevier articles have abstracts 14 % of the time, Springer 12 %; 44 of 181 expert-set articles
exist in OpenAlex with no abstract and are therefore invisible to semantic search, maps and lists.

Do, cheapest first: (a) the GSIS copies from §1; (b) a 1,000-DOI probe against PubMed and Europe PMC to see
how many abstract-less biomedical Elsevier/Springer papers they cover, then bulk-join by PMID (free,
licence-clean); (c) OpenAIRE / CORE for repository copies; (d) the PMC mirror we already hold. Semantic
Scholar's abstracts are the largest source but their dataset licence is for research: ask before using.

Cost: probe half a day; PubMed join 1–2 days, ~40 GB. Gain: likely several million more papers in the
semantic index, concentrated on exactly the mainstream journals readers expect.

## 3. Authors, step one: mechanical fixes and an evaluation set (no model yet)

Measured: 16.5 % of authorships in the site corpus have no author record at all (OpenAlex has not
disambiguated its 2025–26 additions); 3.7 % of ORCIDs seen twice sit on two or more OpenAlex author
records; 0.36 % of authorships contradict the publisher's ORCID; splits run 4 % overall and 36 % for
prolific authors; merges are rare (~0.2 %); the biggest "authors" are "Anonymous", ":unav" and
organisations; 626k records span 70+ years.

Do:
1. Merge author records that share an ORCID (certain). Move authorships whose publisher ORCID names
   another record (certain). Attach orphan authorships to a record when exactly one same-name record
   shares a co-author, institution or ORCID on that paper; otherwise give them a provisional person id so
   they appear somewhere.
2. Exclude non-person records from people pages: names in a stop-list (:unav, Anonymous, et al.), names
   with organisation words, records with > 3,000 works and > 60-year spans.
3. Normalise names once, properly: "Surname, Given" order, initials, diacritics, East-Asian order
   (11.5 % of raw names differ from the record name only in format).
4. Build the evaluation set: positives = same publisher ORCID, hard negatives = same normalised name,
   different ORCID; several million pairs for free [Kim 2021; LAGOS-AND]. Add 300–500 hand-checked pairs for
   famous and historical names where ORCID does not exist (the records readers tick together in the
   Author view, if logged with consent; Wikidata). Report B³ / pairwise F1 by name origin, productivity and
   era, as S2AND does, so the model is not tuned on Western names only.

Cost: 3–4 days; CPU only. Gain: closes the certain splits immediately, makes every authorship visible,
and gives the number that tells whether step 4 is working.

## 4. Authors, step two: the disambiguation model

What the best groups do: a pairwise scorer plus agglomerative clustering inside name blocks (S2AND, which
halved Semantic Scholar's production error; WhoIsWho's deployed baseline at AMiner). Features: name
compatibility, co-author overlap, affiliations, venues, year gap, ORCID/email, and **paper-content
similarity** — we have an unusual asset here: 203M paper embeddings already computed, which S2AND had to
make with SPECTER.

Do: gradient-boosted pairwise model (LightGBM) first — trains in minutes, explainable, strong on these
features; cluster within blocks (surname + first initial); for the largest blocks (Wei Wang-size, 10⁵
mentions) generate candidates by embedding neighbours instead of all pairs. Then, where Maxime wants the
small Qwen: use it as a **second-stage judge on the uncertain middle band and on famous/historical cases**,
not on all 857M authorships (a 0.8B model at ~0.03 s per pair cannot read 10⁹ pairs; a tree model can).
Train it on the ORCID pairs plus the ticked-records examples; evaluate on the held-out split from §3.

Cost: ~1–2 weeks; tree model CPU-only; Qwen judge a few GPU-hours to fine-tune and ~10–50 GPU-hours to
run on the uncertain band. Gain: the 36 % split rate for prolific authors is the target; expect to get well
under 10 % on the ORCID test, and to make the Author view's "other records with this name" mostly empty.
Risk: merging two real people is worse than leaving a split; set the threshold for precision and keep the
UI's un-merge.

## 5. Fields and topics: replace, don't repair, OpenAlex topics

Measured: OpenAlex's primary topic agrees with the expert's subfield 47 % (disciplinary journals) and
24 % (multidisciplinary) of the time, 20–35 points below one human expert and below Comparison's 2B;
the coarse field is wrong for ~30 %; 12.5 % of the site corpus has a topic score under 0.5; one junk topic
("Military Technology and Strategies") absorbs 4.6 % of all publications, nearly all non-English; an
independent 2026 study finds 54–83 % label precision by field.

Comparison is already training the classifier; do not duplicate that. What to add:
1. **Distil into the embeddings.** Label ~2M papers with the 2B (≈ 35–70 GPU-hours, or ~$10–20 rented),
   train a small head (logistic regression or a 2-layer MLP) on the existing qwen3-embed vectors, classify
   all 125M embedded papers on CPU in hours. Measure the loss against the 2B on a held-out 20k; I expect a
   few points. Running the 2B itself over everything is ~7,000 GPU-hours (~$1,800) today, ~1,400 after the
   list-first retrain: the head is the way to cover the whole corpus now, the 2B the way to label the
   long tail and new daily papers.
2. Until then: treat primary topic as unknown when score < 0.5 or topic is the attractor; label the map
   from the subfield distribution of embedding neighbours, not from one record's topic.
3. Keep OpenAlex topics only as *one feature* for the maps' labels; our 176-subfield layer becomes the
   field of record for the map, counts and reading-list filters.

Cost: 2–3 days plus the GPU hours above. Gain: a consistent field layer for the whole corpus at roughly
expert level, and the map's regions stop lying about where papers belong.

## 6. Quick fixes (one day in total)

- **Retractions**: also hide titles starting "WITHDRAWN"; import the Retraction Watch database (free CSV via
  Crossref) and mark by DOI. Measured: ~24 % of retracted-looking papers are unflagged.
- **Language**: run a fast language id (fastText lid.176 or lingua) over title+abstract of the 'en' corpus;
  ~1 % is not English and is polluting embeddings and lists.
- **Dates**: 30.6 % of dates are 1 January; show the year only in that case.
- **Zero-author records** (2.9 %) and non-research "articles": hide from author counts; consider the open
  document-type classifier of Scientometrics 2026.

## 7. References, venues, institutions (later, in that order)

- **References**: 34.5 % of site-corpus articles since 2000 have none. Comparison is filling arXiv; next
  parse reference lists from the 3.9M PMC full texts we already mirror and link by DOI/PMID (1–2 days,
  CPU). This feeds the citation walk and the maps directly.
- **Venues**: 39 % of conference papers and 36 % of chapters have no source; match raw venue names to
  sources by normalised name, and DBLP for computer science (open). 2 days.
- **Institutions**: 4.7 % of affiliation strings unlinked, ~0.7 % linked to the wrong organisation
  (MDACC → a community college), 34 % of authorships with no affiliation. Use the ROR affiliation matcher
  or AI2's S2AFF on the unlinked strings; low priority for the site, matters for institution pages. 3 days.

## Suggested order and why

1 (dedupe + abstract rescue) and 6 (quick fixes) first: largest visible gain per day, no GPU, no risk.
Then 3 (author mechanics + eval set) because it is certain and prerequisite to any model. Then 5.1 (the
embedding head) in coordination with Comparison, because the GPUs are busy and the head needs almost none.
Then 4 (the author model) with Maxime involved, as he asked. Then 2b–d, 7.
