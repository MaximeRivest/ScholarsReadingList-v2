# Paper and author identity on top of OpenAlex (2026-10-19)

Steps 1 and 3 of `research/audit/RECOMMENDATIONS.md`. OpenAlex stays the raw layer; these are derived tables,
rebuilt per snapshot, applied by the release (rows) and the search API (pages, results). Nothing in OpenAlex's ids
is changed; every decision is reversible (`?records=one` on author pages; releases without `--identity`).

Outputs: `/mnt/fast/scholarsreadinglist/identity/openalex-2026-09-23/{versions,authorships,authors}/`.

## 1. Paper versions (`build_versions.py`, ~35 min CPU over all 476M records)

Two records are versions of one paper when, among publication types: their folded titles are equal (≥ 25 letters
and digits, ≥ 4 words), their years are at most 2 apart, and they share a surname among their first 10 personal
authors (organisations and placeholders such as "None Available" never count). Groups are connected components,
except: titles held by > 200 records (generic) are skipped; two different DOIs of one publisher and type cannot
meet (a book and each of its chapters; letters of one archive), unless one DOI extends the other ("X" and "X.v1",
"X/pdf") or the publisher is a repository or mirror; a group that still holds such a pair is a chain and is left
ungrouped. Canonical record (the version of record): not a mirror (DOI 10.60692, "Greater South Information
System"), a valid publisher DOI, published type (article > chapter/conference > report/thesis > preprint), has an
abstract, most cited, smallest W-number.

| | |
|---|---:|
| records sharing a folded title | 54.2M |
| groups (papers with 2+ records) | **14.04M** |
| records in groups / surplus records | 30.3M / **16.3M** |
| groups pairing a preprint with another version | 3.52M |
| groups with a mirror copy | 0.33M |
| canonical record without abstract, a version has one | 1.06M |
| chains left ungrouped | 155k groups (550k records) |

Precision, read by hand: 145 random groups (45 + 40 from 2020–21 during tuning, 60 from all years on the final
build): 144 right, 1 wrong (three Zenodo taxonomic treatments of one species from different papers). The 25 largest
groups were read too: one editorial printed by 158 journals, joint ACS editorials, repeated Zenodo uploads.
Recall is not measured; the rules are conservative (exact folded title, shared surname).

Effect on search (`pipeline/release/rows.py --versions`, release rows only, 2026-10-19): **4.36M extra versions leave
the semantic index** (125.20M → 120.84M searchable, −3.5 %); among a paper's versions a readable one is kept first,
so the readable count does not fall. In results (`search/srl_search/engine.py shape`, both search modes) a paper is
listed once, as its readable or canonical version, with `versions` (the others) and `versions_count`; `/works/{ref}`
gives `versions` and `version_of_record`.

Note: the versions build used the person-name rule as it was on 2026-10-18 (stricter about digits, commas and
long names than the current `names.is_person_name`); the difference only makes it group slightly fewer records.

## 3. Author identity (`extract_authorships.py`, `authors.py`, `evaluate.py`)

`extract_authorships.py`: every authorship of the snapshot as a row (1.32B rows, 33 GB, ~50 min): the record OpenAlex
chose and its ORCID, the name and ORCID the publisher printed, institutions. `authors.py` (~15 min) applies only
corrections that rest on an ORCID; names only veto:

1. Records carrying one ORCID are one person (anchored on the record with most ORCID-printed papers) when their
   names are compatible (spelling, transliteration, nicknames, double surnames pass: `ncompat_orcid`; an ORCID on the
   wrong person, "Karel Pacák" / "Abhishek Jha", does not) and the ORCID evidence covers the record: a quarter of its
   papers carry that printed ORCID, or it has at most 3, and none carries another. OpenAlex's own records can be
   mixtures of namesakes ("M. Jones", 291 papers from 1975, holding a 2000s researcher's ORCID through one paper):
   those do not join whole; their ORCID-printed papers move one by one (rule 2).
2. A publication authorship whose printed ORCID names another person moves to that person (or, without a record,
   joins them), if the printed name is compatible, the person is not already on the paper, and the ORCID is not one
   OpenAlex put on two incompatible names.
3. Records that are not people are flagged (organisations, placeholders, lists of names; 3,000+ papers over 60+
   years): hidden from people search, pages say so.
4. Evaluation set: every publication authorship with a printed ORCID, name block, 10 % of ORCIDs held out (test).

### Results (final run, 2026-10-19)

| | |
|---|---:|
| author records / with an ORCID | 127.7M / 12.2M |
| ORCIDs carried by 2+ records | 1.14M (2.95M records) |
| records merged into their person | **1.52M** |
| same ORCID, incompatible names, left alone | 84.5k |
| same ORCID, compatible name, evidence too thin to join whole | 210k |
| orphan publication authorships attached by printed ORCID | **6.74M** (of 170M orphans: 4 %) |
| publication authorships moved to the ORCID's person | **939k** (82k vetoed: that person was already on the paper) |
| records flagged as not people | 397k (2.67M publications) |

Evaluation (`scores-test.json`; 9.67M publication authorships of 1.06M held-out ORCIDs; B3 = for each authorship, how
much of its page is truly that person / how much of that person its page holds):

| | B3 precision | B3 recall | pairwise recall in name blocks |
|---|---:|---:|---:|
| OpenAlex | 0.9994 | **0.832** | 0.573 |
| overlay | 0.9996 | **0.980** | 0.971 |
| overlay, no evidence from the ORCID scored | 0.9993 | 0.832 | 0.573 |

Recall by group, OpenAlex → overlay: ORCIDs with 100+ scored authorships 0.731 → 0.961; East-Asian surnames 0.852 →
0.975; papers of the 2020s 0.758 → 0.972; 1990s 0.991 → 0.999.

How to read it. Truth comes from the same ORCIDs the overlay uses, so the overlay row measures what readers of
ORCID holders' pages will see, not a disambiguator that generalises; the third row shows the honest limit: for
authorships whose ORCID gives no evidence, nothing changes. Precision on this set cannot see an ORCID that is wrong
in both places, hence the hand checks: 40 random merges that only the same-ORCID name test allowed: ~38 right
(doubtful: 'Z. K. Jia' / 'Z. Jiao', two Miroshnikovs with different middle initials; 'Amanda Salis' / 'Amanda
Sainsbury' looks wrong and is right: Amanda Sainsbury-Salis); 25 random record-to-record moves: 25 right; 40 random
records flagged not a person: 36 right before URLs were stripped from catalogue names (the rest: catalogue names
with Library of Congress links, 'Brigitte Bureau'). The rules were tuned in four rounds of such reading (strict
names: 328k conflicts, mostly transliterations; then too lenient: 'J. W. Chen' / 'Yi-Chun Chen', '沈志强' / 'Paul
Tiede'; then given names required to agree; then the ORCID evidence required to cover a record before it joins
whole, after a page of the ORCID 0000-0002-9951-4583 'Matthew Jones' showed papers from 1960). The largest record of
that ORCID ('M. Jones', 1,284 papers from 1974) is itself OpenAlex's mixture: no ORCID rule can split it.

What this does not do: authors without ORCIDs (most of them before 2015, and most splits of historical figures) are
untouched; that is the disambiguation model's job (RECOMMENDATIONS.md step 4), and `eval.parquet` (87.2M training,
9.67M test authorships, name blocks) is the set to train and measure it on.
