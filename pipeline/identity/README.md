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

1. Records carrying one ORCID are one person (anchored on the record with most papers) when their names are
   compatible: spelling, transliteration, nicknames and double surnames pass (`ncompat_orcid`), an ORCID on the
   wrong person ("Karel Pacák" / "Abhishek Jha") does not.
2. A publication authorship whose printed ORCID names another person moves to that person (or, without a record,
   joins them), if the printed name is compatible, the person is not already on the paper, and the ORCID is not one
   OpenAlex put on two incompatible names.
3. Records that are not people are flagged (organisations, placeholders, lists of names; 3,000+ papers over 60+
   years): hidden from people search, pages say so.
4. Evaluation set: every publication authorship with a printed ORCID, name block, 10 % of ORCIDs held out (test).

Results and their check by hand: see the bottom of this file (filled in by the final run).
