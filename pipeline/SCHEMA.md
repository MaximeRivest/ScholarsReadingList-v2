# Data model (schema `tidy-rows-v3`, adapter `openalex-import-v3`)

Executable contracts: `pipeline/tidy/contracts.py`. Readable column list:
[CONTRACTS.md](../research/openalex/tidy-row-example/CONTRACTS.md). How to build
and read the dataset: [README.md](README.md).

## The pattern

**Unit tables name the things. Link tables name the two things they connect.**

- One row per defined unit; one scalar value per cell (no lists, no structs).
- Four unit tables: `content`, `signatures`, `entities`, `classifications`.
- Links are named `links_<left table>_<right table>` with the exact unit-table
  names; endpoint columns name their units (`signature_id`, `entity_id`, …).
  Same-table links use `from_<unit>_id` / `to_<unit>_id`.
- Every link carries `relation`, `method`, and its evidence: `source_record`
  (the work it came from), `source_revision`, `source_field` (exact JSON path).
  `score`/`score_kind` only when the source gives one — never an invented 1.0.

```text
content ─ links_content_signatures ─ signatures ─ links_signatures_entities ─ entities
content ─ links_content_classifications ─ classifications
content ─ links_content_entities ─ entities                    (funded_by)
content ─ links_content_content ─ content                      (cites)
classifications ─ links_classifications_classifications ─ classifications   (broader)
classifications ─ links_classifications_entities ─ entities   (funded_by, host_organization)
```

Metrics are computed over **distinct content**: an entity reaches content via its
signatures, a classification via its content links. Several signatures (an
author's two affiliations, three co-authors at one university) never multiply a
paper's contribution.

## Unit tables

| Table | One row per | Main columns |
|---|---|---|
| `content` | current version of a work | title, doi, type, language, date, year, abstract, citations, references_count, `is_complete`, is_retracted, volume, issue, first_page, last_page, is_oa, oa_status, oa_url, fwci, is_paratext, pmid, pmcid, mag |
| `signatures` | author–affiliation pair as printed on one work | author_text, affiliation_text (raw, unchanged), author_order, affiliation_order, author_position, is_corresponding, raw_orcid |
| `entities` | person or organization (source ID) | entity_type (`person`/`organization`), name, orcid, ror, country_code, organization_type |
| `classifications` | named grouping of content | classification_type, scheme, name, notation |

`is_complete` = English, has an abstract, has at least one reference (any type;
filter `type == 'article'` separately). `fwci` that is not finite is stored as
unknown and counted.

### Signatures

An author with two affiliations has **two** signatures; an author with none has
one with empty affiliation fields. Pairings are exactly OpenAlex's
per-affiliation-string mapping; aggregate institution lists are never spread
over every string. Same text on two works = two signatures.

### Entities and classifications

| Kind | Source | Notes |
|---|---|---|
| person | `authorships[].author` | upstream author cluster, not re-disambiguated |
| organization | institutions, affiliation mappings, funders, award funders, source hosts | institutions (I…), funders (F…), publishers (P…) keep their own IDs; no merging by name |
| topic → subfield → field → domain | `topics`, `primary_topic` | hierarchy as `broader` links |
| keyword | `keywords` | |
| journal, repository, conference, … | location sources | `notation` = ISSN-L; host organization link |
| mesh_descriptor, mesh_supplementary_concept, mesh_heading | `mesh` | `D…`, `C…`, descriptor/qualifier pairs; duplicates collapsed; `is_primary` = major topic (true if any entry is, unknown if none true and some unknown) |
| sdg | `sustainable_development_goals` | `unsdg:N` |
| award | `awards` | `notation` = funder's award number; `funded_by` link to the funder |

Deferred: OpenAlex's deprecated **concepts** (~12 per work, ~6 B rows) and
organization lineage. Both remain in the raw files; counts are in run notes.

### Location links

`links_content_classifications` with `published_in` only for recognized venues
(journal, conference, book series) whose flags confirm the published version;
everything else is `hosted_by`. Each copy keeps its own URL, PDF, version,
licence and open-access flag.

## One row per key across the dataset

`entities`, `classifications`, `links_classifications_classifications` and
`links_classifications_entities` describe vocabulary, not one work. Each has
one row per key; among all observations in current works the representative is
the highest by: observation time (work's `updated_date`), then completeness
(populated properties), then work number, then source field. The same order is
used inside a partition and across partitions, so separate extraction followed
by reduction equals one extraction (tested). Vocabulary link IDs depend only on
their endpoints, relation and method.

## Versions and identifiers

- One current version per work: latest `updated_date`; ties go to the later
  source file, then later row. Superseded rows are excluded before extraction.
- IDs carry their origin: `openalex:W…`, `openalex:A…`, `mesh:D…`, `unsdg:3`.
  `source_id` keeps the exact upstream value.
- Signature and content-owned link IDs are SHA-256 over their source context;
  stable for the same input and transform, independent of worker count.
- A value that is not a recognized identifier, a malformed MeSH code or a
  malformed abstract does **not** fail a file: that value is left out, counted,
  and a few examples kept (`notes` in each job and the generation manifest).
  A schema change in the source files does fail the job.
- Citation targets outside the snapshot remain plain IDs; no rows are invented.
