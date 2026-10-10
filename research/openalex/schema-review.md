# A smaller, shared schema for scholarly and other content

**Research date:** October 5, 2026  
**Status:** Historical research proposal; the universal-record recommendation
was not adopted. The current agreement is in
[SCHEMA.md](../../pipeline/SCHEMA.md): separate content, signatures, entities,
and classifications, with `links_<table>_<table>` connections. The source
research below remains background, not the implementation plan.

## Recommendation in brief

Use **three reusable table shapes**:

1. **records** — things with an identity: content, source observations, and clusters;
2. **links** — typed, attributed connections between records;
3. **values** — extra scalar properties of a record or link.

Keep `content`, `signatures`, `signature_clusters`, and `content_clusters` as human-friendly views. Keep our existing wide, flat content Parquet table as a materialized view for fast analysis. This is a shared logical model, not a requirement to put every kind into one enormous file or turn every existing content column into a separate property row.

The important distinction is **evidence versus interpretation**, not author versus institution versus venue. Another important distinction is **identity versus grouping**: two names referring to one person is not the same relationship as two papers discussing one topic.

This is my proposed synthesis of the systems below; it is not a claim that any of them uses these exact physical tables internally.

## 1. What the sources actually show

### OpenAlex: raw evidence and resolved objects coexist

An authorship is a work-specific author entry. It preserves `raw_author_name`, raw affiliations, author position, and corresponding status alongside the resolved author. Its `affiliations` component retains the mapping from each raw string to zero or more institution IDs. The flattened institutions list is a convenience, not an equivalent replacement for that mapping. [S1]

A location similarly retains a raw venue name alongside a resolved source, URLs, version, and access information. Different copies of a work can carry different licences. OpenAlex also describes its work-building process as matching harvested locations into works. **Content identity itself can therefore be a clustering problem**, not just author identity. [S2]

**Design lesson:** preserve contextual source observations; attach resolution results to them. A publisher copy and a repository copy cannot be reduced to a venue name alone.

### Semantic Scholar: signatures and contextual relationships

The official S2AND input model separates papers from signatures. A signature includes its paper ID, author position, parsed name, and affiliations. Signatures are the inputs to author clustering. The Academic Graph API also exposes citation-specific fields such as contexts, intents, and an influence indicator. [S3, S4]

**Design lesson:** an author's entry is a contextual observation, not just a deduplicated string. Links sometimes carry useful data of their own.

### Google Scholar: useful behaviour, not a published internal schema

Its public help describes author profiles, article grouping, and merging versions. Merging two article records deduplicates citing papers rather than simply adding citation totals. Google also states that it does not provide bulk record access. [S5, S6]

**Design lesson:** retain source records and reversible version/grouping decisions; deduplicate content before metrics. The public documentation does not establish Google's internal database design, and this review does not claim to have inspected it.

### OpenAIRE: one reusable relationship representation

OpenAIRE documents research products, organizations, projects, data sources, and other entities. Its 11.4.0 relationship format provides source, target, relationship semantics, provenance, and validation information. Its authorship relation additionally preserves byline rank, contribution roles, corresponding status, and raw-affiliation-to-organization matches with provenance. [S7, S8]

**Design lesson:** one link schema can replace many pair-specific link tables, without replacing meaningful relationship names with one vague `belongs_to` label.

### Crossref and DataCite: distinguish relationship meanings

Their vocabularies distinguish citations, versions, parts, translations, preprints, and other links between research objects. DataCite 4.6 also explicitly supports awards and projects as resource types. [S9, S10]

**Design lesson:** a grant can be an ordinary record when it has its own identity; no special grant table is required. A translation, a preprint, and an identical duplicate must not all become `same_as`.

### ROR: organizations have multiple roles

ROR allows multiple organization types. Its own schema example gives the University of Bern both education and funder types. It also records names, external identifiers, and typed organizational relationships. [S11]

**Design lesson:** a university that funds research and publishes a journal need not become three unrelated objects. Identity belongs to the organization; affiliation, funding, and publishing are roles or relations. Cross-source identity matching still requires evidence.

### Wikidata, SKOS, Schema.org, and PROV

Wikidata statements have identity and can carry qualifiers and references. SKOS distinguishes concepts, schemes, broader/narrower relations, collections, and mappings; it allows multiple broader paths. Schema.org Role adds context to a relationship. PROV models attribution and derivation. [S12–S15]

**Design lesson:** keep evidence and scope on claims; allow multiple parents where meaningful; represent context explicitly. We can borrow these principles without adopting a graph database or implementing every standard.

## 2. Why our earlier design kept growing

We were treating each new kind as a reason for another table:

- a printed author entry → signatures;
- a printed institution → affiliation strings;
- a funding record → funding statements;
- a venue occurrence → locations;
- and another pair-specific table for each kind of connection.

My proposed simplification is to treat these as **kinds of contextual observations** sharing one record shape.

The cluster split also overlaps. A journal can be the resolved target of a raw venue name and the venue grouping a set of papers. An organization can be resolved from affiliation text, appear in a funding record, and publish a journal. Keeping separate identities for every role creates unnecessary reconciliation work.

This does not mean that everything is interchangeable. It means that the distinctions belong in record types and relationship types rather than in a growing list of physical table definitions.

## 3. The proposed shared tables

### A. `records`: things we can refer to

Illustrative common columns:

| Column | Meaning |
|---|---|
| `record_id` | Stable, namespaced identifier |
| `layer` | `content`, `observation`, or `cluster` |
| `kind` | Article, blog post, author entry, affiliation, location, person, organization, venue, topic, etc. |
| `label` | Display name; for a raw-text observation, the exact source text where available |
| `context_id` | Optional immediate containing record, such as the author entry containing an affiliation |
| `source_key` | Provider-native identifier or a pointer into the original source record |
| `batch_id` | Import or inference run that produced this record assertion |

These are proposed columns, not an implemented contract. Revision handling must be finalized before production ingestion. A source observation should be identifiable by source record revision plus its path/position, not just a hash of text. Two identical names on the same paper can still be different entries.

Examples of the three layers:

- **content:** the article, blog post, patent document, Wikipedia page;
- **observation:** an author entry on an item, one of its affiliation lines, a location record, or a structured funding entry;
- **cluster:** a resolved person/organization/venue, a subject concept, or a reading list.

Here “cluster” is deliberately broad. Some entries are imported registries or user-created groups, not the output of a clustering algorithm. A human or organization exists independently of our grouping decisions; the cluster is our representation of its identity.

These layers are organizational labels, not a claim that content cannot itself be clustered. Source publication records can `resolve_to` a content record, just as author entries resolve to a person. An OpenAlex work is already an upstream interpretation of source records: our snapshot is not necessarily the untouched publisher input. Retain that boundary and do not infer original source-version context when the input no longer supplies it.

### B. `links`: one schema for connections

Illustrative columns:

`link_id, from_id, relation, to_id, evidence_id, method_id, batch_id, score, score_kind, status`

- `evidence_id` points to the observation or source record supporting the claim.
- `method_id` distinguishes an imported OpenAlex assignment from our own model or a manual correction.
- A score is optional, with a named interpretation. An imported ID without a supplied confidence must not silently acquire confidence 1.0.
- Taxonomy membership and inference provenance are separate concepts: a MeSH term's scheme is not the algorithm assigning it.

Example relationships:

| Relation | Meaning |
|---|---|
| `resolves_to` | This observation is interpreted as referring to this entity |
| `authored_by` | Content was authored by this person; can be derived from signatures |
| `about` | Content discusses a concept |
| `member_of` | Content belongs to a selected collection |
| `published_in` | Content appeared in this venue |
| `hosted_by` | A copy is hosted by this source |
| `funded_by` | Content or an award is supported by this organization |
| `cites` | One content item cites another |
| `version_of` | One content item is a version of another |
| `broader` | A concept has a broader concept |
| `published_by` | A venue is published by an organization |
| `same_as` | Two entity records are asserted to identify the same thing |

Memberships may overlap. One raw affiliation can resolve to several organizations. A subject may have several broader concepts. Competing providers or models can retain separate assertions instead of overwriting one another.

Use `context_id` for unambiguous source containment, not for taxonomy parents. Shared links, such as one printed affiliation marker referenced by several authors, belong in the links table. When a provider only exposes affiliation strings repeated under each author, retain that source structure rather than inventing a shared printed marker.

### C. `values`: scalar details without a new side table per kind

Illustrative columns:

`value_id, subject_id, subject_kind, property, ordinal, text_value, int_value, float_value, bool_value, date_value, timestamp_value, batch_id`

Exactly one typed value field is populated. `subject_kind` distinguishes a record from a link, so ownership is not ambiguous. Repeated values become multiple rows; `ordinal` is used only when ordering matters.

Examples:

- an author entry: byline order and corresponding-author status;
- a location: landing URL, licence, accepted/published version;
- a person: ORCID;
- an organization: ROR identifier;
- a venue: ISSN;
- a funding observation: award number as supplied;
- a relation: validity start/end dates or another qualifier.

No nested JSON is required inside these canonical tables. The original source payload is retained separately as evidence, not silently discarded.

**Storage caveat:** this table is an extension mechanism, not an instruction to unpivot all 38 content columns across the entire corpus. Wide, typed tables remain the efficient reading/analysis interface. We should benchmark before deciding which scalar fields also need canonical property rows versus immutable source storage and materialized columns.

## 4. How our existing names survive

| Familiar name | Proposed implementation |
|---|---|
| `content` | Wide materialized view of content records and their selected metadata |
| `signatures` | Author-entry observations with their name and associated affiliation observations |
| `signature_clusters` | View of cluster records reached by relevant resolution links |
| `content_clusters` | View of clusters grouping content through named relations |
| `authors`, `institutions`, `sources` | Typed views, not independent identity systems |
| pair-specific link tables | Filters of `links` by relation and endpoint kinds |

These views are allowed to overlap. In particular, a venue can appear in both cluster views. It is stored once, but can be queried from both directions.

If the name `signatures` should include raw affiliation occurrences directly, that is a view choice. The base model does not require us to call funding metadata a signature or fabricate a single raw string by joining a name and affiliations that were separate upstream.

## 5. Test against two real records in our snapshot

I re-read the two-paper fixture used earlier. The following raw strings and OpenAlex mappings are from that local fixture; the short observation IDs are invented only for this diagram.

### Verified source records

- **W2083027281 (1995):** Caudal raphe-dorsal vagal complex peptidergic projections: Role in gastric vagal control.
- **W2609272741 (2017):** Activation of Brain Somatostatin Signaling Suppresses CRF Receptor-Mediated Stress Response.
- Author names: **`Y. Taché`** on the former and **`Yvette F. Taché`** on the latter.
- Both authorship entries resolve in our snapshot to **A5010241733**, whose display name is Yvette Taché.
- One exact affiliation on the 1995 entry is **`Department of Medicine, UCLA, Los Angeles, CA 90073 USA`**, mapped to **I161318765**.
- The 1995 primary raw source name is **`Peptides`**, resolved to **S120382301**.

### Shared records

| record_id | layer | kind | label | context_id |
|---|---|---|---|---|
| openalex:W2083027281 | content | article | Caudal raphe-dorsal vagal complex… [display abbreviation] | — |
| obs:1995-author | observation | author_entry | Y. Taché | openalex:W2083027281 |
| obs:1995-affiliation | observation | affiliation | Department of Medicine, UCLA, Los Angeles, CA 90073 USA | obs:1995-author |
| obs:1995-location | observation | location | Peptides | openalex:W2083027281 |
| obs:2017-author | observation | author_entry | Yvette F. Taché | openalex:W2609272741 |
| openalex:A5010241733 | cluster | person | Yvette Taché | — |
| openalex:I161318765 | cluster | organization | University of California, Los Angeles | — |
| openalex:S120382301 | cluster | venue | Peptides | — |

### Shared links

| from | relation | to | provenance |
|---|---|---|---|
| obs:1995-author | resolves_to | openalex:A5010241733 | imported OpenAlex assignment |
| obs:2017-author | resolves_to | openalex:A5010241733 | imported OpenAlex assignment |
| obs:1995-affiliation | resolves_to | openalex:I161318765 | imported OpenAlex assignment |
| obs:1995-location | resolves_to | openalex:S120382301 | imported OpenAlex assignment |

The affiliation stays attached to the right author entry. No separate institution-membership table is needed. We can derive paper→person and paper→institution links for fast queries, but label them as derived and retain the supporting path.

These mappings are **OpenAlex's assertions in our snapshot**, not independent proof of identity. An ORCID on a resolved profile is also not independent proof that the profile was correctly clustered.

## 6. Cases the shared model must pass

| Case | Required behaviour |
|---|---|
| Two J. Smiths, even on one paper | Distinct occurrences identified by source position, not string alone |
| One name plus three affiliations | Contextual child observations; one author entry |
| One affiliation naming a university and a hospital | Multiple resolution links |
| Two providers disagree about an author | Separate source assertions and methods; no destructive overwrite |
| A university funds and publishes work | One organization identity, different relationships |
| A journal changes publisher | Dated `published_by` links, not an immutable parent field |
| A paper has publisher and repository copies | Separate location observations with their own access/version details |
| A grant appears only as structured fields | Preserve structured evidence; no invented raw funding sentence |
| A topic has two broader subjects | Multiple `broader` links |
| A preprint and journal article | Typed version/preprint relation, not automatic identity collapse |
| A Wikipedia page describes a scientist | Content `about` person; page and person remain different |
| Three collaborators from the same university | Institution metrics count the paper once, not three signatures |

## 7. Corrections to earlier proposals

1. **Not every relation is membership.** `about`, `resolves_to`, `published_in`, `funded_by`, and `cites` should stay distinguishable.
2. **A single `parent_id` is insufficient.** Classification hierarchies can have multiple parents; publishers are organizations linked to venues, not simply broader topics. [S13, S11]
3. **Do not infer raw funding statements from cleaned funder names and award numbers.** In our fixture we have structured award metadata, not the original acknowledgements text.
4. **Do not promise repeated affiliation text is nearly free.** Compression is an implementation detail whose actual storage cost must be measured. It should not determine what one signature means.
5. **Imported metrics and metrics computed from our collected links are different quantities.** Preserve source and snapshot. Before h-index aggregation, define the content set, identity/version policy, citation source, and time window.
6. **A title/author/topic alone is not proof of identity.** Keep provider mappings and local inference separately inspectable.

## 8. Alternatives and the trade-offs

| Approach | Strength | Cost |
|---|---|---|
| Separate domain tables and pair-specific links | Very explicit columns and easy constraints | Many schemas and repeated plumbing |
| Content + observations + clusters + shared links | Strong simplification with familiar concepts | Extra properties still need extensions or kind-specific columns |
| Shared records + links + typed values | Few reusable ingestion, validation, and query mechanisms | More joins; property vocabularies and validation must be governed |
| One table of arbitrary subject–predicate–object values | Minimal table count | Harder typing, weaker readability, potentially many more rows |

**My recommendation:** shared records/links/values as the extensible logical core, plus ordinary wide views for everyday use. Accept a small amount of materialized duplication for readability and speed. Do not switch storage engines merely because the model is graph-shaped.

We still need an import/run manifest and schema/vocabulary definitions. They are operational metadata, not new scientific entity tables; calling this a three-shape model must not hide those requirements.

A narrow physical citation partition may omit unused qualifier columns while presenting the shared logical link schema. Similarly, partition records by layer/kind and relations by predicate, with further bucketing determined by benchmark results. Avoid one file containing all kinds merely to achieve a small table count.

## 9. A safe next step

Before a corpus-wide rebuild:

1. Agree on a minimal type and relation vocabulary.
2. Build a small fixture covering every case in section 6, using the real two-paper example plus explicitly labelled synthetic ambiguity cases.
3. Prototype the shared tables and the familiar views over one bounded raw-data sample.
4. Verify reconstruction of raw authorship/location/award fields, with source paths preserved.
5. Compare views against the existing content table, explicitly allowing documented improvements rather than hiding differences.
6. Benchmark disk size, peak RAM, author→papers lookup, topic filtering, and update/retraction behaviour.
7. Only then choose the physical layout and expand ingestion.

No processing-time estimate is justified yet. The earlier pipeline demonstrates that small-sample timings and schema elegance alone do not predict full-corpus memory behaviour.

## Sources

I fetched the first-party documentation and repository files directly over HTTP and inspected their text and, for the Semantic Scholar API, its machine-readable specification. Only first-party documentation or project repositories are used below. The DataCite comparison specifically uses version 4.6, not a claim about the newest version. Google Scholar evidence concerns documented public behaviour, not internal storage.

- **S1 — OpenAlex authorships:** `https://help.openalex.org/data/authorships/`
- **S2 — OpenAlex locations:** `https://help.openalex.org/data/locations/`
- **S3 — Allen Institute S2AND, documented signature input:** `https://github.com/allenai/S2AND/blob/main/docs/production_inference.md`
- **S4 — Semantic Scholar Academic Graph API:** `https://api.semanticscholar.org/api-docs/graphs`; machine-readable specification inspected at `https://api.semanticscholar.org/graph/v1/swagger.json`
- **S5 — Google Scholar Profiles, merging and citation counts:** `https://scholar.google.com/intl/en-US/scholar/citations.html`
- **S6 — Google Scholar Search Help, bulk access:** `https://scholar.google.com/intl/en/scholar/help.html`
- **S7 — OpenAIRE data model, version 11.4.0:** `https://graph.openaire.eu/docs/data-model/`
- **S8 — OpenAIRE relationships and authorship, version 11.4.0:** `https://graph.openaire.eu/docs/data-model/relationships/relationship-object/` and `https://graph.openaire.eu/docs/data-model/relationships/person/authorship_relation/`
- **S9 — Crossref relationship types:** `https://www.crossref.org/documentation/schema-library/markup-guide-metadata-segments/relationships/`
- **S10 — DataCite 4.6 and related identifiers:** `https://schema.datacite.org/meta/kernel-4.6/` and `https://datacite-metadata-schema.readthedocs.io/en/4.6/properties/relatedidentifier/`
- **S11 — ROR data structure:** `https://ror.readme.io/docs/ror-data-structure`
- **S12 — Wikibase statement/qualifier/reference data model:** `https://doc.wikimedia.org/Wikibase/REL1_35/php/md_docs_2topics_2json.html`
- **S13 — W3C SKOS Reference:** `https://www.w3.org/TR/skos-reference/`
- **S14 — Schema.org Role:** `https://schema.org/Role`
- **S15 — W3C PROV-O:** `https://www.w3.org/TR/prov-o/`

Local evidence: the two records re-read from `/tmp/ex.parquet` during this review and preserved in `research/openalex/schema-review-examples.parquet`. Raw text is quoted exactly above except where a display abbreviation is explicitly marked; no new author, institution, or grant matches were inferred.
