# First row-model test: real OpenAlex papers

These are the two saved source records, converted into the newly agreed tables. No full-corpus scan or production change.

## Rows

| Table | Rows |
|---|---:|
| `content` | 2 |
| `signatures` | 16 |
| `entities` | 19 |
| `classifications` | 80 |
| `links_content_signatures` | 16 |
| `links_signatures_entities` | 24 |
| `links_content_classifications` | 75 |
| `links_classifications_classifications` | 24 |
| `links_classifications_entities` | 19 |
| `links_content_entities` | 9 |
| `links_content_content` | 248 |

## Content

| Content ID | Year | Title | Citations |
|---|---:|---|---:|
| openalex:W2083027281 | 1995 | Caudal raphe-dorsal vagal complex peptidergic projections: Role in gastric vagal control | 71 |
| openalex:W2609272741 | 2017 | Activation of Brain Somatostatin Signaling Suppresses CRF Receptor-Mediated Stress Response | 39 |

## Author–affiliation pairs

Each row below is one signature. Long text is shortened **only in this display**; CSV and Parquet preserve the exact source strings. The source record and 1-based author/affiliation positions locate each occurrence.

| Content | Author order | Affiliation order | Raw author | Raw affiliation (preview) |
|---|---:|---:|---|---|
| openalex:W2083027281 | 1 | 1 | Y. Taché | CURE/Gastroenteric Biology Center, Department of Medicine, UCLA 90073, USA |
| openalex:W2083027281 | 1 | 2 | Y. Taché | Brain Research Institute, UCLA, Los Angeles, CA 90073 USA |
| openalex:W2083027281 | 1 | 3 | Y. Taché | CURE/Gastroenteric Biology Center, UCLA, Los Angeles, CA 90073 USA |
| openalex:W2083027281 | 1 | 4 | Y. Taché | Department of Medicine, UCLA, Los Angeles, CA 90073 USA |
| openalex:W2083027281 | 2 | 1 | H. Yang | Brain Research Institute, UCLA, Los Angeles, CA 90073 USA |
| openalex:W2083027281 | 2 | 2 | H. Yang | CURE/Gastroenteric Biology Center, UCLA, Los Angeles, CA 90073 USA |
| openalex:W2083027281 | 2 | 3 | H. Yang | Department of Medicine, UCLA, Los Angeles, CA 90073 USA |
| openalex:W2083027281 | 3 | 1 | H. Kaneko | Brain Research Institute, UCLA, Los Angeles, CA 90073 USA |
| openalex:W2083027281 | 3 | 2 | H. Kaneko | CURE/Gastroenteric Biology Center, UCLA, Los Angeles, CA 90073 USA |
| openalex:W2083027281 | 3 | 3 | H. Kaneko | Department of Medicine, UCLA, Los Angeles, CA 90073 USA |
| openalex:W2609272741 | 1 | 1 | Andreas Stengel | Division of Psychosomatic Medicine, Charité Center for Internal Medicine and Dermatology, Chari… |
| openalex:W2609272741 | 1 | 2 | Andreas Stengel | Division of Psychosomatic Medicine, Charité Center for Internal Medicine and Dermatology, Chari… |
| openalex:W2609272741 | 2 | 1 | Yvette F. Taché | VA Greater Los Angeles Health Care SystemLos Angeles, CA, USA |
| openalex:W2609272741 | 2 | 2 | Yvette F. Taché | Vatche and Tamar Manoukian Digestive Diseases Division, CURE Digestive Diseases Research Center… |
| openalex:W2609272741 | 2 | 3 | Yvette F. Taché | VA Greater Los Angeles Health Care System, Los Angeles, CA, USA |
| openalex:W2609272741 | 2 | 4 | Yvette F. Taché | Vatche and Tamar Manoukian Digestive Diseases Division, CURE Digestive Diseases Research Center… |

## Boundaries of this first test

- Four unit tables and seven link tables; every cell is scalar.
- Content includes bibliographic, open-access and identifier fields; the old flat table is unchanged.
- Signatures are author–affiliation pairs, never venue or funding entries.
- Topics and their hierarchy, keywords, sources and their hosts, MeSH, SDGs, awards and funders are represented. Deprecated OpenAlex concepts are listed in `checks.json` as deferred.
- Source assignment confidence remains null when OpenAlex provides none.
- Referenced works outside these two rows remain external IDs; no fabricated content rows.
- Repeated entity/category profiles are selected by source update time, then a stable tie-break, with conflicting observations reported in `checks.json`.
- Default extraction uses native Polars; `--engine reference` runs the independent Python oracle with the same source revisions.
- This tiny example is for inspecting rows. Representative whole-file timings are in `research/openalex/tidy-pilot/README.md`.

Exact schemas and nullability: [CONTRACTS.md](CONTRACTS.md). Counts, coverage notes, and endpoint checks: [checks.json](checks.json).
