# Executable table contracts

Version: `tidy-rows-v3`.
Generated from `pipeline/tidy/contracts.py`. This is the first row-model contract, not the full publisher-metadata projection.

## `content`

Primary key: `content_id`.

| Column | Type | Nullable |
|---|---|---|
| `content_id` | String | no |
| `source_id` | String | no |
| `source_revision` | String | no |
| `title` | String | yes |
| `doi` | String | yes |
| `type` | String | yes |
| `language` | String | yes |
| `date` | Date | yes |
| `year` | Int64 | yes |
| `abstract` | String | yes |
| `citations` | Int64 | yes |
| `references_count` | Int64 | yes |
| `is_complete` | Boolean | no |
| `is_retracted` | Boolean | yes |
| `updated_at` | Datetime(time_unit='us', time_zone='UTC') | yes |
| `volume` | String | yes |
| `issue` | String | yes |
| `first_page` | String | yes |
| `last_page` | String | yes |
| `is_oa` | Boolean | yes |
| `oa_status` | String | yes |
| `oa_url` | String | yes |
| `fwci` | Float64 | yes |
| `is_paratext` | Boolean | yes |
| `pmid` | String | yes |
| `pmcid` | String | yes |
| `mag` | String | yes |

## `signatures`

Primary key: `signature_id`.

| Column | Type | Nullable |
|---|---|---|
| `signature_id` | String | no |
| `author_text` | String | yes |
| `affiliation_text` | String | yes |
| `author_order` | Int64 | no |
| `affiliation_order` | Int64 | yes |
| `author_position` | String | yes |
| `is_corresponding` | Boolean | yes |
| `raw_orcid` | String | yes |
| `source_record` | String | no |
| `source_revision` | String | no |
| `author_source_field` | String | no |
| `affiliation_source_field` | String | yes |

## `entities`

Primary key: `entity_id`.

| Column | Type | Nullable |
|---|---|---|
| `entity_id` | String | no |
| `entity_type` | String | no |
| `name` | String | yes |
| `orcid` | String | yes |
| `ror` | String | yes |
| `country_code` | String | yes |
| `organization_type` | String | yes |
| `source_id` | String | no |
| `source_record` | String | no |
| `source_revision` | String | no |
| `source_field` | String | no |
| `observed_at` | Datetime(time_unit='us', time_zone='UTC') | yes |

## `classifications`

Primary key: `classification_id`.

| Column | Type | Nullable |
|---|---|---|
| `classification_id` | String | no |
| `classification_type` | String | no |
| `scheme` | String | no |
| `name` | String | yes |
| `notation` | String | yes |
| `source_id` | String | no |
| `source_record` | String | no |
| `source_revision` | String | no |
| `source_field` | String | no |
| `observed_at` | Datetime(time_unit='us', time_zone='UTC') | yes |

## `links_content_signatures`

Primary key: `link_id`.

| Column | Type | Nullable |
|---|---|---|
| `content_id` | String | no |
| `signature_id` | String | no |
| `link_id` | String | no |
| `relation` | String | no |
| `method` | String | no |
| `source_record` | String | no |
| `source_revision` | String | no |
| `source_field` | String | no |
| `score` | Float64 | yes |
| `score_kind` | String | yes |

## `links_signatures_entities`

Primary key: `link_id`.

| Column | Type | Nullable |
|---|---|---|
| `signature_id` | String | no |
| `entity_id` | String | no |
| `link_id` | String | no |
| `relation` | String | no |
| `method` | String | no |
| `source_record` | String | no |
| `source_revision` | String | no |
| `source_field` | String | no |
| `score` | Float64 | yes |
| `score_kind` | String | yes |

## `links_content_classifications`

Primary key: `link_id`.

| Column | Type | Nullable |
|---|---|---|
| `content_id` | String | no |
| `classification_id` | String | no |
| `link_id` | String | no |
| `relation` | String | no |
| `method` | String | no |
| `source_record` | String | no |
| `source_revision` | String | no |
| `source_field` | String | no |
| `score` | Float64 | yes |
| `score_kind` | String | yes |
| `is_primary` | Boolean | yes |
| `source_location_id` | String | yes |
| `raw_source_name` | String | yes |
| `landing_page_url` | String | yes |
| `pdf_url` | String | yes |
| `version` | String | yes |
| `license` | String | yes |
| `is_oa` | Boolean | yes |

## `links_classifications_classifications`

Primary key: `link_id`.

| Column | Type | Nullable |
|---|---|---|
| `from_classification_id` | String | no |
| `to_classification_id` | String | no |
| `link_id` | String | no |
| `relation` | String | no |
| `method` | String | no |
| `source_record` | String | no |
| `source_revision` | String | no |
| `source_field` | String | no |
| `score` | Float64 | yes |
| `score_kind` | String | yes |
| `observed_at` | Datetime(time_unit='us', time_zone='UTC') | yes |

## `links_classifications_entities`

Primary key: `link_id`.

| Column | Type | Nullable |
|---|---|---|
| `classification_id` | String | no |
| `entity_id` | String | no |
| `link_id` | String | no |
| `relation` | String | no |
| `method` | String | no |
| `source_record` | String | no |
| `source_revision` | String | no |
| `source_field` | String | no |
| `score` | Float64 | yes |
| `score_kind` | String | yes |
| `observed_at` | Datetime(time_unit='us', time_zone='UTC') | yes |

## `links_content_entities`

Primary key: `link_id`.

| Column | Type | Nullable |
|---|---|---|
| `content_id` | String | no |
| `entity_id` | String | no |
| `link_id` | String | no |
| `relation` | String | no |
| `method` | String | no |
| `source_record` | String | no |
| `source_revision` | String | no |
| `source_field` | String | no |
| `score` | Float64 | yes |
| `score_kind` | String | yes |

## `links_content_content`

Primary key: `link_id`.

| Column | Type | Nullable |
|---|---|---|
| `from_content_id` | String | no |
| `to_content_id` | String | no |
| `link_id` | String | no |
| `relation` | String | no |
| `method` | String | no |
| `source_record` | String | no |
| `source_revision` | String | no |
| `source_field` | String | no |
| `score` | Float64 | yes |
| `score_kind` | String | yes |
