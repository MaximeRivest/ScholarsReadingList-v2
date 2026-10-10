"""Scalar table schemas and small-data integrity checks for the agreed model."""
from dataclasses import dataclass
from datetime import date, datetime
import math

import polars as pl

SCHEMA_VERSION = 'tidy-rows-v3'


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Table:
    columns: dict
    required: tuple[str, ...]
    key: str


STR, INT, FLOAT, BOOL, DATE = pl.String, pl.Int64, pl.Float64, pl.Boolean, pl.Date
UTC = pl.Datetime('us', 'UTC')
ORIGIN = {'source_record': STR, 'source_revision': STR, 'source_field': STR}
CLAIM = {'link_id': STR, 'relation': STR, 'method': STR, **ORIGIN,
         'score': FLOAT, 'score_kind': STR}
CLAIM_REQUIRED = ('link_id', 'relation', 'method', *ORIGIN)

TABLES = {
    'content': Table({
        'content_id': STR, 'source_id': STR, 'source_revision': STR,
        'title': STR, 'doi': STR, 'type': STR, 'language': STR,
        'date': DATE, 'year': INT, 'abstract': STR, 'citations': INT,
        'references_count': INT, 'is_complete': BOOL, 'is_retracted': BOOL,
        'updated_at': UTC, 'volume': STR, 'issue': STR, 'first_page': STR, 'last_page': STR,
        'is_oa': BOOL, 'oa_status': STR, 'oa_url': STR, 'fwci': FLOAT, 'is_paratext': BOOL,
        'pmid': STR, 'pmcid': STR, 'mag': STR,
    }, ('content_id', 'source_id', 'source_revision', 'is_complete'), 'content_id'),
    'signatures': Table({
        'signature_id': STR, 'author_text': STR, 'affiliation_text': STR,
        'author_order': INT, 'affiliation_order': INT, 'author_position': STR,
        'is_corresponding': BOOL, 'raw_orcid': STR,
        'source_record': STR, 'source_revision': STR,
        'author_source_field': STR, 'affiliation_source_field': STR,
    }, ('signature_id', 'author_order', 'source_record', 'source_revision',
        'author_source_field'), 'signature_id'),
    'entities': Table({
        'entity_id': STR, 'entity_type': STR, 'name': STR, 'orcid': STR,
        'ror': STR, 'country_code': STR, 'organization_type': STR,
        'source_id': STR, **ORIGIN, 'observed_at': UTC,
    }, ('entity_id', 'entity_type', 'source_id', *ORIGIN), 'entity_id'),
    'classifications': Table({
        'classification_id': STR, 'classification_type': STR, 'scheme': STR,
        'name': STR, 'notation': STR, 'source_id': STR, **ORIGIN, 'observed_at': UTC,
    }, ('classification_id', 'classification_type', 'scheme', 'source_id', *ORIGIN),
        'classification_id'),
    'links_content_signatures': Table({
        'content_id': STR, 'signature_id': STR, **CLAIM,
    }, ('content_id', 'signature_id', *CLAIM_REQUIRED), 'link_id'),
    'links_signatures_entities': Table({
        'signature_id': STR, 'entity_id': STR, **CLAIM,
    }, ('signature_id', 'entity_id', *CLAIM_REQUIRED), 'link_id'),
    'links_content_classifications': Table({
        'content_id': STR, 'classification_id': STR, **CLAIM,
        'is_primary': BOOL, 'source_location_id': STR, 'raw_source_name': STR,
        'landing_page_url': STR, 'pdf_url': STR, 'version': STR,
        'license': STR, 'is_oa': BOOL,
    }, ('content_id', 'classification_id', *CLAIM_REQUIRED), 'link_id'),
    'links_classifications_classifications': Table({
        'from_classification_id': STR, 'to_classification_id': STR, **CLAIM, 'observed_at': UTC,
    }, ('from_classification_id', 'to_classification_id', *CLAIM_REQUIRED), 'link_id'),
    'links_classifications_entities': Table({
        'classification_id': STR, 'entity_id': STR, **CLAIM, 'observed_at': UTC,
    }, ('classification_id', 'entity_id', *CLAIM_REQUIRED), 'link_id'),
    'links_content_entities': Table({
        'content_id': STR, 'entity_id': STR, **CLAIM,
    }, ('content_id', 'entity_id', *CLAIM_REQUIRED), 'link_id'),
    'links_content_content': Table({
        'from_content_id': STR, 'to_content_id': STR, **CLAIM,
    }, ('from_content_id', 'to_content_id', *CLAIM_REQUIRED), 'link_id'),
}

RELATIONS = {
    'links_content_signatures': {'has_signature'},
    'links_signatures_entities': {'identifies_person', 'identifies_institution'},
    'links_content_classifications': {'about', 'published_in', 'hosted_by', 'funded_by'},
    'links_classifications_classifications': {'broader'},
    'links_content_content': {'cites'},
    'links_classifications_entities': {'host_organization', 'funded_by'},
    'links_content_entities': {'funded_by'},
}

# Vocabulary/profile tables: one row per key across the whole dataset, chosen by
# reduction.RANK from all current-source observations. Everything else is owned
# by exactly one content item and needs no cross-partition reduction.
REDUCED = ('entities', 'classifications', 'links_classifications_classifications',
           'links_classifications_entities')
CONTENT_OWNED = tuple(t for t in TABLES if t not in REDUCED)
OWNER = {'content': 'content_id', 'signatures': 'source_record',
         'links_content_signatures': 'content_id', 'links_signatures_entities': 'source_record',
         'links_content_classifications': 'content_id', 'links_content_content': 'from_content_id',
         'links_content_entities': 'content_id'}


def _scalar_ok(value, dtype):
    if dtype == STR:
        return isinstance(value, str)
    if dtype == INT:
        return type(value) is int and -(2**63) <= value < 2**63
    if dtype == FLOAT:
        return type(value) in (float, int) and math.isfinite(value)
    if dtype == BOOL:
        return type(value) is bool
    if dtype == DATE:
        return type(value) is date
    if dtype == UTC:
        return isinstance(value, datetime) and value.utcoffset() is not None
    return False


def frame(name, rows):
    """Reject nested/unexpected/wrongly typed values before Polars can coerce them."""
    contract = TABLES[name]
    checked = []
    for i, row in enumerate(rows):
        unknown = set(row) - contract.columns.keys()
        if unknown:
            raise ContractError(f'{name}[{i}]: unexpected columns {sorted(unknown)}')
        out = {col: row.get(col) for col in contract.columns}
        for col, value in out.items():
            if value is None:
                if col in contract.required:
                    raise ContractError(f'{name}[{i}].{col}: required value is missing')
            elif not _scalar_ok(value, contract.columns[col]):
                raise ContractError(f'{name}[{i}].{col}: expected {contract.columns[col]}, got {type(value).__name__}')
        checked.append(out)
    return pl.DataFrame(checked, schema=contract.columns, strict=True)


def validate(tables):
    """Validate a small complete sample; citation targets may be outside it.

    This uses in-memory sets and is a reference checker, not the large-corpus
    validation implementation. Endpoint IDs are table-specific, never member_id.
    """
    if set(tables) != set(TABLES):
        raise ContractError('Expected exactly the declared unit and link tables')
    rows, ids = {}, {}
    for name, contract in TABLES.items():
        df = tables[name]
        if dict(df.schema) != contract.columns:
            raise ContractError(f'{name}: schema differs from contract')
        records = df.to_dicts()
        frame(name, records)
        keys = [r[contract.key] for r in records]
        if len(keys) != len(set(keys)):
            raise ContractError(f'{name}: duplicate {contract.key}')
        if any(not key for key in keys):
            raise ContractError(f'{name}: empty key')
        rows[name], ids[name] = records, set(keys)
        for r in records:
            if name in RELATIONS and r['relation'] not in RELATIONS[name]:
                raise ContractError(f'{name}: unsupported relation {r["relation"]}')
            if name.startswith('links_'):
                if (r['score'] is None) != (r['score_kind'] is None):
                    raise ContractError(f'{name}: score and score_kind must be supplied together')
                if r['source_record'] not in ids.get('content', set()):
                    raise ContractError(f'{name}: unknown evidence content')
        if name.startswith('links_'):
            claims = [tuple(r[c] for c in contract.columns if c != 'link_id') for r in records]
            if len(claims) != len(set(claims)):
                raise ContractError(f'{name}: duplicate source assertion')
    foreign_keys = {
        'links_content_signatures': {'content_id': 'content', 'signature_id': 'signatures'},
        'links_signatures_entities': {'signature_id': 'signatures', 'entity_id': 'entities'},
        'links_content_classifications': {'content_id': 'content', 'classification_id': 'classifications'},
        'links_classifications_classifications': {
            'from_classification_id': 'classifications', 'to_classification_id': 'classifications'},
        'links_content_content': {'from_content_id': 'content'},
        'links_classifications_entities': {'classification_id': 'classifications', 'entity_id': 'entities'},
        'links_content_entities': {'content_id': 'content', 'entity_id': 'entities'},
    }
    for table, fields in foreign_keys.items():
        for r in rows[table]:
            for col, target in fields.items():
                if r[col] not in ids[target]:
                    raise ContractError(f'{table}.{col}: unknown endpoint {r[col]}')
    owners = {}
    content = {r['content_id']: r for r in rows['content']}
    for table in ('entities', 'classifications'):
        for r in rows[table]:
            source = content.get(r['source_record'])
            if source is None or source['source_revision'] != r['source_revision']:
                raise ContractError(f'{table}: profile evidence must identify a current source record')
    for r in rows['content']:
        if any(r[c] is not None and r[c] < 0 for c in ('citations', 'references_count')):
            raise ContractError('Citation/reference counts cannot be negative')
    for r in rows['links_content_signatures']:
        if r['signature_id'] in owners:
            raise ContractError('Each signature must have exactly one content owner')
        owners[r['signature_id']] = r['content_id']
    if set(owners) != ids['signatures']:
        raise ContractError('Every signature must have a content owner')
    positions = set()
    sigs = {r['signature_id']: r for r in rows['signatures']}
    for r in rows['signatures']:
        owner = owners[r['signature_id']]
        if (owner, content[owner]['source_revision']) != (r['source_record'], r['source_revision']):
            raise ContractError('Signature must belong to the current owning content revision')
        if r['author_order'] < 1 or (r['affiliation_order'] is not None and r['affiliation_order'] < 1):
            raise ContractError('Source occurrence orders are 1-based')
        if r['author_position'] not in (None, 'first', 'middle', 'last'):
            raise ContractError('Unknown byline position')
        if r['affiliation_order'] is None and r['affiliation_text'] is not None:
            raise ContractError('An affiliation text requires its source order')
        key = (owner, r['author_order'], r['affiliation_order'])
        if key in positions:
            raise ContractError('Duplicate author/affiliation occurrence')
        positions.add(key)
    entity_types = {r['entity_id']: r['entity_type'] for r in rows['entities']}
    if set(entity_types.values()) - {'person', 'organization'}:
        raise ContractError('Unknown entity type')
    for name in RELATIONS:
        for r in rows[name]:
            if r['source_revision'] != content[r['source_record']]['source_revision']:
                raise ContractError('Link evidence must use the current source revision')
            endpoint = OWNER.get(name)
            if endpoint and r[endpoint] != r['source_record']:
                raise ContractError('Work-owned link evidence comes from another content item')
            if name in ('links_classifications_entities', 'links_content_entities') and entity_types[r['entity_id']] != 'organization':
                raise ContractError('Funding/host links must point to organizations')
            if name == 'links_signatures_entities':
                signature = sigs[r['signature_id']]
                if r['source_record'] != signature['source_record']:
                    raise ContractError('Signature match evidence comes from another content item')
                expected = 'person' if r['relation'] == 'identifies_person' else 'organization'
                if entity_types[r['entity_id']] != expected:
                    raise ContractError('Signature match relation has the wrong entity type')
            if name == 'links_classifications_classifications' and r['from_classification_id'] == r['to_classification_id']:
                raise ContractError('A classification cannot be broader than itself')
    cited = [(r['from_content_id'], r['to_content_id']) for r in rows['links_content_content']]
    if len(cited) != len(set(cited)):
        raise ContractError('Duplicate citation pair in this source assignment')
    # A sample need not contain every referenced work. Report, do not fabricate it.
    return {'external_reference_targets': len({target for _, target in cited} - ids['content'])}
