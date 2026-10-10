"""In-memory, partition-sized native Polars extraction. No per-row Python UDFs.

Dependencies are pinned in requirements.txt. All arrays are expanded by source
occurrence; raw affiliation mappings, not aggregate institutions, determine links.
Must produce exactly the rows of reference.py (tests compare all tables).
"""
from hashlib import sha256
import json

import polars as pl
import polars_hash as ph

from .contracts import TABLES, REDUCED, ContractError, UTC
from .keys import ENDPOINTS, expr_key, expr_link_key
from .reduction import reduce_frame
from .reference import METHOD, Result
from .source_roles import PUBLICATION_VENUES, KNOWN_VERSIONS

C = pl.col
L = pl.lit


def cat(*parts):
    return pl.concat_str([L(p) if isinstance(p, str) else p for p in parts])


def field(df, *path, dtype=None, missing=pl.String):
    schema = df.schema
    actual = schema.get(path[0], pl.Null)
    e = C(path[0])
    for p in path[1:]:
        if not isinstance(actual, pl.Struct) or p not in dict(actual):
            actual = pl.Null
            break
        actual = dict(actual)[p]
        e = e.struct.field(p)
    if actual == pl.Null or (isinstance(actual, pl.Struct) and not actual.fields):
        return L(None, dtype=dtype or missing)
    expected = dtype or missing
    where = '.'.join(path)
    if expected == pl.String and actual != pl.String:
        raise ContractError(f'{where} must be text, not {actual}')
    if expected == pl.Boolean and actual != pl.Boolean:
        raise ContractError(f'{where} must be boolean, not {actual}')
    if expected == pl.Int64 and not actual.is_integer():
        raise ContractError(f'{where} must be integer, not {actual}')
    if expected == pl.Float64 and not actual.is_numeric():
        raise ContractError(f'{where} must be numeric, not {actual}')
    if isinstance(expected, pl.List) and not isinstance(actual, pl.List):
        raise ContractError(f'{where} must be a list, not {actual}')
    if isinstance(expected, pl.Struct) and not isinstance(actual, pl.Struct):
        raise ContractError(f'{where} must be a struct, not {actual}')
    if dtype == pl.Date and actual not in (pl.Date, pl.String):
        raise ContractError(f'{where} must be a date or ISO date text, not {actual}')
    if dtype == pl.Date and actual == pl.String:
        return e.str.to_date(strict=True)
    return e.cast(dtype, strict=True) if dtype is not None else e


def array(df, *path):
    return field(df, *path, missing=pl.List(pl.Null)).fill_null([])


def oid(e):
    valid = e.str.contains(r'^(https?://openalex\.org/|openalex:)(?s:.+)$')
    return pl.when(valid).then(e.str.replace(r'^https?://openalex\.org/', 'openalex:'))


def nonempty(name):
    return C(name).is_not_null() & (C(name) != '')


def timestamp(df, column):
    dtype = df.schema.get(column)
    e = field(df, column, missing=dtype or UTC)
    if dtype == pl.String:
        return pl.coalesce(e.str.to_datetime(format='%+', time_zone='UTC', strict=False),
                           e.str.to_datetime(format='%Y-%m-%dT%H:%M:%S%.f', time_zone='UTC', strict=False),
                           e.str.to_datetime(format='%Y-%m-%d %H:%M:%S%.f', time_zone='UTC', strict=False),
                           e.str.to_date('%Y-%m-%d', strict=False).cast(UTC))
    if isinstance(dtype, pl.Datetime):
        return e.dt.replace_time_zone('UTC').cast(UTC) if dtype.time_zone is None else e.dt.convert_time_zone('UTC').cast(UTC)
    if dtype in (None, pl.Null):
        return L(None, UTC)
    raise ContractError(f'{column} must be an ISO timestamp or Datetime, not {dtype}')


def _canonical(e, dtype):
    if isinstance(dtype, pl.Struct):
        if not dtype.fields:
            return e
        return pl.when(e.is_not_null()).then(pl.struct([
            _canonical(e.struct.field(f.name), f.dtype).alias(f.name)
            for f in sorted(dtype.fields, key=lambda f: f.name)]))
    if isinstance(dtype, pl.List):
        return e.list.eval(_canonical(pl.element(), dtype.inner))
    return e


def source_revision_expr(raw):
    """Versioned canonical native row encoding + SHA-256, not Expr.hash().

    Column/struct names are included via a schema-shape token; field order is
    canonical. The source schema must be preserved when splitting a partition.
    """
    def shape(dtype):
        if isinstance(dtype, pl.Struct):
            return {f.name: shape(f.dtype) for f in sorted(dtype.fields, key=lambda f: f.name)}
        if isinstance(dtype, pl.List):
            return [shape(dtype.inner)]
        return 'scalar'  # encoder v1 normalizes equivalent numeric/time storage widths
    schema = raw.schema
    names = sorted(schema)
    shape_token = sha256(json.dumps({n: shape(schema[n]) for n in names}, sort_keys=True).encode()).hexdigest()
    expressions = [L(shape_token).alias('_shape')] + [_canonical(C(n), schema[n]).alias(n) for n in names]
    return cat('record:row-v1:sha256:', ph.hash_rows(expressions, version=1).chash.sha2_256())


def source_revisions(raw):
    """Small parity-test helper; production extraction keeps this as an expression."""
    return dict(raw.select(oid(C('id')).alias('id'), source_revision_expr(raw).alias('revision')).iter_rows())


def _coerce(table, df):
    return df.select([field(df, name, dtype=dtype).alias(name) for name, dtype in TABLES[table].columns.items()])


def _explode(df, name, index):
    return (df.with_columns(pl.int_ranges(0, C(name).list.len(), dtype=pl.Int64).alias(index))
            .filter(C(name).list.len() > 0).explode([name, index], empty_as_null=False))


def _ids_value(df, key):
    dtype = df.schema.get('ids', pl.Null)
    if isinstance(dtype, pl.Struct):
        return field(df, 'ids', key)
    if isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct):
        inner = dict(dtype.inner)
        if inner.get('key') != pl.String or inner.get('value') != pl.String:
            raise ContractError('ids entries must be {key: text, value: text}')
        e = pl.element()
        return C('ids').list.eval(e.struct.field('value').filter(e.struct.field('key') == key)).list.first()
    if dtype == pl.Null or (isinstance(dtype, pl.List) and dtype.inner == pl.Null):
        return L(None, pl.String)
    raise ContractError(f'Unsupported ids type {dtype}')


# Exact regular grammar for the supported JSON object: string -> list[int >= 0].
# Validate the WHOLE object before extracting, so malformed tails/negative values
# cannot silently disappear. Duplicate keys and their original order are retained.
_WS = r'[ \t\r\n]*'
_STRING = r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
_INT = r'(?:-?0|[1-9][0-9]*)'
_VALUES = rf'(?:{_INT}{_WS}(?:,{_WS}{_INT}{_WS})*)?'
_PAIR = rf'({_STRING}){_WS}:{_WS}\[({_WS}{_VALUES})\]'
_OBJECT = rf'^{_WS}\{{{_WS}(?:{_PAIR}{_WS}(?:,{_WS}{_PAIR}{_WS})*)?\}}{_WS}$'


def abstracts(base):
    df = base.select('_w', '_cid', field(base, 'abstract_inverted_index').alias('_json'))
    absent = C('_json').is_null() | (C('_json') == '') | (C('_json').str.strip_chars() == 'null')
    grammar_bad = df.filter(~absent & ~C('_json').str.contains(_OBJECT)).select('_w', '_cid')
    df = df.filter(~absent).join(grammar_bad, on='_w', how='anti')
    # Keep only the extracted pairs: carrying the whole JSON text into the exploded
    # frame copied every abstract once per word pair (measured +60 GiB on one file).
    pairs = _explode(df.select('_w', C('_json').str.extract_all(_PAIR).alias('_pairs')), '_pairs', '_pair_order')
    words = (pairs.select('_w', '_pair_order',
                         C('_pairs').str.extract(_PAIR, 1).str.json_decode(pl.String).alias('_word'),
                         C('_pairs').str.extract(_PAIR, 2).str.extract_all(r'-?[0-9]+').alias('_positions'))
             .filter(C('_positions').list.len() > 0).explode('_positions', empty_as_null=False)
             .with_columns(C('_positions').cast(pl.Int64, strict=False).alias('_position')))
    overflow = words.filter(C('_position').is_null()).select('_w').unique()  # beyond Int64
    words = words.join(overflow, on='_w', how='anti').sort('_w', '_position', '_pair_order')
    bad = pl.concat([grammar_bad.select('_w'), overflow]).unique()
    result = (words.group_by('_w').agg(C('_word').str.join(' ').alias('abstract'))
              .with_columns(pl.when(C('abstract') != '').then(C('abstract')).alias('abstract')))
    return result, bad


class _Native:
    def __init__(self, raw):
        if any(c.startswith('_') for c in raw.columns):
            raise ContractError('Input uses reserved internal column names')
        self.parts = {name: [] for name in TABLES}
        self.notes = []
        self.base = raw.with_columns(
            oid(field(raw, 'id')).alias('_cid'), source_revision_expr(raw).alias('_rev'),
            timestamp(raw, 'updated_date').alias('_when'))
        if self.base['_cid'].null_count():
            raise ContractError('Invalid/missing content ID')
        if self.base.filter(~C('_cid').str.contains(r'^openalex:W[0-9]+$')).height:
            raise ContractError('Unsupported work identifier (expected openalex:W<number>)')
        if 'updated_date' in raw.columns and self.base.filter(C('updated_date').is_not_null() & C('_when').is_null()).height:
            raise ContractError('Invalid source update timestamp')
        conflicts = self.base.group_by('_cid').agg(C('_rev').n_unique().alias('n')).filter(C('n') > 1)
        if conflicts.height:
            raise ContractError('Select one source revision per work before row extraction')
        self.base = self.base.unique('_cid', keep='first', maintain_order=True).with_row_index('_w')

    def note(self, df, code, path, detail):
        if not df.height:
            return
        # Count everything, keep only a few examples: never millions of Python messages.
        examples = df.select(C('_cid').alias('content_id'), path.alias('source_field')).head(5).to_dicts()
        self.notes.append(dict(code=code, count=df.height, detail=detail, examples=examples))

    def note_rows(self, df, code, detail):
        if df.height:
            examples = df.select(C('source_record').alias('content_id'), 'source_field').head(5).to_dicts()
            self.notes.append(dict(code=code, count=df.height, detail=detail, examples=examples))

    def link(self, table, df, relation, path, **fields):
        if not df.height:
            return
        r = L(relation) if isinstance(relation, str) else relation
        df = df.with_columns(C('_cid').alias('source_record'), C('_rev').alias('source_revision'),
                             path.alias('source_field'), r.alias('relation'), L(METHOD).alias('method'),
                             C('_when').alias('observed_at'), **fields)
        df = _coerce(table, df.with_columns(L(None, pl.String).alias('link_id')))
        invalid = pl.any_horizontal([C(c).is_null() for c in ENDPOINTS[table]])
        self.note_rows(df.filter(invalid), 'invalid_identifier', f'{table}: link with an unrecognized identifier dropped')
        df = df.filter(~invalid)
        if df.height:
            self.parts[table].append(df.with_columns(expr_link_key(table).alias('link_id')))

    def catalog(self, table, df, raw_id, ident, path, **fields):
        if not df.height:
            return
        df = df.with_columns(raw_id.alias('source_id'), ident.alias(TABLES[table].key),
                             C('_cid').alias('source_record'), C('_rev').alias('source_revision'),
                             path.alias('source_field'), C('_when').alias('observed_at'), **fields)
        df = _coerce(table, df.filter(nonempty('source_id')))
        invalid = C(TABLES[table].key).is_null()
        self.note_rows(df.filter(invalid), 'invalid_identifier', f'{table}: unrecognized identifier not catalogued')
        df = df.filter(~invalid)
        if df.height:
            self.parts[table].append(df)

    def entity(self, df, raw_id, path, kind, name=None, orcid=None, ror=None, country=None, otype=None):
        none = L(None, pl.String)
        self.catalog('entities', df, raw_id, oid(raw_id), path, entity_type=L(kind),
                     name=none if name is None else name, orcid=none if orcid is None else orcid,
                     ror=none if ror is None else ror, country_code=none if country is None else country,
                     organization_type=none if otype is None else otype)

    def classification(self, df, raw_id, ident, path, kind, scheme, name, notation=None):
        self.catalog('classifications', df, raw_id, ident, path, classification_type=kind if isinstance(kind, pl.Expr) else L(kind),
                     scheme=L(scheme), name=name, notation=L(None, pl.String) if notation is None else notation)

    def people(self):
        b = self.base
        authors = _explode(b.select('_cid', '_rev', '_when', '_w', array(b, 'authorships').alias('_a')), '_a', '_ai')
        if not authors.height:
            return
        a = authors.with_columns(cat('authorships[', C('_ai').cast(pl.String), ']').alias('_ap'))
        a = a.with_columns(
            field(a, '_a', 'raw_author_name').alias('author_text'),
            field(a, '_a', 'raw_orcid').alias('raw_orcid'),
            field(a, '_a', 'author_position').alias('author_position'),
            field(a, '_a', 'is_corresponding', dtype=pl.Boolean).alias('is_corresponding'),
            field(a, '_a', 'author', 'id').alias('_person_raw'),
            array(a, '_a', 'affiliations').alias('_mapped'),
            array(a, '_a', 'raw_affiliation_strings').alias('_fallback'),
            array(a, '_a', 'institutions').alias('_institutions'))
        self.entity(a, C('_person_raw'), cat(C('_ap'), '.author'), 'person',
                    name=field(a, '_a', 'author', 'display_name'), orcid=field(a, '_a', 'author', 'orcid'),
                    ror=field(a, '_a', 'author', 'ror'))
        inst = _explode(a.select('_cid', '_rev', '_when', '_ai', '_ap', '_institutions'), '_institutions', '_ii')
        inst = inst.with_columns(field(inst, '_institutions', 'id').alias('_org_raw'),
                                 field(inst, '_institutions', 'display_name').alias('_org_name'),
                                 field(inst, '_institutions', 'ror').alias('_org_ror'),
                                 field(inst, '_institutions', 'country_code').alias('_org_country'),
                                 field(inst, '_institutions', 'type').alias('_org_type'))
        org_fields = dict(name=C('_org_name'), ror=C('_org_ror'), country=C('_org_country'), otype=C('_org_type'))
        self.entity(inst, C('_org_raw'), cat(C('_ap'), '.institutions[', C('_ii').cast(pl.String), ']'),
                    'organization', **org_fields)
        common = ['_cid', '_rev', '_when', '_ai', '_ap', '_person_raw', 'author_text',
                  'raw_orcid', 'author_position', 'is_corresponding']
        mapped = _explode(a.filter(C('_mapped').list.len() > 0).select(*common, '_mapped'), '_mapped', '_fi')
        mapped = mapped.select(*common, (C('_fi') + 1).alias('affiliation_order'),
                                field(mapped, '_mapped', 'raw_affiliation_string').alias('affiliation_text'),
                                cat(C('_ap'), '.affiliations[', C('_fi').cast(pl.String), '].raw_affiliation_string').alias('_fp'),
                                cat(C('_ap'), '.affiliations[', C('_fi').cast(pl.String), '].institution_ids').alias('_ip'),
                                array(mapped, '_mapped', 'institution_ids').cast(pl.List(pl.String)).alias('_org_ids'))
        fallback = _explode(a.filter(C('_mapped').list.len() == 0).select(*common, '_fallback'), '_fallback', '_fi')
        self.note(fallback, 'raw_affiliation_fallback', C('_ap'), 'Raw strings retained; no guessed institution matches')
        fallback = fallback.select(*common, (C('_fi') + 1).alias('affiliation_order'),
                                    C('_fallback').cast(pl.String).alias('affiliation_text'),
                                    cat(C('_ap'), '.raw_affiliation_strings[', C('_fi').cast(pl.String), ']').alias('_fp'),
                                    L(None, pl.String).alias('_ip'), L([], pl.List(pl.String)).alias('_org_ids'))
        missing = a.filter((C('_mapped').list.len() == 0) & (C('_fallback').list.len() == 0)).select(
            *common, L(None, pl.Int64).alias('affiliation_order'), L(None, pl.String).alias('affiliation_text'),
            L(None, pl.String).alias('_fp'), L(None, pl.String).alias('_ip'), L([], pl.List(pl.String)).alias('_org_ids'))
        pairs = pl.concat([mapped, fallback, missing], how='vertical_relaxed').with_columns(
            expr_key('signature:v2:', C('_cid'), C('_rev'), C('_ap'), C('_fp')).alias('signature_id'))
        sigs = pairs.with_columns((C('_ai') + 1).alias('author_order'), C('_cid').alias('source_record'),
                                   C('_rev').alias('source_revision'), cat(C('_ap'), '.raw_author_name').alias('author_source_field'),
                                   C('_fp').alias('affiliation_source_field'))
        self.parts['signatures'].append(_coerce('signatures', sigs))
        self.link('links_content_signatures', pairs, 'has_signature', pl.coalesce('_fp', '_ap'),
                  content_id=C('_cid'), signature_id=C('signature_id'))
        self.link('links_signatures_entities', pairs.filter(nonempty('_person_raw')),
                  'identifies_person', cat(C('_ap'), '.author.id'),
                  signature_id=C('signature_id'), entity_id=oid(C('_person_raw')))
        orgs = (pairs.filter(C('_org_ids').list.len() > 0).explode('_org_ids', empty_as_null=False)
                .rename({'_org_ids': '_org_raw'}).unique(['signature_id', '_org_raw'], keep='first', maintain_order=True))
        lookup = (inst.filter(nonempty('_org_raw'))
                  .unique(['_cid', '_ai', '_org_raw'], keep='last', maintain_order=True)
                  .select('_cid', '_ai', '_org_raw', '_org_name', '_org_ror', '_org_country', '_org_type'))
        orgs = orgs.join(lookup, on=['_cid', '_ai', '_org_raw'], how='left')
        self.entity(orgs, C('_org_raw'), C('_ip'), 'organization', **org_fields)
        self.link('links_signatures_entities', orgs, 'identifies_institution', C('_ip'),
                  signature_id=C('signature_id'), entity_id=oid(C('_org_raw')))
        assigned = orgs.select('_cid', '_ai').unique()
        aggregate_only = inst.filter(C('_org_raw').is_not_null()).join(assigned, on=['_cid', '_ai'], how='anti').unique(['_cid', '_ai'])
        self.note(aggregate_only, 'aggregate_institutions_only', cat(C('_ap'), '.institutions'), 'Aggregate organizations retained, no unsupported signature matches')

    def subjects(self):
        b = self.base
        topics = _explode(b.select('_cid', '_rev', '_when', array(b, 'topics').alias('_topic')), '_topic', '_ti')
        topics = topics.with_columns(cat('topics[', C('_ti').cast(pl.String), ']').alias('_tp'))
        primary = b.select('_cid', '_rev', '_when', field(b, 'primary_topic', missing=pl.Struct({'id': pl.String})).alias('_topic'))
        primary = primary.with_columns(field(primary, '_topic', 'id').alias('_primary_id'))
        pids = primary.select('_cid', '_primary_id')
        # Struct schemas can differ in hand-written fixtures: flatten each branch before concat.
        def flatten_topics(t):
            return t.select('_cid', '_rev', '_when', '_tp',
                            field(t, '_topic', 'id').alias('_tid'), field(t, '_topic', 'display_name').alias('_name'),
                            field(t, '_topic', 'score', dtype=pl.Float64).alias('_score'),
                            *[field(t, '_topic', k, v).alias(f'_{k}_{v}')
                              for k in ('subfield', 'field', 'domain') for v in ('id', 'display_name')])
        topics = pl.concat([flatten_topics(topics), flatten_topics(primary.with_columns(L('primary_topic').alias('_tp')))])
        topics = topics.filter(nonempty('_tid')).unique(['_cid', '_tid'], keep='first', maintain_order=True).join(pids, on='_cid', how='left')
        self.classification(topics, C('_tid'), oid(C('_tid')), C('_tp'), 'topic', 'openalex.topics', C('_name'))
        self.link('links_content_classifications', topics, 'about', C('_tp'), content_id=C('_cid'), classification_id=oid(C('_tid')),
                  score=C('_score'), score_kind=pl.when(C('_score').is_not_null()).then(L('openalex_topic_score')),
                  is_primary=pl.when(nonempty('_primary_id')).then(C('_tid') == C('_primary_id')))
        child = '_tid'
        for level in ('subfield', 'field', 'domain'):
            ident = f'_{level}_id'
            self.classification(topics, C(ident), oid(C(ident)), cat(C('_tp'), '.' + level), level, 'openalex.topics',
                                C(f'_{level}_display_name'))
            edges = topics.filter(nonempty(child) & nonempty(ident))
            self.link('links_classifications_classifications', edges, 'broader', cat(C('_tp'), '.' + level),
                      from_classification_id=oid(C(child)), to_classification_id=oid(C(ident)))
            child = ident
        kw = _explode(b.select('_cid', '_rev', '_when', array(b, 'keywords').alias('_k')), '_k', '_ki')
        kw = kw.with_columns(field(kw, '_k', 'id').alias('_kid'), field(kw, '_k', 'score', dtype=pl.Float64).alias('_score'),
                             cat('keywords[', C('_ki').cast(pl.String), ']').alias('_kp'))
        self.note(kw.filter(~nonempty('_kid')), 'keyword_without_id', C('_kp'), 'No name-derived ID invented')
        kw = kw.filter(nonempty('_kid'))
        self.classification(kw, C('_kid'), oid(C('_kid')), C('_kp'), 'keyword', 'openalex.keywords', field(kw, '_k', 'display_name'))
        self.link('links_content_classifications', kw, 'about', C('_kp'), content_id=C('_cid'), classification_id=oid(C('_kid')),
                  score=C('_score'), score_kind=pl.when(C('_score').is_not_null()).then(L('openalex_keyword_score')))
        self.mesh()
        self.sdgs()
        self.note(b.filter(array(b, 'concepts').list.len() > 0), 'deferred_scheme', L('concepts'),
                  'Deprecated OpenAlex concepts not extracted; retained in raw input')

    def mesh(self):
        b = self.base
        m = _explode(b.select('_cid', '_rev', '_when', array(b, 'mesh').alias('_m')), '_m', '_mi')
        if not m.height:
            return
        m = m.with_columns(field(m, '_m', 'descriptor_ui').alias('_d'), field(m, '_m', 'qualifier_ui').alias('_q'),
                           field(m, '_m', 'descriptor_name').alias('_dn'), field(m, '_m', 'qualifier_name').alias('_qn'),
                           field(m, '_m', 'is_major_topic', dtype=pl.Boolean).alias('_major'))
        self.note(m.filter(~nonempty('_d')), 'mesh_without_descriptor', cat('mesh[', C('_mi').cast(pl.String), ']'),
                  'Entry without descriptor ignored')
        m = m.filter(nonempty('_d'))
        shape = C('_d').str.contains(r'^[A-Z][0-9]+$') & (C('_q').is_null() | C('_q').str.contains(r'^Q[0-9]+$'))
        self.note(m.filter(~shape), 'mesh_malformed_identifier', cat('mesh[', C('_mi').cast(pl.String), ']'),
                  'Entry with a malformed MeSH identifier ignored')
        m = m.filter(shape)
        self.note(m.filter(~C('_d').str.slice(0, 1).is_in(['D', 'C'])), 'mesh_unexpected_record_type',
                  cat('mesh[', C('_mi').cast(pl.String), ']'), 'Kept as mesh_record; not a descriptor or supplementary concept')
        entry = _canonical(C('_m'), m.schema['_m']).struct.json_encode() if isinstance(m.schema['_m'], pl.Struct) else C('_d')
        dup = m.with_columns(entry.alias('_entry')).filter(pl.int_range(pl.len()).over('_cid', '_entry') > 0)
        self.note(dup, 'mesh_duplicate_entry', cat('mesh[', C('_mi').cast(pl.String), ']'), 'Exact duplicate MeSH entry collapsed')
        m = m.sort('_cid', '_mi')
        d = (m.group_by('_cid', '_rev', '_when', '_d', maintain_order=True)
             .agg(C('_mi').first().alias('_i'), C('_dn').first().alias('_name'), C('_major').any(ignore_nulls=False).alias('_primary'))
             .with_columns(cat('mesh[', C('_i').cast(pl.String), '].descriptor_ui').alias('_path'),
                           cat('mesh:', C('_d')).alias('_id')))
        kind = (pl.when(C('_d').str.starts_with('D')).then(L('mesh_descriptor'))
                .when(C('_d').str.starts_with('C')).then(L('mesh_supplementary_concept')).otherwise(L('mesh_record')))
        self.classification(d, C('_d'), C('_id'), C('_path'), kind, 'nlm.mesh', C('_name'), C('_d'))
        self.link('links_content_classifications', d, 'about', C('_path'), content_id=C('_cid'),
                  classification_id=C('_id'), is_primary=C('_primary'))
        h = (m.filter(C('_q').is_not_null()).group_by('_cid', '_rev', '_when', '_d', '_q', maintain_order=True)
             .agg(C('_mi').first().alias('_i'), C('_dn').first().alias('_dname'), C('_qn').first().alias('_qname'),
                  C('_major').any(ignore_nulls=False).alias('_primary'))
             .with_columns(cat('mesh[', C('_i').cast(pl.String), ']').alias('_path'),
                           cat(C('_d'), '/', C('_q')).alias('_code'), cat('mesh:', C('_d'), '/', C('_q')).alias('_id'),
                           cat(C('_dname'), '/', C('_qname')).alias('_name')))
        self.classification(h, C('_code'), C('_id'), C('_path'), 'mesh_heading', 'nlm.mesh', C('_name'), C('_code'))
        self.link('links_content_classifications', h, 'about', C('_path'), content_id=C('_cid'),
                  classification_id=C('_id'), is_primary=C('_primary'))
        self.link('links_classifications_classifications', h, 'broader', C('_path'),
                  from_classification_id=C('_id'), to_classification_id=cat('mesh:', C('_d')))

    def sdgs(self):
        b = self.base
        s = _explode(b.select('_cid', '_rev', '_when', array(b, 'sustainable_development_goals').alias('_s')), '_s', '_si')
        if not s.height:
            return
        s = s.with_columns(field(s, '_s', 'id').alias('_sid'), field(s, '_s', 'display_name').alias('_name'),
                           field(s, '_s', 'score', dtype=pl.Float64).alias('_score'),
                           cat('sustainable_development_goals[', C('_si').cast(pl.String), ']').alias('_path'))
        self.note(s.filter(~nonempty('_sid')), 'sdg_without_id', C('_path'), 'No name-derived ID invented')
        s = s.filter(nonempty('_sid')).unique(['_cid', '_sid'], keep='first', maintain_order=True)
        s = s.with_columns(C('_sid').str.extract(r'^https?://metadata\.un\.org/sdg/([0-9]+)$').alias('_n'))
        self.note(s.filter(C('_n').is_null()), 'invalid_identifier', C('_path'), 'Unrecognized SDG identifier dropped')
        s = s.filter(C('_n').is_not_null())
        s = s.with_columns(cat('unsdg:', C('_n')).alias('_id'))
        self.classification(s, C('_sid'), C('_id'), C('_path'), 'sdg', 'un.sdg', C('_name'), C('_n'))
        self.link('links_content_classifications', s, 'about', C('_path'), content_id=C('_cid'), classification_id=C('_id'),
                  score=C('_score'), score_kind=pl.when(C('_score').is_not_null()).then(L('openalex_sdg_score')))

    def funding(self):
        b = self.base
        f = _explode(b.select('_cid', '_rev', '_when', array(b, 'funders').alias('_f')), '_f', '_fi')
        if f.height:
            f = f.with_columns(field(f, '_f', 'id').alias('_fid'), cat('funders[', C('_fi').cast(pl.String), ']').alias('_path'))
            self.note(f.filter(~nonempty('_fid')), 'funder_without_id', C('_path'), 'No name-derived ID invented')
            f = f.filter(nonempty('_fid')).unique(['_cid', '_fid'], keep='first', maintain_order=True)
            self.entity(f, C('_fid'), C('_path'), 'organization', name=field(f, '_f', 'display_name'), ror=field(f, '_f', 'ror'))
            self.link('links_content_entities', f, 'funded_by', C('_path'), content_id=C('_cid'), entity_id=oid(C('_fid')))
        a = _explode(b.select('_cid', '_rev', '_when', array(b, 'awards').alias('_g')), '_g', '_gi')
        if not a.height:
            return
        a = a.with_columns(field(a, '_g', 'id').alias('_gid'), field(a, '_g', 'funder_id').alias('_funder'),
                           cat('awards[', C('_gi').cast(pl.String), ']').alias('_path'))
        self.note(a.filter(~nonempty('_gid')), 'award_without_id', C('_path'), 'Award without identifier not extracted')
        a = a.filter(nonempty('_gid')).unique(['_cid', '_gid'], keep='first', maintain_order=True)
        self.classification(a, C('_gid'), oid(C('_gid')), C('_path'), 'award', 'openalex.awards',
                            field(a, '_g', 'display_name'), field(a, '_g', 'funder_award_id'))
        self.link('links_content_classifications', a, 'funded_by', C('_path'), content_id=C('_cid'), classification_id=oid(C('_gid')))
        af = a.filter(nonempty('_funder')).with_columns(cat(C('_path'), '.funder_id').alias('_fpath'))
        self.entity(af, C('_funder'), C('_fpath'), 'organization', name=field(af, '_g', 'funder_display_name'))
        self.link('links_classifications_entities', af, 'funded_by', C('_fpath'),
                  classification_id=oid(C('_gid')), entity_id=oid(C('_funder')))

    def locations(self):
        b = self.base
        locations = _explode(b.select('_cid', '_rev', '_when', array(b, 'locations').alias('_loc')), '_loc', '_li')
        locations = locations.with_columns(cat('locations[', C('_li').cast(pl.String), ']').alias('_lp'))
        p = b.select('_cid', '_rev', '_when', field(b, 'primary_location', missing=pl.Struct({'id': pl.String})).alias('_loc'))
        def flatten_locs(df):
            dtype = df.schema['_loc']
            whole = _canonical(C('_loc'), dtype).struct.json_encode() if isinstance(dtype, pl.Struct) else L(None, pl.String)
            return df.select('_cid', '_rev', '_when', '_lp',
                             pl.coalesce(field(df, '_loc', 'id'), whole).alias('_lk'),
                             field(df, '_loc', 'id').alias('source_location_id'),
                             field(df, '_loc', 'source', 'id').alias('_src'),
                             field(df, '_loc', 'source', 'display_name').alias('_name'),
                             field(df, '_loc', 'source', 'type').alias('_kind'),
                             field(df, '_loc', 'source', 'issn_l').alias('_issn'),
                             field(df, '_loc', 'source', 'host_organization').alias('_host'),
                             field(df, '_loc', 'source', 'host_organization_name').alias('_host_name'),
                             *[field(df, '_loc', k).alias(k) for k in ('raw_source_name', 'landing_page_url', 'pdf_url', 'version', 'license')],
                             field(df, '_loc', 'is_oa', dtype=pl.Boolean).alias('is_oa'),
                             field(df, '_loc', 'is_published', dtype=pl.Boolean).alias('_published'))
        locations = flatten_locs(locations)
        primary = flatten_locs(p.with_columns(L('primary_location').alias('_lp'))).filter(C('_lk').is_not_null())
        pkeys = primary.select('_cid', C('_lk').alias('_pk'))
        locations = pl.concat([locations, primary.join(locations.select('_cid', '_lk'), on=['_cid', '_lk'], how='anti')])
        locations = locations.join(pkeys, on='_cid', how='left')
        self.note(locations.filter(~nonempty('_src')), 'location_without_source_id', C('_lp'), 'No venue classification invented')
        locations = locations.filter(nonempty('_src')).with_columns(
            pl.when(nonempty('_kind')).then(C('_kind')).otherwise(L('source')).alias('_kind'))
        self.classification(locations, C('_src'), oid(C('_src')), cat(C('_lp'), '.source'), C('_kind'), 'openalex.sources',
                            C('_name'), C('_issn'))
        conflict = C('_published').is_not_null() & C('version').is_in(KNOWN_VERSIONS) & (C('_published') != (C('version') == 'publishedVersion'))
        self.note(locations.filter(conflict), 'location_version_conflict', C('_lp'), 'Conflicting publication flags; no publication claim inferred')
        confirmed = ((C('_published') & (C('version').is_null() | (C('version') == 'publishedVersion')))
                     | (C('_published').is_null() & (C('version') == 'publishedVersion'))).fill_null(False)
        relation = pl.when(C('_kind').is_in(PUBLICATION_VENUES) & confirmed).then(L('published_in')).otherwise(L('hosted_by'))
        self.link('links_content_classifications', locations, relation, C('_lp'),
                  content_id=C('_cid'), classification_id=oid(C('_src')),
                  is_primary=pl.when(C('_pk').is_not_null()).then(C('_lk') == C('_pk')),
                  **{k: C(k) for k in ('source_location_id', 'raw_source_name', 'landing_page_url', 'pdf_url', 'version', 'license', 'is_oa')})
        hosts = locations.filter(nonempty('_host')).with_columns(cat(C('_lp'), '.source.host_organization').alias('_hp'))
        self.entity(hosts, C('_host'), C('_hp'), 'organization', name=C('_host_name'))
        self.link('links_classifications_entities', hosts, 'host_organization', C('_hp'),
                  classification_id=oid(C('_src')), entity_id=oid(C('_host')))

    def content_and_references(self):
        b = self.base
        for name in ('publication_year', 'cited_by_count', 'referenced_works_count'):
            dtype = b.schema.get(name, pl.Null)
            if dtype != pl.Null and not dtype.is_integer():
                raise ContractError(f'{name} must contain integers, not {dtype}')
        fwci = field(b, 'fwci', dtype=pl.Float64)
        self.note(b.filter(fwci.is_not_null() & ~fwci.is_finite()), 'nonfinite_fwci', L('fwci'), 'Nonfinite FWCI stored as unknown')
        texts, bad_abstracts = abstracts(b)
        self.note(b.join(bad_abstracts, on='_w', how='semi'), 'invalid_abstract', L('abstract_inverted_index'),
                  'Malformed inverted index: abstract stored as unknown, no fragments kept')
        content = b.join(texts, on='_w', how='left').with_columns(
            C('_cid').alias('content_id'), C('id').alias('source_id'), C('_rev').alias('source_revision'),
            field(b, 'publication_date', dtype=pl.Date).alias('date'),
            field(b, 'publication_year', dtype=pl.Int64).alias('year'),
            field(b, 'cited_by_count', dtype=pl.Int64).alias('citations'),
            field(b, 'referenced_works_count', dtype=pl.Int64).alias('references_count'), C('_when').alias('updated_at'),
            *[field(b, 'biblio', k).alias(k) for k in ('volume', 'issue', 'first_page', 'last_page')],
            field(b, 'open_access', 'is_oa', dtype=pl.Boolean).alias('is_oa'),
            field(b, 'open_access', 'oa_status').alias('oa_status'), field(b, 'open_access', 'oa_url').alias('oa_url'),
            pl.when(fwci.is_finite()).then(fwci).alias('fwci'),
            *[_ids_value(b, k).alias(k) for k in ('pmid', 'pmcid', 'mag')])
        content = content.with_columns(((field(content, 'language') == 'en').fill_null(False) & C('abstract').is_not_null()
                                         & (array(b, 'referenced_works').list.len() > 0)).alias('is_complete'))
        if content.filter((C('citations') < 0) | (C('references_count') < 0)).height:
            raise ContractError('Citation/reference counts cannot be negative')
        self.parts['content'].append(_coerce('content', content))
        refs = _explode(b.select('_cid', '_rev', '_when', array(b, 'referenced_works').alias('_refs')), '_refs', '_ri')
        refs = refs.with_columns(oid(C('_refs').cast(pl.String)).alias('_target')).unique(['_cid', '_target'], keep='first', maintain_order=True)
        self.link('links_content_content', refs, 'cites', cat('referenced_works[', C('_ri').cast(pl.String), ']'),
                  from_content_id=C('_cid'), to_content_id=C('_target'))
        counts = refs.group_by('_cid').len(name='_actual')
        diff = b.join(counts, on='_cid', how='left').filter(
            field(b, 'referenced_works_count', dtype=pl.Int64).is_not_null()
            & (field(b, 'referenced_works_count', dtype=pl.Int64) != C('_actual').fill_null(0)))
        self.note(diff, 'reference_count_differs', L('referenced_works'), 'Reported reference count differs from distinct extracted targets')

    def finish(self):
        tables = {}
        for name, contract in TABLES.items():
            parts = self.parts[name]
            df = pl.concat(parts) if parts else pl.DataFrame(schema=contract.columns)
            df = reduce_frame(name, df, self.notes) if name in REDUCED else df.unique(contract.key, keep='first', maintain_order=True)
            if df.select(pl.any_horizontal([C(c).is_null() for c in contract.required]).any()).item():
                raise ContractError(f'{name}: missing/invalid required value or ID')
            if 'score' in df.columns and df.filter(C('score').is_not_null() & ~C('score').is_finite()).height:
                raise ContractError(f'{name}: scores must be finite or null')
            tables[name] = df
        return Result(tables, self.notes)


def extract(raw, *, timings=None):
    """Extract an already-version-selected source DataFrame, entirely in RAM.

    Logical row order is unspecified. Source schema is preserved across slices;
    never re-infer it from each batch. No file IO or publication happens here.
    """
    if not raw.height:
        return Result({name: pl.DataFrame(schema=t.columns) for name, t in TABLES.items()}, [])
    from time import perf_counter
    started = perf_counter()
    n = _Native(raw)
    if timings is not None:
        timings['prepare'] = perf_counter() - started
    for name, operation in [('content_and_references', n.content_and_references),
                            ('signatures_and_entities', n.people), ('classifications', n.subjects),
                            ('funding', n.funding), ('locations', n.locations)]:
        started = perf_counter()
        operation()
        if timings is not None:
            timings[name] = perf_counter() - started
    started = perf_counter()
    result = n.finish()
    if timings is not None:
        timings['reduce_and_keys'] = perf_counter() - started
    return result
