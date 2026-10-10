"""Portable keys shared by the independent row oracle and native transforms.

Only string/null tuples are encoded, so number/date formatting never changes an
ID. Content-owned assertions are keyed by their source context. Vocabulary-level
links (hierarchy, award funder, source host) are keyed by endpoints only: they
are reduced to one row per edge across the dataset, with one recorded witness.
"""
from hashlib import sha256
import json

ENDPOINTS = {
    'links_content_signatures': ('content_id', 'signature_id'),
    'links_signatures_entities': ('signature_id', 'entity_id'),
    'links_content_classifications': ('content_id', 'classification_id'),
    'links_classifications_classifications': ('from_classification_id', 'to_classification_id'),
    'links_content_content': ('from_content_id', 'to_content_id'),
    'links_classifications_entities': ('classification_id', 'entity_id'),
    'links_content_entities': ('content_id', 'entity_id'),
}
VOCABULARY_LINKS = ('links_classifications_classifications', 'links_classifications_entities')


def key(prefix, *parts):
    if not all(p is None or isinstance(p, str) for p in parts):
        raise TypeError('Stable key parts must be strings or null')
    text = json.dumps({f'k{i}': p for i, p in enumerate(parts)}, ensure_ascii=False, separators=(',', ':'))
    return prefix + sha256(text.encode()).hexdigest()


def signature_key(content_id, revision, author_path, affiliation_path):
    return key('signature:v2:', content_id, revision, author_path, affiliation_path)


def link_key(table, row):
    if table in VOCABULARY_LINKS:
        return key('link:v3:', table, row['method'], row['relation'], *(row[c] for c in ENDPOINTS[table]))
    return key('link:v2:', table, row['source_record'], row['source_revision'],
               row['source_field'], row['method'], row['relation'],
               *(row[c] for c in ENDPOINTS[table]))


def expr_key(prefix, *parts):
    import polars as pl
    import polars_hash  # registers the native .chash namespace
    return pl.concat_str(pl.lit(prefix), pl.struct(
        [p.cast(pl.String).alias(f'k{i}') for i, p in enumerate(parts)]
    ).struct.json_encode().chash.sha2_256())


def expr_link_key(table):
    import polars as pl
    C, L = pl.col, pl.lit
    if table in VOCABULARY_LINKS:
        return expr_key('link:v3:', L(table), C('method'), C('relation'), *(C(c) for c in ENDPOINTS[table]))
    return expr_key('link:v2:', L(table), C('source_record'), C('source_revision'), C('source_field'),
                    C('method'), C('relation'), *(C(c) for c in ENDPOINTS[table]))
