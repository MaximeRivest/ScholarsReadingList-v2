"""One total order for choosing the representative observation of a reduced row.

Used identically inside a partition (native + reference) and across partitions,
so extracting files separately and then reducing equals extracting them together.

Order, highest wins:
  1. observed_at (source work's updated time; missing = earliest)
  2. completeness (count of populated descriptive properties)
  3. source work number (openalex:W123 -> 123)
  4. source_field (string; UTF-8 byte order == code-point order)
Within a current snapshot each work is observed once, so (3, 4) identify one
observation and the order is total.
"""
from datetime import datetime, timezone
import re

import polars as pl

from .contracts import TABLES, ContractError

PROPS = {
    'entities': ('entity_type', 'name', 'orcid', 'ror', 'country_code', 'organization_type'),
    'classifications': ('classification_type', 'scheme', 'name', 'notation'),
    'links_classifications_classifications': (),
    'links_classifications_entities': (),
}
EARLIEST = datetime(1, 1, 1, tzinfo=timezone.utc)
_WORK = re.compile(r'^openalex:W([0-9]+)$')
C = pl.col


def work_number(content_id):
    m = _WORK.match(content_id or '')
    if not m:
        raise ContractError(f'Unsupported source work identifier: {content_id!r}')
    return int(m.group(1))


def python_rank(table, row):
    return (row['observed_at'] or EARLIEST, sum(row[p] is not None for p in PROPS[table]),
            work_number(row['source_record']), row['source_field'])


def rank_exprs(table):
    props = PROPS[table]
    complete = pl.sum_horizontal([C(p).is_not_null().cast(pl.Int64) for p in props]) if props else pl.lit(0, pl.Int64)
    return [C('observed_at').fill_null(pl.lit(EARLIEST, dtype=TABLES[table].columns['observed_at'])).alias('_r_when'),
            complete.alias('_r_complete'),
            C('source_record').str.extract(r'^openalex:W([0-9]+)$').cast(pl.Int64).alias('_r_work'),
            C('source_field').alias('_r_field')]


def reduce_frame(table, df, notes=None):
    """Keep one row per key by RANK. Returns contract columns only."""
    key = TABLES[table].key
    cols = list(TABLES[table].columns)
    if not df.height:
        return df.select(cols)
    ranked = df.with_columns(rank_exprs(table))
    if ranked['_r_work'].null_count():
        raise ContractError(f'{table}: unsupported source_record identifier')
    if notes is not None and PROPS[table]:
        variation = ranked.group_by(key).agg(pl.struct(PROPS[table]).n_unique().alias('_v')).filter(C('_v') > 1)
        if variation.height:
            notes.append(dict(code='catalog_variation', table=table, count=variation.height,
                              detail='Several observed profiles; one chosen by the documented rank'))
    return (ranked.sort(['_r_when', '_r_complete', '_r_work', '_r_field'], descending=True)
            .unique(key, keep='first', maintain_order=True).select(cols))
