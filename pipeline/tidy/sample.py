"""Export the tested row model on a small fixture, without touching production.

From the repository root:
  uv run --with-requirements pipeline/tidy/requirements.txt python -m pipeline.tidy.sample
"""
import argparse
import json
from pathlib import Path

import polars as pl

from .contracts import SCHEMA_VERSION, TABLES, validate
from .reference import extract as reference_extract
from . import native

ROOT = Path(__file__).resolve().parents[2]


def preview(value, limit=95):
    if value is None:
        return '—'
    text = str(value).replace('\n', ' ↵ ').replace('|', '\\|')
    return text if len(text) <= limit else text[:limit] + '…'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'research/openalex/schema-review-examples.parquet')
    parser.add_argument('--out', type=Path, default=ROOT / 'research/openalex/tidy-row-example')
    parser.add_argument('--engine', choices=('native', 'reference'), default='native')
    args = parser.parse_args()
    count = pl.scan_parquet(args.input).select(pl.len().cast(pl.UInt64)).collect().item()
    if count > 200:
        parser.error(f'This reference example accepts at most 200 source records, not {count:,}; use a saved small fixture')
    raw = pl.read_parquet(args.input)
    result = native.extract(raw) if args.engine == 'native' else reference_extract(raw.to_dicts(), native.source_revisions(raw))
    check = validate(result.tables)
    args.out.mkdir(parents=True, exist_ok=True)
    schema_doc = ['# Executable table contracts', '', f'Version: `{SCHEMA_VERSION}`.',
                  'Generated from `pipeline/tidy/contracts.py`. This is the first row-model contract, not the full publisher-metadata projection.', '']
    for name, df in result.tables.items():
        df.write_parquet(args.out / (name + '.parquet'))
        df.write_csv(args.out / (name + '.csv'))
        c = TABLES[name]
        schema_doc += [f'## `{name}`', '', f'Primary key: `{c.key}`.', '',
                       '| Column | Type | Nullable |', '|---|---|---|']
        schema_doc += [f'| `{col}` | {dtype} | {"no" if col in c.required else "yes"} |'
                       for col, dtype in c.columns.items()]
        schema_doc.append('')
    (args.out / 'CONTRACTS.md').write_text('\n'.join(schema_doc))
    counts = {name: df.height for name, df in result.tables.items()}
    (args.out / 'checks.json').write_text(json.dumps(dict(schema_version=SCHEMA_VERSION, source=str(args.input),
                                                       engine=args.engine, rows=counts, validation=check, notes=result.notes), indent=2))
    report = ['# First row-model test: real OpenAlex papers', '',
              'These are the two saved source records, converted into the newly agreed tables. No full-corpus scan or production change.', '',
              '## Rows', '', '| Table | Rows |', '|---|---:|']
    report += [f'| `{name}` | {n} |' for name, n in counts.items()]
    report += ['', '## Content', '', '| Content ID | Year | Title | Citations |', '|---|---:|---|---:|']
    for r in result.tables['content'].sort('year').to_dicts():
        report.append(f"| {r['content_id']} | {r['year']} | {preview(r['title'], 200)} | {r['citations']} |")
    report += ['', '## Author–affiliation pairs', '',
               'Each row below is one signature. Long text is shortened **only in this display**; CSV and Parquet preserve the exact source strings. The source record and 1-based author/affiliation positions locate each occurrence.', '',
               '| Content | Author order | Affiliation order | Raw author | Raw affiliation (preview) |', '|---|---:|---:|---|---|']
    for r in result.tables['signatures'].sort('source_record', 'author_order', 'affiliation_order').to_dicts():
        report.append(f"| {r['source_record']} | {r['author_order']} | {r['affiliation_order']} | {preview(r['author_text'])} | {preview(r['affiliation_text'])} |")
    report += ['', '## Boundaries of this first test', '',
               '- Four unit tables and seven link tables; every cell is scalar.',
               '- Content includes bibliographic, open-access and identifier fields; the old flat table is unchanged.',
               '- Signatures are author–affiliation pairs, never venue or funding entries.',
               '- Topics and their hierarchy, keywords, sources and their hosts, MeSH, SDGs, awards and funders are represented. Deprecated OpenAlex concepts are listed in `checks.json` as deferred.',
               '- Source assignment confidence remains null when OpenAlex provides none.',
               '- Referenced works outside these two rows remain external IDs; no fabricated content rows.',
               '- Repeated entity/category profiles are selected by source update time, then a stable tie-break, with conflicting observations reported in `checks.json`.',
               '- Default extraction uses native Polars; `--engine reference` runs the independent Python oracle with the same source revisions.',
               '- This tiny example is for inspecting rows. Representative whole-file timings are in `research/openalex/tidy-pilot/README.md`.', '',
               'Exact schemas and nullability: [CONTRACTS.md](CONTRACTS.md). Counts, coverage notes, and endpoint checks: [checks.json](checks.json).']
    (args.out / 'EXAMPLE.md').write_text('\n'.join(report) + '\n')
    for name, n in counts.items():
        print(f'{name:45} {n:6,}')
    print(f'Validated. Example files: {args.out}')


if __name__ == '__main__':
    main()
