"""Read the currently published tidy tables.

    from pipeline.tidy.dataset import scan, manifest
    scan('signatures').filter(...)      # a lazy Polars frame over the CURRENT generation

Always resolve files through the generation manifest: never glob the store,
which also contains in-progress and superseded outputs.
"""
import json
import os
from pathlib import Path

import polars as pl

DEFAULT = Path(os.environ.get('OPENALEX_TIDY', '/mnt/bulk/openalex/tidy'))


def manifest(root=DEFAULT, generation=None):
    root = Path(root)
    generation = generation or (root / 'CURRENT').read_text().strip()
    return json.loads((root / 'generations' / generation / 'manifest.json').read_text())


def scan(table, root=DEFAULT, generation=None):
    files = manifest(root, generation)['tables'][table]['files']
    return pl.scan_parquet(files)


def tables(root=DEFAULT, generation=None):
    return list(manifest(root, generation)['tables'])
