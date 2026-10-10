"""What the atlas maps: scholarly publications among the embedding plan's works.

The backfill embeds every English work with an abstract, whatever its OpenAlex type. In the
pilot sample, `dataset` alone was 28 % of all dots and 62–79 % of 2024–2026 (mostly machine-made
NIFS instrument records such as "LHD FMP #184398.1"), which would draw a fake explosion of
"science". The atlas therefore keeps publication types and drops data, front matter and notices.
"""
import json
from pathlib import Path

SCHOLARLY_TYPES = frozenset({
    'article', 'review', 'conference-paper', 'conference-abstract', 'preprint', 'book', 'book-chapter',
    'dissertation', 'report', 'letter', 'editorial', 'data-paper', 'software-paper'})
EXCLUDED_NOTE = ('dataset, other, paratext, supplementary-materials, software, book-review, reference-entry, '
                 'peer-review, libguides, erratum, standard, retraction')
TYPE_COUNTS = Path('/mnt/fast/science-atlas/sample/type_counts.json')


def counts_per_year(path=TYPE_COUNTS):
    """{year: (kept, all)} exact eligible counts from count_types.py."""
    doc = json.loads(Path(path).read_text())
    return {int(y): (sum(n for t, n in c.items() if t in SCHOLARLY_TYPES), sum(c.values())) for y, c in doc.items()}
