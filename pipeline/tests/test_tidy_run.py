"""End-to-end runner: versions across files, resumable jobs, reduction, publication."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from pipeline.tidy import native, run, dataset
from pipeline.tidy.contracts import TABLES
from pipeline.tests.test_tidy_rows import work, author, institution, topic, location, url

ARGS = ['--threads', '1', '--jobs', '2', '--buckets', '4', '--reserve-gib', '0.5', '--critical-gib', '0.25',
        '--output-ratio', '1', '--disk-reserve-gib', '0']


def corpus():
    works = []
    for i in range(1, 10):
        w = work('W' + str(i), [author('Name ' + str(i % 3), 'A' + str(i % 4), [('Inst', ['I1'])], [institution('I1', 'U' + str(i % 2))])])
        w['topics'] = [topic('T' + str(i % 3))]
        w['funders'] = [dict(id=url('F1'), display_name='Fund ' + str(i), ror=None)]
        w['updated_date'] = datetime(2024, 1, 1 + i, tzinfo=timezone.utc)
        works.append(w)
    newer = dict(works[2]); newer['title'] = 'Newer W3'; newer['updated_date'] = datetime(2025, 1, 1, tzinfo=timezone.utc)
    works.append(newer)
    # One shared schema for all files, so source revisions equal a single extraction.
    return pl.from_dicts(works, infer_schema_length=None)


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.df = corpus()
        self.layout = {'updated_date=2024-01-01/part_0000': (0, 4), 'updated_date=2024-01-01/part_0001': (4, 4),
                       'updated_date=2025-01-01/part_0000': (8, 2)}
        self.write_snapshot()

    def tearDown(self):
        self.tmp.cleanup()

    def write_snapshot(self, state='complete', only=None):
        files = []
        for rel, (start, n) in self.layout.items():
            path = self.root / 'works' / (rel + '.parquet')
            path.parent.mkdir(parents=True, exist_ok=True)
            if only is None or rel in only:
                self.df.slice(start, n).write_parquet(path)
            files.append(dict(url='s3://openalex/data/parquet/works/' + rel + '.parquet',
                              meta=dict(content_length=path.stat().st_size, record_count=n)))
        (self.root / 'manifest.parquet.json').write_text(json.dumps(dict(date='test', entities=[dict(entity='works', files=files)])))
        (self.root / 'status.json').write_text(json.dumps(dict(state=state)))

    def published(self):
        return {t: dataset.scan(t, root=self.root / 'tidy').collect() for t in TABLES}

    def expected(self):
        winners = self.df.filter(~((pl.col('id') == url('W3')) & (pl.col('title') != 'Newer W3')))
        return native.extract(winners).tables

    def assert_matches(self):
        got, want = self.published(), self.expected()
        for t, spec in TABLES.items():
            assert_frame_equal(got[t].sort(spec.key), want[t].sort(spec.key), check_exact=True)

    def test_end_to_end_resume_and_incremental_update(self):
        run.main(['all', '--root', str(self.root), *ARGS])
        self.assert_matches()
        content = self.published()['content']
        self.assertEqual(content.filter(pl.col('content_id') == 'openalex:W3')['title'].to_list(), ['Newer W3'])
        m = dataset.manifest(self.root / 'tidy')
        self.assertEqual(m['superseded_rows'], 1)
        self.assertTrue(all(v is True for k, v in m['checks'].items() if k != 'content_rows'))
        done = {p: p.stat().st_mtime_ns for p in (self.root / 'tidy' / 'extract').glob('*/*/done.json')}
        self.assertEqual(len(done), 3)
        first_generation = m['generation']

        run.main(['all', '--root', str(self.root), *ARGS])  # unchanged: no job reruns
        self.assertEqual({p: p.stat().st_mtime_ns for p in (self.root / 'tidy' / 'extract').glob('*/*/done.json')}, done)

        self.df = self.df.with_columns(pl.when(pl.col('id') == url('W5')).then(pl.lit('Edited W5'))
                                       .otherwise(pl.col('title')).alias('title'))
        self.write_snapshot(only={'updated_date=2024-01-01/part_0001'})
        run.main(['all', '--root', str(self.root), *ARGS])
        now = list((self.root / 'tidy' / 'extract').glob('*/*/done.json'))
        self.assertEqual(len(set(now) - set(done)), 1)  # only the edited file's job is new
        self.assert_matches()
        self.assertNotEqual(dataset.manifest(self.root / 'tidy')['generation'], first_generation)

    def test_refuses_incomplete_download(self):
        self.write_snapshot(state='running')
        with self.assertRaises(SystemExit):
            run.main(['all', '--root', str(self.root), *ARGS])
        (self.root / 'works' / 'updated_date=2024-01-01' / 'part_0001.parquet').unlink()
        self.write_snapshot()
        (self.root / 'works' / 'updated_date=2024-01-01' / 'part_0001.parquet').write_bytes(b'truncated')
        with self.assertRaises(SystemExit):
            run.main(['all', '--root', str(self.root), *ARGS])


if __name__ == '__main__':
    unittest.main()
