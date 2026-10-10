"""Run: uv run --with polars==1.44.2 python -m unittest discover -s pipeline/tests -v"""
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import polars as pl

from pipeline.tidy.contracts import ContractError, TABLES, frame, validate
from pipeline.tidy.reference import abstract_text, extract


def url(identifier):
    return 'https://openalex.org/' + identifier


def institution(identifier='I1', name='University One'):
    return dict(id=url(identifier), display_name=name, ror=None)


def author(name='Alice Smith', aid='A1', affiliations=(), institutions=()):
    return dict(raw_author_name=name, raw_orcid=None,
                author=dict(id=url(aid), display_name='Resolved ' + (name or 'name'), orcid=None),
                author_position='first', is_corresponding=True,
                affiliations=[dict(raw_affiliation_string=text, institution_ids=[url(x) for x in ids])
                              for text, ids in affiliations],
                raw_affiliation_strings=[text for text, _ in affiliations],
                institutions=list(institutions))


def work(identifier='W1', authors=()):
    return dict(id=url(identifier), title='A test article', type='article', language='en',
                publication_year=2024, publication_date='2024-01-02', updated_date='2024-02-01T00:00:00Z',
                cited_by_count=12, referenced_works_count=1, referenced_works=[url('W999')],
                abstract_inverted_index='{"Hello": [0], "world": [1]}',
                is_retracted=False, authorships=list(authors), topics=[], keywords=[], locations=[])


def topic(tid='T1', name='Neuroscience', sub='1'):
    return dict(id=url(tid), display_name=name, score=0.8,
                subfield=dict(id=url('subfields/' + sub), display_name='Subfield ' + sub),
                field=dict(id=url('fields/1'), display_name='Field'),
                domain=dict(id=url('domains/1'), display_name='Domain'))


def location(identifier, source='S1', kind='journal', version='publishedVersion', license='cc-by'):
    return dict(id=identifier, source=dict(id=url(source), display_name='Example source', type=kind),
                raw_source_name='Example src.', landing_page_url='https://example.org/' + identifier,
                pdf_url=None, version=version, license=license, is_oa=True)


class RowRules(unittest.TestCase):
    def test_empty_inputs_have_all_nine_scalar_schemas(self):
        result = extract([])
        self.assertEqual(set(result.tables), set(TABLES))
        for name, df in result.tables.items():
            self.assertEqual(df.height, 0)
            self.assertEqual(dict(df.schema), TABLES[name].columns)
            self.assertFalse(any(t.is_nested() or t == pl.Object for t in df.schema.values()))

    def test_superscript_pairings_not_cartesian_product(self):
        alice = author(affiliations=[('Toronto', ['I1']), ('McGill', ['I2'])],
                       institutions=[institution(), institution('I2', 'McGill')])
        bob = author('Bob Jones', 'A2', [('McGill', ['I2'])], [institution('I2', 'McGill')])
        bob['author_position'] = 'last'
        tables = extract([work(authors=[alice, bob])]).tables
        rows = tables['signatures'].sort('author_order', 'affiliation_order')
        self.assertEqual(rows.select('author_text', 'affiliation_text', 'author_order', 'affiliation_order').rows(),
                         [('Alice Smith', 'Toronto', 1, 1), ('Alice Smith', 'McGill', 1, 2), ('Bob Jones', 'McGill', 2, 1)])
        self.assertEqual(tables['links_content_signatures'].height, 3)
        self.assertEqual(tables['links_signatures_entities'].height, 6)
        self.assertEqual(tables['entities'].height, 4)
        self.assertNotIn('content_id', rows.columns)  # ownership is in its link table

    def test_author_without_affiliation_still_has_signature(self):
        t = extract([work(authors=[author()])]).tables
        s = t['signatures'].row(0, named=True)
        self.assertIsNone(s['affiliation_text'])
        self.assertIsNone(s['affiliation_order'])
        self.assertEqual(t['links_signatures_entities'].height, 1)

    def test_same_name_twice_on_same_work_is_two_occurrences(self):
        t = extract([work(authors=[author('J. Smith'), author('J. Smith', 'A2')])]).tables
        self.assertEqual(t['signatures']['signature_id'].n_unique(), 2)
        self.assertEqual(t['entities'].height, 2)

    def test_same_text_across_content_is_not_the_same_signature(self):
        a = author(affiliations=[('Same university', [])])
        t = extract([work('W1', [a]), work('W2', [a])]).tables
        self.assertEqual(t['signatures'].height, 2)
        self.assertEqual(t['entities'].height, 1)

    def test_one_affiliation_can_match_two_institutions(self):
        a = author(affiliations=[('University and Hospital', ['I1', 'I2'])],
                   institutions=[institution(), institution('I2', 'Hospital')])
        t = extract([work(authors=[a])]).tables
        self.assertEqual(t['signatures'].height, 1)
        self.assertEqual(t['links_signatures_entities'].filter(pl.col('relation') == 'identifies_institution').height, 2)

    def test_repeated_raw_affiliation_occurrences_are_not_collapsed(self):
        a = author(affiliations=[('Same text', ['I1']), ('Same text', ['I1'])])
        t = extract([work(authors=[a])]).tables
        self.assertEqual(t['signatures'].height, 2)
        self.assertEqual(set(t['signatures']['affiliation_order']), {1, 2})

    def test_duplicate_institution_ids_in_one_mapping_do_not_duplicate_links(self):
        a = author(affiliations=[('University', ['I1', 'I1'])])
        t = extract([work(authors=[a])]).tables
        self.assertEqual(t['links_signatures_entities'].height, 2)  # one person + one organization
        org = t['entities'].filter(pl.col('entity_type') == 'organization').row(0, named=True)
        self.assertIsNone(org['name'])  # a source ID with no supplied profile is not a fabricated name

    def test_prefer_per_string_mappings_do_not_duplicate_fallback_strings(self):
        a = author(affiliations=[('Known', ['I1'])])
        a['raw_affiliation_strings'] = ['Known', 'Alternative rendering']
        t = extract([work(authors=[a])]).tables
        self.assertEqual(t['signatures']['affiliation_text'].to_list(), ['Known'])

    def test_fallback_preserves_text_without_guessing_institution_matches(self):
        a = author(institutions=[institution()])
        a['raw_affiliation_strings'] = ['University One']
        r = extract([work(authors=[a])])
        self.assertEqual(r.tables['signatures']['affiliation_text'].to_list(), ['University One'])
        self.assertEqual(r.tables['links_signatures_entities'].height, 1)
        self.assertIn('aggregate_institutions_only', {n['code'] for n in r.notes})

    def test_unmatched_affiliation_is_not_lost(self):
        r = extract([work(authors=[author(affiliations=[('Unknown lab', [])])])])
        self.assertEqual(r.tables['signatures']['affiliation_text'].to_list(), ['Unknown lab'])
        self.assertEqual(r.tables['entities']['entity_type'].to_list(), ['person'])

    def test_raw_strings_are_exact_including_separators_whitespace_and_unicode(self):
        name = '  A. | Smith ¦\n'
        aff = '  Département | A\\B\nMontréal  '
        t = extract([work(authors=[author(name, affiliations=[(aff, [])])])]).tables
        self.assertEqual(t['signatures']['author_text'][0], name)
        self.assertEqual(t['signatures']['affiliation_text'][0], aff)

    def test_missing_raw_name_does_not_use_resolved_name(self):
        a = author(None)
        a['author']['display_name'] = 'Known full name'
        t = extract([work(authors=[a])]).tables
        self.assertIsNone(t['signatures']['author_text'][0])
        self.assertEqual(t['entities']['name'][0], 'Known full name')

    def test_raw_orcid_and_profile_orcid_remain_separate(self):
        a = author()
        a['raw_orcid'], a['author']['orcid'] = 'raw-orcid', 'profile-orcid'
        t = extract([work(authors=[a])]).tables
        self.assertEqual(t['signatures']['raw_orcid'][0], 'raw-orcid')
        self.assertEqual(t['entities']['orcid'][0], 'profile-orcid')

    def test_unresolved_author_keeps_signature(self):
        a = author()
        a['author'] = None
        t = extract([work(authors=[a])]).tables
        self.assertEqual(t['signatures'].height, 1)
        self.assertEqual(t['links_signatures_entities'].height, 0)

    def test_unknown_confidence_is_null_not_one(self):
        t = extract([work(authors=[author()])]).tables
        r = t['links_signatures_entities'].row(0, named=True)
        self.assertIsNone(r['score'])
        self.assertIsNone(r['score_kind'])

    def test_scalar_and_exact_type_contracts(self):
        base = dict(signature_id='s', author_order=1, source_record='p', source_revision='r', author_source_field='x')
        for bad in ({'author_text': ['A']}, {'author_order': True}, {'author_order': 1.5}, {'unknown': 'x'}):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                frame('signatures', [{**base, **bad}])

    def test_required_fields(self):
        with self.assertRaises(ContractError):
            frame('content', [{'content_id': 'openalex:W1'}])

    def test_dates_and_large_counts(self):
        w = work()
        w['cited_by_count'] = 5_000_000_000
        r = extract([w]).tables['content'].row(0, named=True)
        self.assertEqual(r['date'], date(2024, 1, 2))
        self.assertEqual(r['updated_at'], datetime(2024, 2, 1, tzinfo=timezone.utc))
        self.assertEqual(r['citations'], 5_000_000_000)

    def test_completeness_does_not_imply_article(self):
        w = work()
        w['type'] = 'book'
        self.assertTrue(extract([w]).tables['content']['is_complete'][0])
        for field, value in [('language', None), ('abstract_inverted_index', '{}'), ('referenced_works', [])]:
            v = {**w, field: value}
            self.assertFalse(extract([v]).tables['content']['is_complete'][0])

    def test_missing_counts_are_unknown_not_zero(self):
        w = work()
        w.pop('cited_by_count')
        self.assertIsNone(extract([w]).tables['content']['citations'][0])

    def test_ids_and_rows_are_independent_of_input_processing_order(self):
        a = author()
        works = [work('W2', [a]), work('W1', [a])]
        left, right = extract(works), extract(reversed(works))
        for name in TABLES:
            self.assertTrue(left.tables[name].equals(right.tables[name]), name)

    def test_duplicate_same_revision_is_idempotent_but_competing_revisions_rejected(self):
        w = work(authors=[author()])
        self.assertEqual(extract([w, deepcopy(w)]).tables['signatures'].height, 1)
        changed = {**w, 'title': 'Changed'}
        with self.assertRaisesRegex(ContractError, 'select one source revision'):
            extract([w, changed])
        old, new = extract([w]), extract([changed])
        self.assertNotEqual(old.tables['signatures']['signature_id'][0], new.tables['signatures']['signature_id'][0])

    def test_latest_observed_profile_wins_deterministically_and_variation_is_noted(self):
        a, b = work('W1', [author()]), work('W2', [author()])
        b['updated_date'] = '2025-01-01'
        b['authorships'][0]['author']['display_name'] = 'New label'
        for data in ([a, b], [b, a]):
            r = extract(data)
            self.assertEqual(r.tables['entities']['name'][0], 'New label')
            self.assertIn('catalog_variation', {n['code'] for n in r.notes})

    def test_classifications_and_hierarchy_with_same_named_different_schemes(self):
        w = work()
        w['primary_topic'] = topic()
        w['topics'] = [topic()]
        w['keywords'] = [dict(id=url('keywords/neuroscience'), display_name='Neuroscience', score=0.0)]
        t = extract([w]).tables
        self.assertEqual(t['classifications'].height, 5)
        self.assertEqual(t['links_content_classifications'].height, 2)
        self.assertEqual(t['links_classifications_classifications'].height, 3)
        self.assertEqual(t['classifications'].filter(pl.col('name') == 'Neuroscience').height, 2)
        keyword = t['links_content_classifications'].filter(pl.col('classification_id').str.contains('keywords/')).row(0, named=True)
        self.assertEqual(keyword['score'], 0.0)
        self.assertEqual(keyword['score_kind'], 'openalex_keyword_score')

    def test_multiple_broader_assertions_can_coexist(self):
        a, b = work('W1'), work('W2')
        a['topics'], b['topics'] = [topic(sub='1')], [topic(sub='2')]
        t = extract([a, b]).tables
        edges = t['links_classifications_classifications'].filter(pl.col('from_classification_id') == 'openalex:T1')
        self.assertEqual(set(edges['to_classification_id']), {'openalex:subfields/1', 'openalex:subfields/2'})

    def test_missing_hierarchy_level_does_not_invent_direct_parent(self):
        w = work()
        q = topic(); q['subfield'] = None
        w['topics'] = [q]
        t = extract([w]).tables
        self.assertEqual(t['links_classifications_classifications']['from_classification_id'].to_list(), ['openalex:fields/1'])

    def test_primary_topic_not_in_topics_is_preserved(self):
        w = work(); w['primary_topic'] = topic()
        t = extract([w]).tables
        self.assertTrue(t['links_content_classifications']['is_primary'][0])

    def test_two_locations_same_source_keep_their_own_licenses_and_versions(self):
        a = location('copy1', kind='repository', version='acceptedVersion', license=None)
        b = location('copy2', kind='repository', license='cc-by')
        w = work(); w['locations'], w['primary_location'] = [a, b], b
        t = extract([w]).tables
        links = t['links_content_classifications'].sort('source_location_id')
        self.assertEqual(t['classifications'].height, 1)
        self.assertEqual(links.height, 2)
        self.assertEqual(links['license'].to_list(), [None, 'cc-by'])
        self.assertEqual(links['is_primary'].to_list(), [False, True])
        self.assertEqual(set(links['relation']), {'hosted_by'})
        self.assertEqual(t['signatures'].height, 0)  # no venue signatures

    def test_unspecified_primary_topic_and_location_remain_unknown(self):
        w = work()
        w['topics'], w['locations'] = [topic()], [location('copy')]
        t = extract([w]).tables
        self.assertEqual(t['links_content_classifications']['is_primary'].to_list(), [None, None])

    def test_primary_location_fallback_and_unknown_source(self):
        w = work(); w['primary_location'] = location('copy')
        self.assertEqual(extract([w]).tables['links_content_classifications'].height, 1)
        w['primary_location']['source'] = None
        r = extract([w])
        self.assertEqual(r.tables['classifications'].height, 0)
        self.assertIn('location_without_source_id', {n['code'] for n in r.notes})

    def test_external_citations_kept_duplicates_removed(self):
        w = work(); w['referenced_works'] *= 2
        t = extract([w]).tables
        self.assertEqual(t['links_content_content'].height, 1)
        self.assertEqual(validate(t)['external_reference_targets'], 1)
        self.assertEqual(t['content'].height, 1)  # no fake content for external targets

    def test_deprecated_concepts_are_explicitly_deferred_but_mesh_is_extracted(self):
        w = work(); w['mesh'] = [{'descriptor_ui': 'D1', 'descriptor_name': 'One'}]
        w['concepts'] = [{'id': url('C1'), 'display_name': 'Old concept', 'level': 0, 'score': 0.5}]
        r = extract([w])
        self.assertIn('deferred_scheme', {n['code'] for n in r.notes})
        self.assertEqual(r.tables['classifications']['classification_id'].to_list(), ['mesh:D1'])

    def test_validator_rejects_broken_endpoint(self):
        t = extract([work(authors=[author()])]).tables
        t['links_signatures_entities'] = t['links_signatures_entities'].with_columns(entity_id=pl.lit('unknown'))
        with self.assertRaisesRegex(ContractError, 'unknown endpoint'):
            validate(t)

    def test_validator_rejects_orphan_and_duplicate_owner(self):
        t = extract([work(authors=[author()])]).tables
        owner = t['links_content_signatures']
        t['links_content_signatures'] = owner.head(0)
        with self.assertRaisesRegex(ContractError, 'content owner'):
            validate(t)
        t['links_content_signatures'] = pl.concat([owner, owner.with_columns(
            link_id=pl.lit('another-link'), method=pl.lit('another assignment'))])
        with self.assertRaisesRegex(ContractError, 'exactly one content owner'):
            validate(t)

    def test_validator_rejects_duplicate_unit_id_and_bad_score_pair(self):
        t = extract([work(authors=[author()])]).tables
        t['content'] = pl.concat([t['content'], t['content']])
        with self.assertRaisesRegex(ContractError, 'duplicate content_id'):
            validate(t)
        t = extract([work(authors=[author()])]).tables
        t['links_signatures_entities'] = t['links_signatures_entities'].with_columns(score=pl.lit(1.0))
        with self.assertRaisesRegex(ContractError, 'score and score_kind'):
            validate(t)

    def test_validator_rejects_duplicate_claim_even_with_different_link_id(self):
        t = extract([work(authors=[author()])]).tables
        links = t['links_signatures_entities']
        t['links_signatures_entities'] = pl.concat([links, links.with_columns(link_id=pl.lit('different-id'))])
        with self.assertRaisesRegex(ContractError, 'duplicate source assertion'):
            validate(t)

    def test_validator_rejects_wrong_source_context_and_negative_counts(self):
        t = extract([work('W1'), work('W2')]).tables
        revisions = dict(t['content'].select('content_id', 'source_revision').rows())
        links = t['links_content_content'].filter(pl.col('from_content_id') == 'openalex:W1')
        t['links_content_content'] = links.with_columns(source_record=pl.lit('openalex:W2'),
                                                       source_revision=pl.lit(revisions['openalex:W2']))
        with self.assertRaisesRegex(ContractError, 'another content item'):
            validate(t)
        with self.assertRaisesRegex(ContractError, 'cannot be negative'):
            extract([{**work(), 'cited_by_count': -1}])

    def test_parquet_roundtrip_preserves_schemas_values_and_validation(self):
        t = extract([work(authors=[author(affiliations=[('University', ['I1'])])])]).tables
        with TemporaryDirectory() as tmp:
            readback = {}
            for name, df in t.items():
                path = Path(tmp) / (name + '.parquet')
                df.write_parquet(path)
                readback[name] = pl.read_parquet(path)
                self.assertTrue(df.equals(readback[name]), name)
            validate(readback)

    def test_abstract_duplicate_keys_escapes_and_sparse_positions(self):
        self.assertEqual(abstract_text('{"one":[0],"two":[2],"one":[3]}'), 'one two one')
        self.assertEqual(abstract_text(r'{"a\"b":[1],"\u00e9":[0],"x\\y":[2]}'), 'é a"b x\\y')
        self.assertIsNone(abstract_text('null'))
        for malformed in ('[]', '{"a":[true]}', '{"a":[-1]}', '{"a":1}', 'not-json'):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                abstract_text(malformed)


class RealPapers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / 'research/openalex/schema-review-examples.parquet'
        cls.raw = pl.read_parquet(path).to_dicts()
        cls.tables = extract(cls.raw).tables

    def test_expected_pair_rows_and_links(self):
        t = self.tables
        self.assertEqual(t['content'].height, 2)
        self.assertEqual(t['signatures'].height, 16)  # 10 on 1995 paper, 6 on 2017 paper
        self.assertEqual(t['links_content_signatures'].height, 16)
        kinds = dict(t['entities'].group_by(pl.col('entity_id').str.extract(r'^openalex:([A-Z])')).len().rows())
        # people, institutions (incl. repository hosts), funders, publishers; venues are classifications
        self.assertEqual(kinds, {'A': 4, 'I': 5, 'F': 8, 'P': 2})
        links = t['links_signatures_entities']
        self.assertEqual(links.filter(pl.col('relation') == 'identifies_person').height, 16)
        self.assertEqual(links.filter(pl.col('relation') == 'identifies_institution').height, 8)
        self.assertEqual(t['links_content_content'].height, 248)

    def test_raw_pairs_exactly_reconstruct_source_occurrences(self):
        observed = self.tables['signatures']
        for work in self.raw:
            cid = work['id'].replace('https://openalex.org/', 'openalex:')
            got = observed.filter(pl.col('source_record') == cid).sort('author_order', 'affiliation_order')
            expected = [(a['raw_author_name'], f['raw_affiliation_string'])
                        for a in work['authorships'] for f in a['affiliations']]
            self.assertEqual(got.select('author_text', 'affiliation_text').rows(), expected)

    def test_deduplicating_paths_does_not_multiply_papers_or_citations(self):
        t = self.tables
        ucla = (t['links_signatures_entities'].filter(pl.col('entity_id') == 'openalex:I161318765')
                .join(t['links_content_signatures'].select('signature_id', 'content_id'), on='signature_id'))
        self.assertEqual(ucla.height, 4)
        papers = ucla.select('content_id').unique().join(t['content'], on='content_id')
        self.assertEqual(papers.height, 2)
        self.assertEqual(papers['citations'].sum(), 110)


if __name__ == '__main__':
    unittest.main()
