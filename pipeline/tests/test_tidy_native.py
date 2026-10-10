"""Native/reference parity, stable IDs, and schema-preserving partition slices."""
from copy import deepcopy
from pathlib import Path
import random
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from pipeline.tidy import native, reference
from pipeline.tidy.contracts import TABLES, ContractError, validate
from pipeline.tidy.keys import key, expr_key
from pipeline.tidy.validation import validate as validate_native
from pipeline.tests.test_tidy_rows import work, author, institution, topic, location, url


def raw_frame(works):
    return pl.from_dicts(works, infer_schema_length=None) if works else pl.DataFrame(schema={'id': pl.String})


class NativeParity(unittest.TestCase):
    def compare(self, works):
        raw = works if isinstance(works, pl.DataFrame) else raw_frame(works)
        got = native.extract(raw)
        expected = reference.extract(raw.to_dicts(), native.source_revisions(raw)) if raw.height else reference.extract([])
        for name, spec in TABLES.items():
            assert_frame_equal(got.tables[name].sort(spec.key), expected.tables[name].sort(spec.key),
                               check_exact=True)
        self.assertEqual(validate(got.tables), validate_native(got.tables))
        return got

    def test_real_papers_all_nine_tables_exact(self):
        p = Path(__file__).resolve().parents[2] / 'research/openalex/schema-review-examples.parquet'
        self.compare(pl.read_parquet(p))

    def test_empty_and_no_authors(self):
        self.compare([])
        self.compare([work()])

    def test_mixed_pair_shapes_raw_fallbacks_and_missing_profiles(self):
        a = author(affiliations=[('One', ['I1', 'I2', 'I2']), ('One', []), ('', [])],
                   institutions=[institution(), institution('I2', 'Two')])
        b = author(None, 'A2', institutions=[institution()]); b['raw_affiliation_strings'] = ['Raw only']
        c = author('Raw name'); c['author'] = None
        self.compare([work(authors=[a, b, c, author()])])

    def test_orcids_same_names_raw_control_characters_and_dates(self):
        a = author(' J. | Smith ¦\n', affiliations=[(' é\\\t" Inst ', ['I99'])])
        a['raw_orcid'], a['author']['orcid'] = 'raw', 'profile'
        w = work(authors=[a, author(' J. | Smith ¦\n', 'A2')]); w['cited_by_count'] = 5_000_000_000
        self.compare([w])

    def test_timestamp_offsets_and_naive_source_times(self):
        stamps=['2024-02-01T00:00:00','2024-02-01 00:00:00.123456',
                '2024-02-01T03:00:00+03:00','2024-02-01']
        self.compare([{**work('W'+str(i)),'updated_date':s} for i,s in enumerate(stamps)])

    def test_profiles_catalog_ties_and_latest_observation(self):
        a, b = work('W1', [author()]), work('W2', [author()])
        b['authorships'][0]['author']['display_name'] = 'é New name "x"'
        self.compare([a, b])
        b['updated_date'] = '2025-01-01'
        self.compare([b, a])

    def test_topics_primary_fallback_keywords_and_partial_hierarchy(self):
        a, b = work('W1'), work('W2')
        a['primary_topic'] = topic(); a['keywords'] = [dict(id=url('keywords/neuroscience'), display_name='Neuroscience', score=0.0), {}]
        b['topics'] = [topic('T2'), topic('T2'), topic('T3')]; b['topics'][0]['subfield'] = None
        b['topics'][2]['score'] = None
        self.compare([a, b])

    def test_multiple_locations_same_source_and_primary_not_in_locations(self):
        w = work()
        w['locations'] = [location('copy1',kind='repository',license=None), location('copy2',kind='repository')]
        w['primary_location'] = deepcopy(w['locations'][1])
        self.compare([w])
        w['primary_location'] = location('copy3',source='S2')
        self.compare([w])
        for loc in w['locations']: loc['id'] = None
        w['primary_location'] = deepcopy(w['locations'][0])
        self.compare([w])

    def test_publication_links_require_explicit_support(self):
        cases=[('journal',None,None,'hosted_by'),('journal',None,True,'published_in'),
               ('journal','publishedVersion',None,'published_in'),
               ('journal','publishedVersion',False,'hosted_by'),
               ('journal','acceptedVersion',True,'hosted_by'),
               ('other','publishedVersion',True,'hosted_by'),
               ('repository','publishedVersion',True,'hosted_by'),
               ('conference','publishedVersion',None,'published_in')]
        works=[]
        for i,(kind,version,flag,expected) in enumerate(cases):
            w=work('W'+str(i));loc=location('copy'+str(i),source='S'+str(i),kind=kind,version=version)
            loc['is_published']=flag;w['primary_location']=loc;works.append(w)
        got=self.compare(works)
        links=dict(got.tables['links_content_classifications'].select('content_id','relation').rows())
        for i,case in enumerate(cases):self.assertEqual(links['openalex:W'+str(i)],case[3])
        self.assertIn('location_version_conflict',{n['code'] for n in got.notes})

    def test_absent_and_empty_objects(self):
        for value in (None, {}):
            w = work(); w['primary_location'] = value; w['primary_topic'] = value
            w['authorships'] = [dict(author=value, affiliations=[], institutions=[])]
            self.compare([w])

    def test_abstract_repeated_words_escapes_ties_empty_and_whitespace(self):
        examples = [None, '', '{}', ' null ', '{"one":[0],"two":[2],"one":[3]}',
                    r'{"é":[0],"a\"b":[1],"\\":[2],"x\ny":[3]}',
                    '{"b":[0],"a":[0],"c":[],"": [2]}', '{ "x" : [ -0 , 10 ] }']
        self.compare([{**work('W'+str(i)), 'abstract_inverted_index': ab} for i,ab in enumerate(examples)])

    def test_invalid_abstracts_are_unknown_and_noted_without_fragments(self):
        bads = ['[]','{"a":[true]}','{"a":[-1]}','{"a":[01]}','{"a":[1.0]}',
                '{"a":[0], "b": false}', '{"a":[0]} garbage', '{"a":[0],}',
                '{"a\\x":[0]}', '   ', '{"a":null}', '{"a":[9223372036854775808]}']
        works = [{**work('W' + str(i)), 'abstract_inverted_index': text} for i, text in enumerate(bads)]
        works.append({**work('W99'), 'abstract_inverted_index': '{"fine":[0]}'})
        got = self.compare(works)
        abstracts = dict(got.tables['content'].select('content_id', 'abstract').rows())
        self.assertEqual(abstracts.pop('openalex:W99'), 'fine')
        self.assertEqual(set(abstracts.values()), {None})
        self.assertEqual(next(n['count'] for n in got.notes if n['code'] == 'invalid_abstract'), len(bads))

    def test_unrecognized_identifiers_are_dropped_consistently_not_fatal(self):
        a = author(affiliations=[('Inst', ['I1', 'not-an-id'])], institutions=[institution(), {'id': 'bad', 'display_name': 'X'}])
        a['author']['id'] = 'orcid:0000'
        w = work(authors=[a])
        w['topics'] = [topic(), {**topic('T2'), 'id': 'nope'}]
        w['topics'][0]['field'] = {'id': 'bad-field', 'display_name': 'F'}
        w['referenced_works'] = [url('W2'), 'doi:10.1/x', None]
        w['funders'] = [dict(id='F-bad', display_name='?', ror=None)]
        got = self.compare([w])
        self.assertIn('invalid_identifier', {n['code'] for n in got.notes})
        self.assertEqual(got.tables['links_content_content']['to_content_id'].to_list(), ['openalex:W2'])
        self.assertEqual(got.tables['signatures'].height, 1)  # the author's signature survives

    def test_duplicate_references_external_targets_and_unknown_count(self):
        w = work(); w['referenced_works'] = [url('W2'), url('W2'), url('W333')]; w['referenced_works_count'] = None
        self.compare([w, work('W2')])

    def test_same_revision_duplicates_and_competing_revisions(self):
        w = work(authors=[author()])
        self.compare([w, deepcopy(w)])
        with self.assertRaises(ContractError):
            native.extract(raw_frame([w, {**w,'title':'Changed'}]))

    def test_stable_keys_are_independently_identical_to_hashlib(self):
        vals = ['', None, 'null', 'é', 'a|b', 'a\\b', 'a\nb', '\u0000', '"', '💡']
        for v in vals:
            expected = key('test:', v, 'x:y', None)
            got = pl.select(expr_key('test:', pl.lit(v,dtype=pl.String),pl.lit('x:y'),pl.lit(None,dtype=pl.String))).item()
            self.assertEqual(expected,got)
        self.assertNotEqual(key('', 'ab','c'),key('','a','bc'))
        self.assertNotEqual(key('', None),key('','null'))

    def test_source_keys_ignore_column_order_and_preserve_null_struct_distinction(self):
        df = raw_frame([work('W1',[author()]), work('W2',[author('Bob','A2')])])
        self.assertEqual(native.source_revisions(df),native.source_revisions(df.select(df.columns[::-1])))
        self.assertEqual(native.source_revisions(df),native.source_revisions(df.reverse()))
        full = native.extract(df)
        parts = [native.extract(df.slice(i,1)) for i in range(df.height)]
        # Work-owned rows must be identical whether extracted together or separately.
        for name in TABLES.keys()-{'entities','classifications'}:
            union=pl.concat([p.tables[name] for p in parts]).sort(TABLES[name].key)
            assert_frame_equal(full.tables[name].sort(TABLES[name].key), union,check_exact=True)

    def test_unused_nonfinite_source_metadata_does_not_break_extraction(self):
        self.compare([{**work(), 'unused': float('nan')}])

    def test_source_keys_preserve_null_struct_vs_present_empty_struct(self):
        df=pl.DataFrame({'id':[url('W1'),url('W1')], 'nested':[None, {'name':None}]},
                        schema={'id':pl.String,'nested':pl.Struct({'name':pl.String})})
        hashes=df.select(native.source_revision_expr(df)).to_series().to_list()
        self.assertNotEqual(hashes[0],hashes[1])

    def test_native_rejects_wrong_scalar_types_and_negative_counts(self):
        examples=[{**work(),'cited_by_count':-1},{**work(),'cited_by_count':1.5},
                  {**work(),'title':{'nested':'not text'}},{**work(),'publication_date':123},
                  {**work(),'updated_date':123},work(authors=[{**author(),'is_corresponding':1}])]
        for w in examples:
            with self.subTest(w=w),self.assertRaises((ContractError,pl.exceptions.PolarsError)):
                native.extract(raw_frame([w]))

    def test_columnar_validator_catches_the_same_broken_links(self):
        base=self.compare([work(authors=[author()])]).tables
        mutations=[]
        x=dict(base);x['entities']=pl.concat([x['entities'],x['entities']]);mutations.append(x)
        x=dict(base);x['links_signatures_entities']=x['links_signatures_entities'].with_columns(entity_id=pl.lit('missing'));mutations.append(x)
        x=dict(base);x['links_content_signatures']=x['links_content_signatures'].head(0);mutations.append(x)
        x=dict(base);x['signatures']=x['signatures'].with_columns(author_order=pl.lit(0,dtype=pl.Int64));mutations.append(x)
        x=dict(base);x['links_signatures_entities']=x['links_signatures_entities'].with_columns(score=pl.lit(1.0));mutations.append(x)
        x=dict(base);x['entities']=x['entities'].with_columns(source_revision=pl.lit('wrong'));mutations.append(x)
        for i,tables in enumerate(mutations):
            for check in (validate,validate_native):
                with self.subTest(case=i,check=check.__module__),self.assertRaises(ContractError):check(tables)

    def test_seeded_generated_authorships(self):
        rng=random.Random(734)
        works=[]
        for i in range(30):
            aa=[]
            for ai in range(rng.randrange(5)):
                affs=[(rng.choice(['Inst A',' Inst B\n','é','']),rng.sample(['I1','I2'],rng.randrange(3))) for _ in range(rng.randrange(4))]
                a=author(rng.choice(['J. Smith','Alice',None]),'A'+str(rng.randrange(5)),affs,[institution(),institution('I2','Other')])
                if rng.random()<0.2: a['affiliations']=[]
                aa.append(a)
            w=work('W'+str(i),aa)
            w['topics']=[topic('T'+str(rng.randrange(4))) for _ in range(rng.randrange(4))]
            w['referenced_works']=[url('W'+str(rng.randrange(50))) for _ in range(rng.randrange(6))]
            works.append(w)
        self.compare(works)


if __name__ == '__main__':
    unittest.main()


class Version3Coverage(unittest.TestCase):
    """New metadata coverage, vocabulary-level reduction, and partition invariance."""
    compare = NativeParity.compare

    def test_mesh_descriptors_headings_duplicates_and_kleene_major_flags(self):
        w = work()
        w['mesh'] = [dict(descriptor_ui='D1', descriptor_name='Heart', qualifier_ui='Q9', qualifier_name='surgery', is_major_topic=False),
                     dict(descriptor_ui='D1', descriptor_name='Heart', qualifier_ui='Q9', qualifier_name='surgery', is_major_topic=False),
                     dict(descriptor_ui='D1', descriptor_name='Heart', qualifier_ui=None, qualifier_name=None, is_major_topic=True),
                     dict(descriptor_ui='D2', descriptor_name='Lung', qualifier_ui=None, qualifier_name=None, is_major_topic=None),
                     dict(descriptor_ui=None, descriptor_name='No id', qualifier_ui=None, qualifier_name=None, is_major_topic=True)]
        got = self.compare([w])
        links = dict(got.tables['links_content_classifications'].select('classification_id', 'is_primary').rows())
        self.assertEqual(links, {'mesh:D1': True, 'mesh:D2': None, 'mesh:D1/Q9': False})
        edge = got.tables['links_classifications_classifications'].row(0, named=True)
        self.assertEqual((edge['from_classification_id'], edge['to_classification_id']), ('mesh:D1/Q9', 'mesh:D1'))
        self.assertTrue({'mesh_duplicate_entry', 'mesh_without_descriptor'} <= {n['code'] for n in got.notes})

    def test_mesh_supplementary_concepts_unexpected_and_malformed_records(self):
        w = work()
        w['mesh'] = [dict(descriptor_ui='C000724312', descriptor_name='Chinese people', qualifier_ui=None, qualifier_name=None, is_major_topic=False),
                     dict(descriptor_ui='X1', descriptor_name='Odd', qualifier_ui=None, qualifier_name=None, is_major_topic=False),
                     dict(descriptor_ui='D12a', descriptor_name='Bad', qualifier_ui=None, qualifier_name=None, is_major_topic=False),
                     dict(descriptor_ui='D1\n', descriptor_name='Newline', qualifier_ui=None, qualifier_name=None, is_major_topic=False)]
        got = self.compare([w])
        kinds = dict(got.tables['classifications'].select('classification_id', 'classification_type').rows())
        self.assertEqual(kinds, {'mesh:C000724312': 'mesh_supplementary_concept', 'mesh:X1': 'mesh_record'})
        self.assertTrue({'mesh_malformed_identifier', 'mesh_unexpected_record_type'} <= {n['code'] for n in got.notes})

    def test_sdgs_awards_funders_and_source_hosts(self):
        w = work()
        w['sustainable_development_goals'] = [dict(id='https://metadata.un.org/sdg/3', display_name='Health', score=0.7)] * 2
        w['funders'] = [dict(id=url('F1'), display_name='Fund', ror='https://ror.org/x'), dict(id=None, display_name='Unknown', ror=None)]
        w['awards'] = [dict(id=url('G1'), display_name=None, funder_award_id='R01-1', funder_id=url('F1'), funder_display_name='Fund name'),
                       dict(id=url('G2'), display_name='Grant', funder_award_id='X', funder_id=None, funder_display_name=None),
                       dict(id=None, display_name=None, funder_award_id='lost', funder_id=url('F2'), funder_display_name='F2')]
        loc = location('copy'); loc['source'].update(issn_l='1234-5678', host_organization=url('P1'), host_organization_name='Publisher')
        w['primary_location'] = loc
        t = self.compare([w]).tables
        cl = t['classifications'].select('classification_id', 'classification_type', 'notation').sort('classification_id').rows()
        self.assertIn(('openalex:G1', 'award', 'R01-1'), cl)
        self.assertIn(('unsdg:3', 'sdg', '3'), cl)
        self.assertIn(('openalex:S1', 'journal', '1234-5678'), cl)
        self.assertEqual(t['links_content_entities'].select('entity_id', 'relation').rows(), [('openalex:F1', 'funded_by')])
        ce = set(t['links_classifications_entities'].select('classification_id', 'relation', 'entity_id').rows())
        self.assertEqual(ce, {('openalex:G1', 'funded_by', 'openalex:F1'), ('openalex:S1', 'host_organization', 'openalex:P1')})
        f1 = t['entities'].filter(pl.col('entity_id') == 'openalex:F1').row(0, named=True)
        self.assertEqual(f1['ror'], 'https://ror.org/x')  # more complete funders[] observation wins
        bad = self.compare([{**work(), 'sustainable_development_goals': [dict(id='sdg-3'), dict(id='https://metadata.un.org/sdg/3\n')]}])
        self.assertEqual(bad.tables['classifications'].height, 0)
        self.assertIn('invalid_identifier', {n['code'] for n in bad.notes})

    def test_content_bibliographic_access_and_identifier_fields(self):
        w = work()
        w.update(biblio=dict(volume='12', issue='3', first_page='e45', last_page=None),
                 open_access=dict(is_oa=True, oa_status='gold', oa_url='https://x.org/a.pdf', any_repository_has_fulltext=False),
                 fwci=float('nan'), is_paratext=False,
                 ids=[dict(key='openalex', value=url('W1')), dict(key='pmid', value='https://pubmed.ncbi.nlm.nih.gov/9'),
                      dict(key='pmid', value='second-ignored'), dict(key='mag', value='123')])
        got = self.compare([w])
        r = got.tables['content'].row(0, named=True)
        self.assertEqual((r['volume'], r['first_page'], r['last_page'], r['oa_status'], r['pmid'], r['pmcid'], r['mag']),
                         ('12', 'e45', None, 'gold', 'https://pubmed.ncbi.nlm.nih.gov/9', None, '123'))
        self.assertIsNone(r['fwci'])
        self.assertIn('nonfinite_fwci', {n['code'] for n in got.notes})

    def test_hierarchy_is_one_row_per_edge_with_latest_witness(self):
        a, b = work('W9'), work('W10')
        a['topics'], b['topics'] = [topic()], [topic()]
        t = self.compare([a, b]).tables
        h = t['links_classifications_classifications']
        self.assertEqual(h.height, 3)
        # Equal timestamps and completeness: higher work NUMBER wins (10 > 9), not string order.
        self.assertEqual(set(h['source_record']), {'openalex:W10'})
        b['updated_date'] = '2023-01-01'
        h = self.compare([a, b]).tables['links_classifications_classifications']
        self.assertEqual(set(h['source_record']), {'openalex:W9'})

    def test_extracting_partitions_separately_then_reducing_equals_one_extraction(self):
        from pipeline.tidy.contracts import REDUCED
        from pipeline.tidy.reduction import reduce_frame
        works = []
        for i in range(12):
            w = work('W' + str(100 + i), [author('Same Name', 'A' + str(i % 3), [('Inst', ['I1'])], [institution('I1', 'Name ' + str(i % 2))])])
            w['updated_date'] = f'2024-0{1 + i % 5}-01'
            w['topics'] = [topic('T' + str(i % 2))]
            w['funders'] = [dict(id=url('F1'), display_name='Fund ' + str(i % 4), ror=None if i % 3 else 'r')]
            loc = location('c' + str(i)); loc['source'].update(host_organization=url('P1'), host_organization_name='Pub ' + str(i))
            w['primary_location'] = loc
            works.append(w)
        raw = raw_frame(works)
        whole = native.extract(raw).tables
        parts = [native.extract(raw.slice(s, 4)).tables for s in (0, 4, 8)]
        for name, spec in TABLES.items():
            merged = pl.concat([p[name] for p in parts])
            merged = reduce_frame(name, merged) if name in REDUCED else merged
            assert_frame_equal(merged.sort(spec.key), whole[name].sort(spec.key), check_exact=True)
