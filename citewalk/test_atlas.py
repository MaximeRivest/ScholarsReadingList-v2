"""Tests for frozen positions, honest missingness, persistence, and embedding reads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas import AtlasStore, AtlasError, fit_projection, unit_vector, publication_key
from server import Embeddings


class FakeGraph:
    meta = {'generation': 'test-generation'}
    def node_of_work(self, number): return number
    def topic_of(self, number): return {'name': 'Topic '+str(number % 2), 'subfield': 'Test field'}


class FakeEmbeddings:
    plan_id = 'test-plan'
    dimensions = 8
    def __init__(self):
        rng=np.random.default_rng(719)
        self.data={i:rng.normal(size=self.dimensions).astype(np.float16) for i in range(100,500)}
    def vectors(self, numbers): return [self.data.get(n) for n in numbers]
    def stats(self): return {'vectors':len(self.data)}


def result(numbers, focus=None):
    numbers=list(numbers); focus=numbers[0] if focus is None else focus
    nodes={f'W{i}':dict(id=f'W{i}',title=f'Publication about research question {i}',year=2020,
                       cited_by=10,type='article',column='x' if i==focus else 'similar',records=[f'W{i}']) for i in numbers}
    return dict(focus=f'W{focus}',nodes=nodes,links=[],answers=[],stats={})


def xy(data): return {n['atlas_key']:(n['mx'],n['my']) for n in data['nodes'].values() if n['map_status']=='placed'}


class FrozenMapTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=AtlasStore(self.temp.name);self.graph=FakeGraph();self.emb=FakeEmbeddings()

    def test_walk_keeps_common_points_bit_for_bit_and_does_not_refit_axes(self):
        first=result(range(100,115));self.store.annotate(self.graph,self.emb,first)
        identifier=first['atlas']['id'];model=self.store.load(identifier)
        mean,axes=model.mean.copy(),model.axes.copy()
        nxt=result(range(106,122),focus=110);self.store.annotate(self.graph,self.emb,nxt,identifier)
        self.assertEqual(nxt['atlas']['id'],identifier)
        common=xy(first).keys()&xy(nxt).keys()
        self.assertGreater(len(common),0)
        self.assertTrue(all(xy(first)[i]==xy(nxt)[i] for i in common))
        np.testing.assert_array_equal(model.axes,axes);np.testing.assert_array_equal(model.mean,mean)
        for n in nxt['nodes'].values():
            v=unit_vector(self.emb.data[int(n['vector_source_id'][1:])])
            np.testing.assert_allclose([n['mx'],n['my']],(v-mean)@axes,atol=1e-14)

    def test_positions_survive_server_restart_and_query_order(self):
        first=result(range(100,112));self.store.annotate(self.graph,self.emb,first)
        new_store=AtlasStore(self.temp.name)
        again=result(reversed(range(100,112)),focus=104)
        new_store.annotate(self.graph,self.emb,again,first['atlas']['id'])
        self.assertEqual(xy(first),xy(again))

    def test_missing_vectors_are_never_placed_via_links(self):
        self.emb.data.pop(104);self.emb.data[105]=np.zeros(8)
        r=result(range(100,110));r['links']=[{'s':'W100','t':'W104','kind':'x-ref'}]
        self.store.annotate(self.graph,self.emb,r)
        self.assertIsNone(r['nodes']['W104']['mx']);self.assertIsNone(r['nodes']['W105']['my'])
        self.assertEqual(r['nodes']['W105']['map_status'],'invalid_vector')
        self.assertEqual(set(r['atlas']['unplaced']),{'W104','W105'})
        self.assertEqual(r['atlas']['plotted'],8)

    def test_backfill_adds_positions_without_moving_existing_ones(self):
        vector=self.emb.data.pop(104)
        first=result(range(100,112));self.store.annotate(self.graph,self.emb,first)
        self.emb.data[104]=vector
        newer=result(range(100,112));self.store.annotate(self.graph,self.emb,newer,first['atlas']['id'])
        self.assertEqual(newer['atlas']['plotted'],12)
        for key,p in xy(first).items():self.assertEqual(xy(newer)[key],p)

    def test_alias_change_does_not_move_a_title_group_or_switch_its_vector(self):
        first=result(range(100,110));self.store.annotate(self.graph,self.emb,first)
        second=result([101,102,103,250]);second['nodes']['W250']['title']=first['nodes']['W100']['title']
        second['nodes']['W250']['records']=['W250','W100']
        self.store.annotate(self.graph,self.emb,second,first['atlas']['id'])
        a,b=first['nodes']['W100'],second['nodes']['W250']
        self.assertEqual((a['mx'],a['my']),(b['mx'],b['my']))
        self.assertEqual(b['vector_source_id'],'W100')

    def test_new_map_is_explicit_and_existing_saved_map_is_not_overwritten(self):
        first=result(range(100,110));self.store.annotate(self.graph,self.emb,first)
        old=xy(first)
        other=result(range(200,210));self.store.annotate(self.graph,self.emb,other)
        self.assertNotEqual(first['atlas']['id'],other['atlas']['id'])
        again=result(range(100,110));self.store.annotate(self.graph,self.emb,again,first['atlas']['id'])
        self.assertEqual(old,xy(again))

    def test_different_model_or_graph_cannot_reuse_map(self):
        first=result(range(100,110));self.store.annotate(self.graph,self.emb,first)
        self.emb.plan_id='different-plan'
        with self.assertRaises(AtlasError):self.store.annotate(self.graph,self.emb,result(range(100,110)),first['atlas']['id'])
        self.emb.plan_id='test-plan';self.graph.meta={'generation':'new-graph'}
        with self.assertRaises(AtlasError):self.store.annotate(self.graph,self.emb,result(range(100,110)),first['atlas']['id'])

    def test_projection_handles_rank_one_without_jitter_and_rejects_identical_vectors(self):
        v=np.asarray([[1.,0],[1.,0],[0.,1]])
        mean,axes,captured=fit_projection(v)
        np.testing.assert_array_equal(axes[:,1],0)
        self.assertAlmostEqual(captured,1)
        with self.assertRaises(AtlasError):fit_projection(np.ones((4,3)))
        self.assertIsNone(unit_vector([float('nan'),1]));self.assertIsNone(unit_vector([0,0]))

    def test_too_few_vectors_keeps_citation_result_and_lists_unplaced(self):
        self.emb.data={i:v for i,v in self.emb.data.items() if i in (100,101)}
        r=result(range(100,106));r['answers']=['W104']
        self.store.annotate(self.graph,self.emb,r)
        self.assertEqual(r['answers'],['W104']);self.assertEqual(r['atlas']['status'],'unavailable')
        self.assertEqual(r['nodes']['W100']['map_status'],'no_projection')
        self.assertTrue(all(n['mx'] is None for n in r['nodes'].values()))

    def test_pca_geometry_and_exact_neighbors_are_separate_and_measured(self):
        r=result(range(100,115));self.store.annotate(self.graph,self.emb,r)
        quality=r['atlas']['quality'];self.assertEqual(quality['k'],5)
        self.assertTrue(0<=quality['neighbor_preservation']<=1)
        self.assertTrue(0<=r['atlas']['anchor_variance_captured']<=1)
        n=r['nodes']['W100'];v=unit_vector(self.emb.data[100])
        expected=sorted(((float(1-v@unit_vector(self.emb.data[i])),f'W{i}') for i in range(101,115)))[:5]
        self.assertEqual([x['id'] for x in n['semantic_neighbors']],[i for _,i in expected])
        for entry,(d,_) in zip(n['semantic_neighbors'],expected):self.assertAlmostEqual(entry['similarity'],1-d)
        self.assertAlmostEqual(n['sem'],0)

    def test_path_traversal_and_unknown_map_rejected(self):
        for bad in ('../../secret','a'*64,'xyz'):
            with self.assertRaises(AtlasError):self.store.load(bad)


class CommittedEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'plan-summary.json').write_text(json.dumps(dict(plan_id='plan',dimensions=3)))
    def shard(self,name,ids,plan='plan',temporary=False):
        p=self.root/'shards/year=2020/rg-000000'/name;p.mkdir(parents=True)
        np.save(p/'work_ids.npy',np.asarray(ids,np.uint64));np.save(p/'vectors.npy',np.ones((len(ids),3),np.float16))
        (p/'commit.json').write_text(json.dumps(dict(plan_id=plan,task={'rows':len(ids)})))
        return p
    def test_temporary_shards_ignored_then_refresh_adds_new_commits(self):
        self.shard('part-2020-01-01-000000000',[100])
        self.shard('.part-2020-01-02-000000000.tmp-x',[101])
        emb=Embeddings(self.root);self.assertEqual(emb.stats()['vectors'],1)
        self.assertIsNone(emb.vectors([101])[0])
        self.shard('part-2020-01-03-000000000',[102]);emb.refresh()
        self.assertEqual(emb.stats()['vectors'],2);self.assertIsNotNone(emb.vectors([102])[0])
    def test_wrong_plan_and_duplicates_do_not_replace_working_index(self):
        self.shard('part-one',[100]);emb=Embeddings(self.root)
        p=self.shard('part-two',[101],plan='wrong')
        with self.assertRaises(ValueError):emb.refresh()
        self.assertEqual(emb.stats()['vectors'],1)
        (p/'commit.json').write_text(json.dumps(dict(plan_id='plan',task={'rows':1})))
        np.save(p/'work_ids.npy',np.asarray([100],np.uint64))
        with self.assertRaises(ValueError):emb.refresh()
        self.assertEqual(emb.stats()['vectors'],1)
    def test_shape_and_plan_changes_fail_clearly(self):
        p=self.shard('part-one',[100]);np.save(p/'vectors.npy',np.ones((1,4),np.float16))
        with self.assertRaises(ValueError):Embeddings(self.root)
        np.save(p/'vectors.npy',np.ones((1,3),np.float16));emb=Embeddings(self.root)
        (self.root/'plan-summary.json').write_text(json.dumps(dict(plan_id='another',dimensions=3)))
        with self.assertRaises(ValueError):emb.refresh()


if __name__=='__main__':unittest.main()
