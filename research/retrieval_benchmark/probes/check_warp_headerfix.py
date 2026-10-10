"""Compare released/patched WARP rankings on an existing index; no rebuild.

First run with the normal environment and --record; then run with the patch overlay
on PYTHONPATH. Use a generous memory limit for the released loader's startup spike.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from xtr_warp import XTRWarp

from bench.common import DATA, DATASETS

p = argparse.ArgumentParser()
p.add_argument('--record', action='store_true')
a = p.parse_args()
torch.set_num_threads(4)
index = XTRWarp(index=str(DATA / 'index2/warp-nb2/lateon_pool4/beir4+openalex1m')).load(device='cpu', mmap=True)
results = {}
for ds in DATASETS:
    folder = DATA / 'emb/lateon_pool4' / ds
    qs = np.load(folder / 'queries.npy')
    lens = np.load(folder / 'qlens.npy')
    results[ds] = [index.search(torch.from_numpy(qs[i:i+1, :lens[i]]).float(), top_k=100, num_threads=4)[0] for i in range(5)]
path = DATA / 'results/compress-v2/headerfix-reference.json'
if a.record:
    path.write_text(json.dumps(results))
    print('Saved released-engine reference: 20 queries, top 100 each')
else:
    ref = json.loads(path.read_text())
    for ds in DATASETS:
        for actual, expected in zip(results[ds], ref[ds]):
            assert [x[0] for x in actual] == [x[0] for x in expected], ds
            np.testing.assert_allclose([x[1] for x in actual], [x[1] for x in expected], rtol=1e-5, atol=1e-5)
    print('Patched loader preserves all 2,000 ranked results and scores')
