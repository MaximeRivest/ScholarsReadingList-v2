import faiss
import numpy as np
import pytest

from bench.disk80m import ondisk_layout


def test_compact_ondisk_layout_roundtrips(tmp_path):
    counts = [7, 11, 2]
    dim = 8
    file = tmp_path / 'postings.data'
    originals = []
    next_id = 0
    with file.open('wb') as f:
        for n in counts:
            codes = np.arange(n*dim, dtype='uint8').reshape(n, dim)
            ids = np.arange(next_id, next_id+n, dtype='int64')
            f.write(codes.tobytes()); f.write(ids.tobytes())
            originals.append((codes, ids)); next_id += n
    inv = ondisk_layout(file, counts, dim)
    assert inv.compute_ntotal() == sum(counts)
    for i, (codes, ids) in enumerate(originals):
        cp, ip = inv.get_codes(i), inv.get_ids(i)
        np.testing.assert_array_equal(faiss.rev_swig_ptr(cp, codes.size).reshape(codes.shape), codes)
        np.testing.assert_array_equal(faiss.rev_swig_ptr(ip, len(ids)), ids)
        inv.release_codes(i, cp); inv.release_ids(i, ip)


def test_layout_rejects_truncated_data(tmp_path):
    file = tmp_path / 'truncated.data'; file.write_bytes(b'abc')
    with pytest.raises(ValueError):
        ondisk_layout(file, [5], 8)
