"""one_per_paper (pipeline/release/rows.py): among kept rows that are versions of one paper, only the best-ranked stays."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "release"))
import rows  # noqa: E402


def test_one_per_paper(tmp_path):
    # groups: {10: canonical, 11, 12} and {20, 21: canonical}; 30 is in no group
    np.save(tmp_path / "work.npy", np.array([10, 11, 12, 20, 21], np.uint64))
    np.save(tmp_path / "cluster.npy", np.array([0, 0, 0, 1, 1], np.uint32))
    np.save(tmp_path / "rank.npy", np.array([0, 1, 2, 1, 0], np.uint16))
    work = np.array([12, 30, 11, 20, 21, 10], np.uint64)
    keep = np.array([True, True, True, True, False, False])     # 21 and 10 not kept (e.g. duplicate text)
    out, dropped = rows.one_per_paper(work, keep, tmp_path)
    # group 0: 11 (rank 1) beats 12 (rank 2); group 1: only 20 kept, stays; 30 untouched
    assert out.tolist() == [False, True, True, True, False, False]
    assert dropped == 1


def test_readable_version_wins(tmp_path):
    np.save(tmp_path / "work.npy", np.array([10, 11], np.uint64))
    np.save(tmp_path / "cluster.npy", np.array([0, 0], np.uint32))
    np.save(tmp_path / "rank.npy", np.array([0, 1], np.uint16))
    work = np.array([10, 11], np.uint64)
    out, _ = rows.one_per_paper(work, np.array([True, True]), tmp_path, prefer=np.array([False, True]))
    assert out.tolist() == [False, True]
