"""Hungarian 算法正确性：暴力解 / scipy 交叉验证。"""

import itertools

import numpy as np
import pytest

from topicforge.core.assign import hungarian


def _brute(cost):
    n = cost.shape[0]
    best = 1e18
    for p in itertools.permutations(range(n)):
        best = min(best, sum(cost[i, p[i]] for i in range(n)))
    return best


def test_hungarian_square():
    rng = np.random.default_rng(0)
    for _ in range(12):
        n = rng.integers(3, 8)
        C = rng.random((n, n))
        rc, total = hungarian(C)
        assert abs(total - _brute(C)) < 1e-6
        assert sorted(rc.tolist()) == list(range(n))


def test_hungarian_rect():
    rng = np.random.default_rng(2)
    C = rng.random((4, 6))
    rc, total = hungarian(C)
    assert rc.shape[0] == 4
    assert set(rc.tolist()).issubset(set(range(6)))


def test_hungarian_vs_scipy():
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(1)
    C = rng.random((7, 7))
    rc, total = hungarian(C)
    row, col = scipy.linear_sum_assignment(C)
    assert abs(total - C[row, col].sum()) < 1e-6
