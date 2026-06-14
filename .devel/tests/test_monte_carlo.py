import numpy as np
import pytest

from shrinkr.monte_carlo import get_guassian_lda_samples, get_large_sample_cov, get_small_sample_cov


@pytest.mark.unit
def test_get_small():
    n = 100
    X, sc, rc = get_small_sample_cov(n=n)
    assert X.shape == (n, 2)
    assert sc.shape == (2, 2)
    assert rc.shape == (2, 2)


@pytest.mark.unit
def test_get_large():
    p = 50
    n = 100
    X, sc, rc = get_large_sample_cov(p, n)
    assert X.shape == (n, p)
    assert sc.shape == (p, p)
    assert rc.shape == (p, p)


@pytest.mark.unit
def test_get_lda_samples():
    p = 50
    n = 100
    X, y = get_guassian_lda_samples(p, n)
    assert X.shape == (2 * n, p)
    assert y.shape == (2 * n,)
    assert np.sum(y) == np.sum(1 - y)


if __name__ == "__main__":
    test_get_small()
    test_get_large()
    test_get_lda_samples()
