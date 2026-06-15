import numpy as np
import pytest

from shrinkr.functional import loss_fm, loss_prial
from shrinkr.monte_carlo import get_large_sample_cov, get_small_sample_cov
from shrinkr.reference import (
    ref_deal,
    ref_lw_analytical,
    ref_lw_analytical_unstable,
    ref_lw_linear,
    ref_oas,
)


@pytest.mark.unit
def test_lw_linear_example():
    X, _, real_cov = get_small_sample_cov()
    _, shrinkage_coefficient = ref_lw_linear(X)
    assert np.allclose(shrinkage_coefficient, 0.23025, atol=1e-4)


@pytest.mark.unit
def test_lw_analytical():
    p = 50
    n = 60
    _, sc, rc = get_large_sample_cov(p=p, n=n, seed=42)

    lam, U = np.linalg.eigh(sc)
    lam1 = ref_lw_analytical(lam, n)

    assert lam1.shape == lam.shape

    sc_hat1 = U @ np.diag(lam1) @ (U.T)
    prial1 = loss_prial(sc, sc_hat1, rc)

    lam2 = ref_lw_analytical_unstable(lam, n)
    sc_hat2 = U @ np.diag(lam2) @ (U.T)
    prial2 = loss_prial(sc, sc_hat2, rc)

    # Expect (at least) some improvement
    # and more stable then original
    assert prial1 > 0.1
    assert prial1 >= prial2 - 1e-6


@pytest.mark.unit
def test_deal():
    p = 50
    n = 60
    _, sc, rc = get_large_sample_cov(p=p, n=n, seed=42)

    lam, U = np.linalg.eigh(sc)
    mean_diff = np.ones(p)
    mean_diff /= np.linalg.norm(mean_diff, ord=2)
    z_vec = U.T @ mean_diff

    v0 = np.linalg.solve(sc, mean_diff)
    fm0 = loss_fm(v0, rc, mean_diff)

    lam1 = ref_deal(lam, z_vec, n - 2)
    assert lam1.shape == lam.shape
    v1 = U @ (z_vec / lam1)

    fm1 = loss_fm(v1, rc, mean_diff)

    # Lower better - Fisher Margin
    assert fm1 <= fm0 <= 0


@pytest.mark.unit
def test_oas():
    X, sample_cov, real_cov = get_small_sample_cov()
    n = X.shape[0]
    sc_hat, shrinkage = ref_oas(sample_cov, n)
    assert 0 <= shrinkage <= 1
