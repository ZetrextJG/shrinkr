import numpy as np

from shrinkr.functional import prial
from shrinkr.monte_carlo import get_large_sample_cov, get_small_sample_cov
from shrinkr.reference.lw_analytical import lw_analytical_shrinkage
from shrinkr.reference.lw_linear import lw_linear_shrinkage


def test_lw_linear_example():
    X, _, real_cov = get_small_sample_cov()
    shrinkage_coefficient = lw_linear_shrinkage(X)
    assert np.allclose(shrinkage_coefficient, 0.23025, atol=1e-4)


def test_lw_analytical():
    p = 50
    n = 60
    _, sc, rc = get_large_sample_cov(p=p, n=n)

    lam, U = np.linalg.eigh(sc)
    lam = lw_analytical_shrinkage(lam, n)
    sc_hat = U @ np.diag(lam) @ (U.T)

    # Expect (at least) small improvement
    assert prial(sc, sc_hat, rc) > 0.1
