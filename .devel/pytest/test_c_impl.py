import numpy as np

from shrinkr.functional import lw_analytical_shrinkage
from shrinkr.monte_carlo import get_large_sample_cov
from shrinkr.reference import (
    lw_analytical_shrinkage as ref_lw_analytical_shrinkage,
)


def test_lw_analytical_p_less_n():
    p = 50
    n = 60

    _, sc, rc = get_large_sample_cov(p=p, n=n, seed=42)
    lam, U = np.linalg.eigh(sc)

    ref = ref_lw_analytical_shrinkage(lam, n)
    value = lw_analytical_shrinkage(lam, n)
    assert np.allclose(ref, value)


def test_lw_analytical_p_greater_n():
    p = 70
    n = 60

    _, sc, rc = get_large_sample_cov(p=p, n=n, seed=42)
    lam, U = np.linalg.eigh(sc)

    ref = ref_lw_analytical_shrinkage(lam, n)
    value = lw_analytical_shrinkage(lam, n)
    assert np.allclose(ref, value)
