import numpy as np

from shrinkr.functional import lw_analytical, oas
from shrinkr.monte_carlo import get_large_sample_cov
from shrinkr.reference import ref_lw_analytical, ref_oas


def test_lw_analytical_p_less_n():
    p = 50
    n = 60

    _, sc, rc = get_large_sample_cov(p=p, n=n, seed=42)
    lam, U = np.linalg.eigh(sc)

    ref = ref_lw_analytical(lam, n)
    value = lw_analytical(lam, n)
    assert np.allclose(ref, value)


def test_lw_analytical_p_greater_n():
    p = 70
    n = 60

    _, sc, rc = get_large_sample_cov(p=p, n=n, seed=42)
    lam, U = np.linalg.eigh(sc)

    ref = ref_lw_analytical(lam, n)
    value = lw_analytical(lam, n)
    assert np.allclose(ref, value)


def test_oas():
    p = 70
    n = 30

    _, sc, rc = get_large_sample_cov(p=p, n=n, seed=42)

    ref, _ = ref_oas(sc, n, p)
    value = oas(sc, n, p)
    ref_diag = np.diag(ref)
    value_diag = np.diag(value)

    assert np.allclose(ref_diag, value_diag)
    assert np.allclose(ref, value)
