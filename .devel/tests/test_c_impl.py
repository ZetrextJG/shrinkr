import numpy as np
import pytest

from shrinkr.functional import deal, deal_objective, lw_analytical, lw_linear, oas
from shrinkr.monte_carlo import get_large_sample_cov
from shrinkr.reference import (
    ref_deal,
    ref_deal_objective,
    ref_lw_analytical,
    ref_lw_linear,
    ref_oas,
)


@pytest.mark.unit
def test_lw_linear():
    p = 50
    n = 60

    X, _, _ = get_large_sample_cov(p=p, n=n, seed=42)

    ref, _ = ref_lw_linear(X, assume_centered=True)
    value, _ = lw_linear(X, assume_centered=True)
    assert np.allclose(ref, value)


@pytest.mark.unit
def test_lw_analytical_p_less_n():
    p = 50
    n = 60

    _, sc, _ = get_large_sample_cov(p=p, n=n, seed=42)
    lam, U = np.linalg.eigh(sc)

    ref = ref_lw_analytical(lam, n)
    value = lw_analytical(lam, n)
    assert np.allclose(ref, value)


@pytest.mark.unit
def test_lw_analytical_p_greater_n():
    p = 70
    n = 60

    _, sc, _ = get_large_sample_cov(p=p, n=n, seed=42)
    lam, U = np.linalg.eigh(sc)

    ref = ref_lw_analytical(lam, n)
    value = lw_analytical(lam, n)
    assert np.allclose(ref, value)


@pytest.mark.unit
def test_oas():
    p = 70
    n = 30

    _, sc, _ = get_large_sample_cov(p=p, n=n, seed=42)

    ref, _ = ref_oas(sc, n, p)
    value, _ = oas(sc, n, p)
    ref_diag = np.diag(ref)
    value_diag = np.diag(value)

    assert np.allclose(ref_diag, value_diag)
    assert np.allclose(ref, value)


@pytest.mark.unit
def test_deal():
    p = 50
    n = 60
    _, sc, rc = get_large_sample_cov(p=p, n=n, seed=42)

    lam, U = np.linalg.eigh(sc)
    lam_lw = lw_analytical(lam, n, p)
    mean_diff = np.ones(p)
    mean_diff /= np.linalg.norm(mean_diff, ord=2)
    z_vec = U.T @ mean_diff

    # Test objectives
    for gamma in np.linspace(0.1, 1.0, 10):
        gamma = float(gamma)
        obj1 = ref_deal_objective(lam_lw, lam, z_vec, gamma, n)[0]
        obj2 = deal_objective(lam_lw, lam, z_vec, gamma, n)
        assert np.allclose(obj1, obj2)

    # Test results
    lam1 = ref_deal(lam, z_vec, n)
    lam2 = deal(lam, z_vec, n)

    assert np.allclose(lam1, lam2)
