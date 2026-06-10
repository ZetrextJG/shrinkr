import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shrinkr.functional import lw_analytical, lw_linear, oas
from shrinkr.monte_carlo import get_large_sample_cov
from shrinkr.reference import ref_lw_analytical, ref_lw_linear, ref_oas


@pytest.mark.prop
@settings(max_examples=100, deadline=None)
@given(
    p=st.integers(min_value=20, max_value=200),
    n=st.integers(min_value=100, max_value=300),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_lw_analytical_p_less_n(p, n, seed):
    _, sc, _ = get_large_sample_cov(p=p, n=n, seed=seed)
    lam, U = np.linalg.eigh(sc)

    ref = ref_lw_analytical(lam, n)
    value = lw_analytical(lam, n)

    np.testing.assert_allclose(value, ref, rtol=1e-5, atol=1e-8)


@pytest.mark.prop
@settings(max_examples=100, deadline=None)
@given(
    p=st.integers(min_value=20, max_value=200),
    n=st.integers(min_value=100, max_value=300),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_lw_linear(p, n, seed):
    X, sc, _ = get_large_sample_cov(p=p, n=n, seed=seed)

    ref, ref_s = ref_lw_linear(X, assume_centered=False)
    value, value_s = lw_linear(X, assume_centered=False)

    np.testing.assert_allclose(ref_s, value_s, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(value, ref, rtol=1e-5, atol=1e-8)


@pytest.mark.prop
@settings(max_examples=100, deadline=None)
@given(
    p=st.integers(min_value=20, max_value=200),
    n=st.integers(min_value=100, max_value=300),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_oas(p, n, seed):
    _, sc, _ = get_large_sample_cov(p=p, n=n, seed=seed)

    ref, ref_s = ref_oas(sc, n)
    value, value_s = oas(sc, n)

    np.testing.assert_allclose(ref_s, value_s, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(value, ref, rtol=1e-5, atol=1e-8)
