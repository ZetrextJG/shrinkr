import numpy as np
import pytest
from data.consts import BENCH_TEST_CASES

from shrinkr.functional import lw_analytical, lw_linear, oas
from shrinkr.monte_carlo import get_large_sample_cov
from shrinkr.reference import ref_lw_analytical, ref_lw_linear, ref_oas


@pytest.mark.parametrize("p, n", BENCH_TEST_CASES)
@pytest.mark.parametrize("func", [ref_lw_analytical, lw_analytical])
def test_lw_analytical(benchmark, p, n, func):
    _, sc, _ = get_large_sample_cov(p=p, n=n, seed=42)
    lam, _ = np.linalg.eigh(sc)
    _ = func(lam, n)  # Warm-up
    benchmark.group = f"LW_Analytical | (p={p}, n={n})"
    benchmark(func, lam, n)


@pytest.mark.parametrize("p, n", BENCH_TEST_CASES)
@pytest.mark.parametrize("func", [ref_lw_linear, lw_linear])
def test_lw_linear(benchmark, p, n, func):
    X, _, _ = get_large_sample_cov(p=p, n=n, seed=42)
    _ = func(X, assume_centered=False)  # Warm-up
    benchmark.group = f"LW_Linear | (p={p}, n={n})"
    benchmark(func, X, assume_centered=False)


@pytest.mark.parametrize("p, n", BENCH_TEST_CASES)
@pytest.mark.parametrize("func", [ref_oas, oas])
def test_oas(benchmark, p, n, func):
    _, sc, _ = get_large_sample_cov(p=p, n=n, seed=42)
    _ = func(sc, n, p)  # Warm-up
    benchmark.group = f"OAS | (p={p}, n={n})"
    benchmark(func, sc, n, p)
