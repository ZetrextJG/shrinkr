import numpy as np
import pytest

from shrinkr.functional import lw_analytical, lw_linear, oas
from shrinkr.monte_carlo import get_large_sample_cov
from shrinkr.reference import ref_lw_analytical, ref_lw_linear, ref_oas

# Define test dimensions
TEST_CASES = [
    (50, 60, "Small (p < n)"),
    (70, 60, "Small (p > n)"),
    (180, 300, "Medium (p > n)"),
    (220, 300, "Medium (p > n)"),
    (500, 600, "Large (p < n)"),
    (700, 600, "Large (p > n)"),
]

# --- 1. Ledoit-Wolf Analytical ---


@pytest.mark.parametrize("p, n, desc", TEST_CASES)
@pytest.mark.parametrize("impl_name, func", [("ref", ref_lw_analytical), ("new", lw_analytical)])
def test_lw_analytical(benchmark, p, n, desc, impl_name, func):
    # Setup Data (this executes outside the timing loop)
    _, sc, _ = get_large_sample_cov(p=p, n=n, seed=42)
    lam, _ = np.linalg.eigh(sc)

    # Group tests in terminal output to easily compare "ref" vs "new"
    benchmark.group = f"LW_Analytical | {desc} (p={p}, n={n})"

    # Run the benchmark
    benchmark(func, lam, n)


# --- 2. Ledoit-Wolf Linear ---


@pytest.mark.parametrize("p, n, desc", TEST_CASES)
@pytest.mark.parametrize("impl_name, func", [("ref", ref_lw_linear), ("new", lw_linear)])
def test_lw_linear(benchmark, p, n, desc, impl_name, func):
    # Setup Data
    X, _, _ = get_large_sample_cov(p=p, n=n, seed=42)

    benchmark.group = f"LW_Linear | {desc} (p={p}, n={n})"

    # kwargs can be passed directly into the benchmark callable
    benchmark(func, X, assume_centered=False)


# --- 3. Oracle Approximating Shrinkage (OAS) ---


@pytest.mark.parametrize("p, n, desc", TEST_CASES)
@pytest.mark.parametrize("impl_name, func", [("ref", ref_oas), ("new", oas)])
def test_oas(benchmark, p, n, desc, impl_name, func):
    # Setup Data
    _, sc, _ = get_large_sample_cov(p=p, n=n, seed=42)

    benchmark.group = f"OAS | {desc} (p={p}, n={n})"

    benchmark(func, sc, n, p)
