import gc
import time
from collections.abc import Callable

import numpy as np

from shrinkr.functional import lw_analytical, lw_linear, oas
from shrinkr.monte_carlo import get_large_sample_cov
from shrinkr.reference import ref_lw_analytical, ref_lw_linear, ref_oas


def time_function(func: Callable, args: tuple, iterations: int = 100, repeats: int = 5) -> float:
    """Runs a function multiple times and returns the best time, handling GC."""
    best_time = float("inf")

    for _ in range(repeats):
        # Disable GC before the timed loop
        gc.disable()

        start = time.perf_counter()
        for _ in range(iterations):
            func(*args)
        elapsed = time.perf_counter() - start

        # Re-enable GC immediately after
        gc.enable()

        if elapsed < best_time:
            best_time = elapsed

    return best_time


def benchmark_shrinkage_implementations(iterations: int = 50, repeats: int = 8) -> None:
    """
    Benchmarks the reference and optimized shrinkage implementations
    across various matrix dimensions (both p < n and p > n).

    Args:
        iterations: Number of times to run each function per test case.
    """
    test_cases = [
        {"p": 50, "n": 60, "desc": "p < n (Small)"},
        {"p": 70, "n": 60, "desc": "p > n (Small)"},
        {"p": 180, "n": 300, "desc": "p > n (Medium)"},
        {"p": 220, "n": 300, "desc": "p > n (Medium)"},
        {"p": 500, "n": 600, "desc": "p < n (Large)"},
        {"p": 700, "n": 600, "desc": "p > n (Large)"},
    ]

    # Group algorithms into suites to test them systematically.
    # Lambda wrappers normalize the inputs and let us implicitly discard return values.
    benchmark_suites = [
        {
            "name": "Ledoit-Wolf Analytical",
            "ref": lambda X, sc, lam, n, p: ref_lw_analytical(lam, n),
            "new": lambda X, sc, lam, n, p: lw_analytical(lam, n),
        },
        {
            "name": "Ledoit-Wolf Linear",
            "ref": lambda X, sc, lam, n, p: ref_lw_linear(X, assume_centered=False),
            "new": lambda X, sc, lam, n, p: lw_linear(X, assume_centered=False),
        },
        {
            "name": "Oracle Approximating Shrinkage (OAS)",
            "ref": lambda X, sc, lam, n, p: ref_oas(sc, n, p),
            "new": lambda X, sc, lam, n, p: oas(sc, n, p),
        },
    ]

    for suite in benchmark_suites:
        print(f"\n--- Benchmarking: {suite['name']} ---")
        print(
            f"{'Test Case':<15} | {'Dimensions':<15} | {'Ref Time (s)':<12} | {'New Time (s)':<12} | {'Speedup'}"
        )
        print("-" * 75)

        for case in test_cases:
            p: int = case["p"]
            n: int = case["n"]

            # 1. Setup Data
            # Note: Assuming get_large_sample_cov returns (X, sample_covariance, true_covariance)
            X, sc, _ = get_large_sample_cov(p=p, n=n, seed=42)
            lam, _ = np.linalg.eigh(sc)

            ref_func: Callable = suite["ref"]
            new_func: Callable = suite["new"]

            # Optional: Warm-up runs to compile JIT/C-extensions if applicable
            ref_func(X, sc, lam, n, p)
            new_func(X, sc, lam, n, p)

            # 2. Benchmark Reference Implementation
            args = (X, sc, lam, n, p)
            ref_time = time_function(ref_func, args, iterations, repeats)
            new_time = time_function(new_func, args, iterations, repeats)

            # 4. Calculate Speedup
            speedup = ref_time / new_time if new_time > 0 else float("inf")

            # Print results row
            dim_str = f"p={p}, n={n}"
            print(
                f"{case['desc']:<15} | {dim_str:<15} | {ref_time:<12.5f} | {new_time:<12.5f} | {speedup:.2f}x"
            )


# Run the benchmark
if __name__ == "__main__":
    benchmark_shrinkage_implementations(iterations=100)
