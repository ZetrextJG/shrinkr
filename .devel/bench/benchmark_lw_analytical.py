import time

import numpy as np

from shrinkr.functional import lw_analytical_shrinkage
from shrinkr.monte_carlo import get_large_sample_cov
from shrinkr.reference import (
    lw_analytical_shrinkage as ref_lw_analytical_shrinkage,
)


def benchmark_shrinkage_implementations(iterations: int = 100) -> None:
    """
    Benchmarks the reference and optimized analytical shrinkage implementations
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

    # Print table header
    print(
        f"{'Test Case':<15} | {'Dimensions':<15} | {'Ref Time (s)':<12} | {'New Time (s)':<12} | {'Speedup'}"
    )
    print("-" * 75)

    for case in test_cases:
        p: int = case["p"]
        n: int = case["n"]

        # 1. Setup Data
        _, sc, _ = get_large_sample_cov(p=p, n=n, seed=42)
        lam, _ = np.linalg.eigh(sc)

        # Optional: Warm-up runs to compile JIT/C-extensions if applicable
        ref_lw_analytical_shrinkage(lam, n)
        lw_analytical_shrinkage(lam, n)

        # 2. Benchmark Reference Implementation
        start_ref = time.perf_counter()
        for _ in range(iterations):
            ref_lw_analytical_shrinkage(lam, n)
        ref_time = time.perf_counter() - start_ref

        # 3. Benchmark New/Optimized Implementation
        start_new = time.perf_counter()
        for _ in range(iterations):
            lw_analytical_shrinkage(lam, n)
        new_time = time.perf_counter() - start_new

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
