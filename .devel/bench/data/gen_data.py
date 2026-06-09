from pathlib import Path

import numpy as np

from shrinkr.monte_carlo import get_large_sample_cov

OUTPUT_DIR = Path(__file__).parent

SIZES = [(20, 40), (500, 1000), (1000, 2000)]


def main():
    seed = 42
    for p, n in SIZES:
        X, sample_cov, real_cov = get_large_sample_cov(p=p, n=n, seed=seed)
        eigenvalues, _ = np.linalg.eigh(sample_cov)

        X.astype(np.float64).flatten().tofile(OUTPUT_DIR / f"data_{p}_{n}.bin")
        sample_cov.astype(np.float64).flatten().tofile(OUTPUT_DIR / f"sample_cov_{p}_{n}.bin")
        eigenvalues.astype(np.float64).flatten().tofile(OUTPUT_DIR / f"eigenvalues_{p}_{n}.bin")


if __name__ == "__main__":
    main()
