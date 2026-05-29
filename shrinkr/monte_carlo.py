import numpy as np


def get_small_sample_cov(n: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a small sample from multivariance normal distribtuion

    Args:
        n: A number of sample to generate
        seed: A seed for the random number generator

    Returns (in a triple):
        X: (n, 2) shaped matrix
        sample_cov: (2, 2) sample covariance matrix
        real_cov: (2, 2) population (real) covariance matrix

    """

    real_cov: np.ndarray = np.array([[0.4, 0.2], [0.2, 0.8]])
    rng = np.random.RandomState(seed)
    X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=n)
    sample_cov = np.cov(X, rowvar=False)
    return X, sample_cov, real_cov


def get_large_sample_cov(p: int = 20, n: int = 200, seed: int = 0):
    """Generate random Gaussian data with a random PSD covariance matrix."""
    rng = np.random.default_rng(seed)

    # Create real cov
    A = rng.standard_normal((p, p))
    real_cov = A @ A.T
    real_cov /= np.trace(real_cov)

    real_cov += 1e-4

    # Empirical cov
    X = rng.multivariate_normal(mean=np.zeros(p), cov=real_cov, size=n)
    sample_cov = np.cov(X, rowvar=False)

    return X, sample_cov, real_cov
