"""Submodule with simple monte carlo methods for getting testing data."""

import numpy as np


def get_small_sample_cov(n: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a small sample from a fixed 2-D multivariate normal distribution.

    The population covariance is ``[[0.4, 0.2], [0.2, 0.8]]``.

    Parameters
    ----------
    n : int, optional
        Number of samples to generate. Default is 50.
    seed : int, optional
        Seed for ``numpy.random.RandomState``. Default is 0.

    Returns
    -------
    X : np.ndarray of shape (n, 2)
        Simulated data matrix.
    sample_cov : np.ndarray of shape (2, 2)
        Sample covariance matrix computed from ``X``.
    real_cov : np.ndarray of shape (2, 2)
        Population (true) covariance matrix.
    """
    real_cov: np.ndarray = np.array([[0.4, 0.2], [0.2, 0.8]])
    rng = np.random.RandomState(seed)
    X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=n)
    sample_cov = np.cov(X, rowvar=False)
    return X, sample_cov, real_cov


def get_large_sample_cov(
    p: int = 20, n: int = 200, seed: int = 0, add_diagonal: float = 1e-1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Gaussian data with a random positive semi-definite covariance matrix.

    The population covariance is constructed as ``A @ A.T`` (normalized to unit
    trace) plus a small ridge ``add_linear`` for numerical stability.

    Parameters
    ----------
    p : int, optional
        Number of features (dimension). Default is 20.
    n : int, optional
        Number of samples. Default is 200.
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Default is 0.
    add_diagonal : float, optional
        Value added to the diagonal of the true covariance matrix. Default is 1e-1.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Simulated data matrix.
    sample_cov : np.ndarray of shape (p, p)
        Sample covariance matrix computed from ``X``.
    real_cov : np.ndarray of shape (p, p)
        Population (true) covariance matrix.
    """
    rng = np.random.default_rng(seed)

    # Create real cov
    A = rng.standard_normal((p, p))
    real_cov = A @ A.T
    real_cov /= np.trace(real_cov)

    real_cov.flat[:: (p + 1)] += add_diagonal

    # Empirical cov
    X = rng.multivariate_normal(mean=np.zeros(p), cov=real_cov, size=n)
    sample_cov = np.cov(X, rowvar=False)

    return X, sample_cov, real_cov


def get_guassian_lda_samples(p: int = 20, n_per_class: int = 100, seed: int = 0):
    """Generate data for Gaussian LDA with two classes. Balanced dataset.

    Parameters
    ----------
    p : int, optional
        Number of features (dimension). Default is 20.
    n_per_class : int, optional
        Number of samples per class. Default is 100.
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Default is 0.
    """
    _, _, cov = get_large_sample_cov(p, n_per_class, seed)

    rng = np.random.default_rng(seed)

    mu = rng.standard_normal(p)  # The difference in means vector
    mu: np.ndarray = mu / np.linalg.vector_norm(mu)
    mu0 = -mu / 2.0
    mu1 = mu / 2.0

    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    X0 = rng.multivariate_normal(mean=mu0, cov=cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean=mu1, cov=cov, size=n_per_class)
    X = np.concatenate([X0, X1], axis=0)

    return X, y
