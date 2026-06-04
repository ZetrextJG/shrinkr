import numpy as np

from shrinkr._native import py_oas


def oas(sample_cov: np.ndarray, n: int, p: int | None = None) -> np.ndarray:
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    The formulation is based on [1]_.
    [1] "Shrinkage algorithms for MMSE covariance estimation.",
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
        https://arxiv.org/pdf/0907.4698.pdf
    """
    if len(sample_cov.shape) != 2:
        raise ValueError("Expected a square numpy matrix")
    if sample_cov.shape[0] != sample_cov.shape[1]:
        raise ValueError("Expected a square numpy matrix")

    if p is None:
        p = sample_cov.shape[0]

    sample_cov: np.ndarray = sample_cov.astype(np.float64)
    sample_cov = np.ascontiguousarray(sample_cov)

    sample_cov_star = py_oas(sample_cov, n, p)

    return sample_cov_star
