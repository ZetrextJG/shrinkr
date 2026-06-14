import numpy as np

from shrinkr._native import py_oas


def oas(sample_cov: np.ndarray, n: int, p: int | None = None) -> tuple[np.ndarray, float]:
    """Oracle Approximating Shrinkage (OAS) covariance estimator.

    The formulation is based on [1]_.

    Parameters
    ----------
    sample_cov : np.ndarray
        Sample covariance matrix of shape (p, p).
    n : int
        Number of observations used to compute the sample covariance.
    p : int, optional
        Number of variables. If None, inferred from ``sample_cov``.

    Returns
    -------
    sample_cov_star : np.ndarray
        Shrinkage-regularized covariance matrix of shape (p, p).
    shrinkage : float
        Optimal shrinkage coefficient.

    References
    ----------
    .. [1] Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010).
       Shrinkage algorithms for MMSE covariance estimation.
       IEEE Transactions on Signal Processing, 58(10), 5016-5029.
       <https://arxiv.org/pdf/0907.4698.pdf>
    """
    if len(sample_cov.shape) != 2:
        raise ValueError("Expected a square numpy matrix")
    if sample_cov.shape[0] != sample_cov.shape[1]:
        raise ValueError("Expected a square numpy matrix")

    if p is None:
        p = sample_cov.shape[0]

    sample_cov: np.ndarray = sample_cov.astype(np.float64)
    sample_cov = np.ascontiguousarray(sample_cov)

    sample_cov_star, shrinakge = py_oas(sample_cov, n, p)
    return sample_cov_star, shrinakge
