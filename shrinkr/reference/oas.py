# Code fragment adapted from the Scikit-Learn implementation
# https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cdbd75ae4e662fda67dcb19277258792b/sklearn/covariance/_shrunk_covariance
#
import numpy as np


def ref_oas(sample_cov: np.ndarray, n: int, p: int | None = None) -> tuple[np.ndarray, float]:
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

    alpha = np.mean(sample_cov**2)
    mu = np.trace(sample_cov) / p
    mu_squared = mu**2

    # The factor 1 / p**2 will cancel out since it is in both the numerator and
    # denominator
    num = alpha + mu_squared
    den = (n + 1) * (alpha - mu_squared / p)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)

    # The shrunk covariance is defined as:
    # (1 - shrinkage) * S + shrinkage * F (cf. Eq. 4 in [1])
    # where S is the empirical covariance and F is the shrinkage target defined as
    # F = trace(S) / n_features * np.identity(n_features) (cf. Eq. 3 in [1])
    shrunk_cov = (1.0 - shrinkage) * sample_cov
    shrunk_cov.flat[:: (p + 1)] += shrinkage * mu

    return shrunk_cov, shrinkage
