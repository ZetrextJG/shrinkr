import numpy as np

from shrinkr._native import py_lw_linear


def lw_linear(X: np.ndarray, assume_centered: bool = False) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf linear shrinkage estimator.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    assume_centered : bool, optional
        If True, data is not mean-centered before computing the covariance.
        Default is False.

    Notes
    -----
    The regularized covariance is:

    ``(1 - shrinkage) * cov + shrinkage * mu * I``

    where ``mu = trace(cov) / n_features``.

    Returns
    -------
    sample_cov_star : np.ndarray
        Shrinkage-regularized covariance matrix of shape (n_features, n_features).
    shrinkage : float
        Optimal shrinkage coefficient.
    """
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        return np.var(X.reshape(-1)).reshape(1, 1)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        print("Only one sample available. You may want to reshape your data array")

    n, p = X.shape

    # optionally center data
    if not assume_centered:
        X = X - X.mean(0)

    sample_cov = np.dot(X.T, X) / n
    sample_cov_star, shrinkage = py_lw_linear(X, sample_cov, n, p)

    return sample_cov_star, shrinkage
