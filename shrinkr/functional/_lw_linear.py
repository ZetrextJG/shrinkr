import numpy as np

from shrinkr._native import py_lw_linear


def lw_linear(X: np.ndarray, assume_centered: bool = False) -> tuple[np.ndarray, float]:
    r"""Ledoit-Wolf linear shrinkage estimator.

    The value of the shrinkage is constructed
    based on the Theorem 3.2 and Lemmata 3.2-3.5 from [1].

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
    $(1 - s) * S_c + s * \mu * I$,
    where $\mu = Tr{(S_c)} / n_\text{features}$, $s$ is the shrinkage value
    and $S_c$ is the sample covariance matrix.

    Returns
    -------
    sample_cov_star : np.ndarray
        Shrinkage-regularized covariance matrix of shape (n_features, n_features).
    shrinkage : float
        Optimal shrinkage coefficient.

    References
    ----------
    [^1]: Ledoit, O., & Wolf, M. (2004).
        A well-conditioned estimator for large-dimensional covariance matrices.
        Journal of multivariate analysis, 88(2), 365-411.
        <http://www.ledoit.net/ole1a.pdf>
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
