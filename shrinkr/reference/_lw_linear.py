# Code fragment adapted from the Scikit-Learn implementation
# https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/covariance/_shrunk_covariance.py
#

import numpy as np


def ref_lw_linear(
    X: np.ndarray, assume_centered: bool = False, block_size: int = 1000
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf linear shrinkage estimator (reference implementation).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    assume_centered : bool, optional
        If True, data is not mean-centered before computing the covariance.
        Default is False.
    block_size : int, optional
        Size of blocks into which the covariance matrix is split for computation.
        Default is 1000.

    Returns
    -------
    sample_cov_star : np.ndarray
        Shrinkage-regularized covariance matrix of shape (n_features, n_features).
    shrinkage : float
        Optimal shrinkage coefficient.

    See Also
    --------
    [`shrinkr.functional.lw_linear`][]
        Optimized implementation of this method.
        Go there for additional notes and references.
        Functions ref_* are reference implementations intended for validation.

    """
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        return np.var(X.reshape(-1)).reshape(1, 1), 0.0
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        print("Only one sample available. You may want to reshape your data array")
    n_samples, n_features = X.shape

    # optionally center data
    if not assume_centered:
        X = X - X.mean(0)

    sample_cov = (X.T @ X) / X.shape[0]
    # A non-blocked version of the computation is present in the tests
    # in tests/test_covariance.py

    # number of blocks to split the covariance matrix into
    n_splits = int(n_features / block_size)
    X2 = X**2
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features

    beta_ = 0.0  # sum of the coefficients of <X2.T, X2>
    delta_ = 0.0  # sum of the *squared* coefficients of <X.T, X>
    # starting block computation
    for i in range(n_splits):
        for j in range(n_splits):
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            beta_ += np.sum(np.dot(X2.T[rows], X2[:, cols]))
            delta_ += np.sum(np.dot(X.T[rows], X[:, cols]) ** 2)
        rows = slice(block_size * i, block_size * (i + 1))
        beta_ += np.sum(np.dot(X2.T[rows], X2[:, block_size * n_splits :]))
        delta_ += np.sum(np.dot(X.T[rows], X[:, block_size * n_splits :]) ** 2)

    for j in range(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        beta_ += np.sum(np.dot(X2.T[block_size * n_splits :], X2[:, cols]))
        delta_ += np.sum(np.dot(X.T[block_size * n_splits :], X[:, cols]) ** 2)

    delta_ += np.sum(np.dot(X.T[block_size * n_splits :], X[:, block_size * n_splits :]) ** 2)
    delta_ /= n_samples**2
    beta_ += np.sum(np.dot(X2.T[block_size * n_splits :], X2[:, block_size * n_splits :]))

    # use delta_ to compute beta
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    # delta is the sum of the squared coefficients of (<X.T,X> - mu*Id) / p
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu**2
    delta /= n_features
    # get final beta as the min between beta and delta
    # We do this to prevent shrinking more than "1", which would invert
    # the value of covariances
    beta = min(beta, delta)
    # finally get shrinkage
    shrinkage = 0 if beta == 0 else beta / delta

    sample_cov_star = ((1 - shrinkage) * sample_cov) + (shrinkage * mu * np.eye(n_features))

    return sample_cov_star, shrinkage
