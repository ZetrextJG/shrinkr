import numpy as np

from shrinkr._native import py_lw_linear


def lw_linear(
    X: np.ndarray, assume_centered: bool = False, use_fast: bool = True
) -> tuple[np.ndarray, float]:

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
