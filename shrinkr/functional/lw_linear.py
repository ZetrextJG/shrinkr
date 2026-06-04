import numpy as np


def lw_linear(X: np.ndarray, assume_centered: bool = False, block_size: int = 1000) -> np.ndarray:

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

    sample_cov = np.cov(X, rowvar=False)

    return sample_cov
