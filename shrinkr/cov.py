import numpy as np

from shrinkr.base import BaseEstimator
from shrinkr.functional import lw_analytical, lw_linear, oas
from shrinkr.reference import ref_lw_analytical, ref_lw_linear, ref_oas

C_METHODS = ["lw_linear", "lw_analytical", "oas"]
REF_METHODS = [f"ref_{name}" for name in C_METHODS]
METHODS = ["empirical"] + C_METHODS + REF_METHODS


class CovarianceEstimator(BaseEstimator):
    def __init__(self, *, param=1, method: str = "empirical", tol: float = 1e-8):
        self.param = param
        self.tol = tol

        if method not in METHODS:
            raise ValueError(f"Method '{method}' must be one of: {METHODS}")
        self.method = method

        self.is_fitted_: bool = False
        self.data_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None
        self.shrunk_cov_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        self.data_ = X
        n, p = X.shape

        ## Methods requiring only data
        if self.method == "lw_linear":
            self.shrunk_cov_, _ = lw_linear(X, assume_centered=False)
            self.is_fitted_ = True
            return self

        if self.method == "ref_lw_linear":
            self.shrunk_cov_, _ = ref_lw_linear(X, assume_centered=False)
            self.is_fitted_ = True
            return self

        ## Methods requiring standard covariance
        self.cov_ = np.cov(X, rowvar=False, bias=True)

        if self.method == "empirical":
            self.shrunk_cov_ = self.cov_  # Empirical usually means unshrunk
            self.is_fitted_ = True
            return self

        if self.method == "oas":
            self.shrunk_cov_ = oas(self.cov_, n, p)
            self.is_fitted_ = True
            return self

        if self.method == "ref_oas":
            self.shrunk_cov_ = ref_oas(self.cov_, n, p)
            self.is_fitted_ = True
            return self

        ## Methods requiring eigenvalue decomposition
        evals, U = np.linalg.eigh(self.cov_)

        if self.method == "lw_analytical":
            shrunk_evals = lw_analytical(evals, n, p, self.tol)
            self.shrunk_cov_ = U @ np.diag(shrunk_evals) @ U.T
        elif self.method == "ref_lw_analytical":
            shrunk_evals = ref_lw_analytical(evals, n, self.tol)
            self.shrunk_cov_ = U @ np.diag(shrunk_evals) @ U.T
        else:
            raise ValueError(f"Implementation for method '{self.method}' not found.")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("This CovarianceEstimator instance is not fitted yet.")

        if self.shrunk_cov_ is not None:
            return self.shrunk_cov_
        return self.cov_

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)
