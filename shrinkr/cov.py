import numpy as np

from shrinkr.base import BaseEstimator
from shrinkr.functional import lw_analytical, lw_linear, oas
from shrinkr.reference import ref_lw_analytical, ref_lw_linear, ref_oas

C_METHODS = ["lw_linear", "lw_analytical", "oas"]
REF_METHODS = [f"ref_{name}" for name in C_METHODS]
METHODS = ["empirical"] + C_METHODS + REF_METHODS


class EmpiricalCovairance(BaseEstimator):
    def __init__(self, *, param=1, method: str):
        self.param = param
        self.is_fitted: bool = False
        self.data: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.shrunk_cov: np.ndarray | None = None

        assert method in METHODS + REF_METHODS
        self.method = method

    def fit(self, X: np.ndarray, y=None):
        assert len(X.shape) == 2
        self.data = X
        self.is_fitted = True

        ## Only uses data

        if self.method == "lw_linear":
            self.shrunk_cov, _ = lw_linear(X, assume_centered=False)
            return None
        if self.method == "ref_lw_linear":
            self.shrunk_cov, _ = ref_lw_linear(X, assume_centered=False)
            return None

        ## Also needs cov already computed

        n, p = X.shape
        self.cov = np.cov(X, rowvar=False, bias=True)

        if self.method == "empirical":
            return None
        if self.method == "oas":
            self.shrunk_cov = oas(self.cov, n, p)
            return None
        if self.method == "ref_oas":
            self.shrunk_cov = ref_oas(self.cov, n, p)
            return None

        ## Also needs eigenvalue decomposition

        evals, U = np.linalg.eigh(self.cov)
        if self.method == "lw_analytical":
            shrunk_evals = lw_analytical(evals, n, p, 1e-8)
            self.shrunk_cov = U @ np.diag(shrunk_evals) @ (U.T)
            return None
        if self.method == "ref_lw_analytical":
            shrunk_evals = ref_lw_analytical(evals, n, 1e-8)
            self.shrunk_cov = U @ np.diag(shrunk_evals) @ (U.T)
            return None

    def predict(self, X):
        if self.is_fitted:
            return None
        if self.shrunk_cov:
            return self.shrunk_cov
        return self.cov

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)
