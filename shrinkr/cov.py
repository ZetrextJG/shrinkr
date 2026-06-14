"""Defines the Covariance Estimator with selectable shrinkage method."""

import numpy as np

from shrinkr.base import BaseEstimator
from shrinkr.functional import lw_analytical, lw_linear, oas
from shrinkr.reference import ref_lw_analytical, ref_lw_linear, ref_oas

C_METHODS = ["lw_linear", "lw_analytical", "oas"]
REF_METHODS = [f"ref_{name}" for name in C_METHODS]
METHODS = ["empirical"] + C_METHODS + REF_METHODS


class CovarianceEstimator(BaseEstimator):
    """Covariance matrix estimator with optional shrinkage.

    Wraps several shrinkage methods behind a scikit-learn-compatible
    ``fit`` / ``predict`` interface.

    Parameters
    ----------
    method : str, optional
        Shrinkage method to apply. One of ``'empirical'``, ``'lw_linear'``,
        ``'lw_analytical'``, ``'oas'``, or their ``'ref_*'`` reference
        counterparts. Default is ``'empirical'``.
    tol : float, optional
        Eigenvalue threshold passed to eigenvalue-based methods. Default is 1e-8.

    Attributes
    ----------
    is_fitted_ : bool
        True after [`fit`][shrinkr.cov.CovarianceEstimator.fit] has been called successfully.
    data_ : np.ndarray
        The data passed to [`fit`][shrinkr.cov.CovarianceEstimator.fit].
    cov_ : np.ndarray or None
        Raw sample covariance matrix. Set only for methods that require it.
    shrunk_cov_ : np.ndarray or None
        Shrinkage-regularized covariance matrix produced by [`fit`][shrinkr.cov.CovarianceEstimator.fit].
    """

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
        """Compute the (shrunk) covariance matrix from data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : CovarianceEstimator
            Fitted estimator.
        """
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
            self.shrunk_cov_, _ = oas(self.cov_, n, p)
            self.is_fitted_ = True
            return self

        if self.method == "ref_oas":
            self.shrunk_cov_, _ = ref_oas(self.cov_, n, p)
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
        """Return the fitted covariance matrix.

        Parameters
        ----------
        X : ignored
            Present for API compatibility.

        Returns
        -------
        np.ndarray
            The shrinkage-regularized covariance matrix, or the raw sample
            covariance if no shrinkage method produced a result.
        """
        if not self.is_fitted_:
            raise ValueError("This CovarianceEstimator instance is not fitted yet.")

        if self.shrunk_cov_ is not None:
            return self.shrunk_cov_
        return self.cov_

    def fit_predict(self, X, y=None):
        """Fit and immediately return the covariance matrix.

        Equivalent to calling [`fit`][shrinkr.cov.CovarianceEstimator.fit] followed by [`predict`][shrinkr.cov.CovarianceEstimator.predict].

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
        y : ignored
            Present for API compatibility.

        Returns
        -------
        np.ndarray
            The shrinkage-regularized covariance matrix.
        """
        return self.fit(X, y).predict(X)
