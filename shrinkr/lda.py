import numpy as np

from shrinkr.base import BaseEstimator
from shrinkr.cov import METHODS, CovarianceEstimator

LDA_ONLY_METHODS = ["deal", "ref_deal"]
LDA_METHODS = METHODS + LDA_ONLY_METHODS


# TODO: Add DEAL
class LinearDiscriminantAnalysis(BaseEstimator):
    def __init__(self, covariance_estimator=None):
        self.covariance_estimator = covariance_estimator

        self.classes_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        if self.covariance_estimator is None:
            self.covariance_estimator = CovarianceEstimator(method="empirical")

        self.classes_, y_indices = np.unique(y, return_inverse=True)

        if len(self.classes_) != 2:
            raise ValueError(f"Number of classes must be 2, but found {len(self.classes_)}.")

        n_samples, n_features = X.shape
        self.priors_ = np.bincount(y_indices) / n_samples
        self.means_ = np.zeros((2, n_features))

        X_centered = np.empty_like(X, dtype=float)
        for idx, cls in enumerate(self.classes_):
            class_mask = y == cls
            X_class = X[class_mask]
            self.means_[idx] = np.mean(X_class, axis=0)
            X_centered[class_mask] = X_class - self.means_[idx]

        self.covariance_estimator.fit(X_centered)
        self.covariance_ = self.covariance_estimator.predict(X_centered)

        self.precision_ = np.linalg.pinv(self.covariance_)

        w = self.precision_ @ (self.means_[1] - self.means_[0])
        b = -0.5 * (
            self.means_[1] @ self.precision_ @ self.means_[1]
            - self.means_[0] @ self.precision_ @ self.means_[0]
        ) + np.log(self.priors_[1] / self.priors_[0])

        self.coef_ = np.array([w])
        self.intercept_ = np.array([b])
        self.is_fitted_ = True

        return self

    def decision_function(self, X: np.ndarray):
        """Returns the log-odds for the positive class."""
        if not self.is_fitted_:
            raise ValueError("This estimator is not fitted yet.")
        return (X @ self.coef_.T + self.intercept_).ravel()

    def predict(self, X: np.ndarray):
        """Predicts binary class labels for X."""
        scores = self.decision_function(X)
        indices = (scores > 0).astype(int)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray):
        """Estimates class probabilities using the logistic sigmoid function."""
        scores = self.decision_function(X)
        prob_1 = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - prob_1, prob_1]).T
