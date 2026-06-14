"""Defines the Linear Discriminant Analysis Classifier with selectable shrinkage method."""

import numpy as np

from shrinkr.base import BaseEstimator
from shrinkr.cov import METHODS, CovarianceEstimator

LDA_ONLY_METHODS = ["deal", "ref_deal"]
LDA_METHODS = METHODS + LDA_ONLY_METHODS


# TODO: Add DEAL
class LinearDiscriminantAnalysis(BaseEstimator):
    """Binary Linear Discriminant Analysis with pluggable covariance shrinkage.

    Fits a two class LDA model and classifies by the
    log-posterior ratio. The pooled covariance can be estimated with any
    method supported by [`CovarianceEstimator`][shrinkr.cov.CovarianceEstimator].

    Parameters
    ----------
    covariance_estimator : CovarianceEstimator, optional
        Estimator used to compute the pooled within-class covariance.
        If None, an empirical (unshrunk) covariance is used.

    Attributes
    ----------
    classes_ : np.ndarray of shape (2,)
        The two class labels seen during [`fit`][shrinkr.lda.LinearDiscriminantAnalysis.fit].
    priors_ : np.ndarray of shape (2,)
        Class prior probabilities estimated from the training data.
    means_ : np.ndarray of shape (2, n_features)
        Per-class sample means.
    covariance_ : np.ndarray of shape (n_features, n_features)
        Pooled within-class covariance matrix (after shrinkage if applicable).
    precision_ : np.ndarray of shape (n_features, n_features)
        Pseudo-inverse of `covariance_`.
    coef_ : np.ndarray of shape (1, n_features)
        Linear discriminant direction.
    intercept_ : np.ndarray of shape (1,)
        Decision boundary offset.
    is_fitted_ : bool
        True after [`fit`][shrinkr.lda.LinearDiscriminantAnalysis.fit] has been called successfully.
    """

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
        """Fit the LDA model on labelled training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Binary class labels. Exactly two distinct values must be present.

        Returns
        -------
        self : LinearDiscriminantAnalysis
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, ``X`` and ``y`` have different lengths, or
            ``y`` does not contain exactly two classes.
        """
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
        """Compute the log-odds score for the positive class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Log-odds of the positive class for each sample.
        """
        if not self.is_fitted_:
            raise ValueError("This estimator is not fitted yet.")
        return (X @ self.coef_.T + self.intercept_).ravel()

    def predict(self, X: np.ndarray):
        """Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class label for each sample.
        """
        scores = self.decision_function(X)
        indices = (scores > 0).astype(int)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray):
        """Estimate class probabilities using the logistic sigmoid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Columns are ``[P(class 0), P(class 1)]`` for each sample.
        """
        scores = self.decision_function(X)
        prob_1 = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - prob_1, prob_1]).T
