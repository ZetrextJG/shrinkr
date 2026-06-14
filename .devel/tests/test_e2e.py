import pytest
from sklearn.metrics import accuracy_score

from shrinkr import CovarianceEstimator, LinearDiscriminantAnalysis
from shrinkr.cov import METHODS
from shrinkr.lda import LDA_METHODS
from shrinkr.monte_carlo import get_guassian_lda_samples, get_large_sample_cov


@pytest.mark.unit
def test_all_covariance_estimators():
    p, n = 20, 100
    X, _, _ = get_large_sample_cov(p, n)

    for method in METHODS:
        cov_estimator = CovarianceEstimator(method=method)
        cov = cov_estimator.fit_predict(X)
        assert cov.shape == (p, p)


@pytest.mark.unit
def test_all_lda_estimators():
    p, n = 20, 100
    X, y = get_guassian_lda_samples(p, n)

    for method in LDA_METHODS:
        lda_classifier = LinearDiscriminantAnalysis(method=method)
        y_pred = lda_classifier.fit_predict(X, y)
        assert y.shape == y_pred.shape
        # This task is so easy that we expect acc > 0.95
        acc_score = accuracy_score(y, y_pred)
        assert acc_score > 0.95
