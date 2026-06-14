from shrinkr import CovarianceEstimator
from shrinkr import LinearDiscriminantAnalysis as LDA
from shrinkr.functional import accuracy
from shrinkr.monte_carlo import get_guassian_lda_samples

# Generate Guassian data for covariance estimation and LDA
X, y = get_guassian_lda_samples(p=20, n_per_class=200, seed=1)

# Shrunk covariance estimation:
# Methods like LW Linear, OAS, LW Analytical.
total_covariance = CovarianceEstimator(method="lw_linear").fit_predict(X)
assert total_covariance.shape == (20, 20)

# Linear Discriminant Analysis with Shrunk covariance estimation:
# Supports all methods from CovarianceEstimator
# but also LDA specialized shrinkages like DEAL.
classifier = LDA(method="deal")
classifier.fit(X, y)
y_pred = classifier.predict(X)
print(accuracy(y, y_pred))  # 1.0, quite a simple task
