"""Defines the base classes for the package."""


class SimpleBaseEstimator:
    """Minimal base estimator used when scikit-learn is not installed.

    Provides the same ``fit`` / ``predict`` interface as
    ``sklearn.base.BaseEstimator`` so that subclasses work regardless of
    whether scikit-learn is available.

    Parameters
    ----------
    param : int, optional
        Placeholder parameter kept for API compatibility. Default is 1.
    """

    def __init__(self, *, param=1):
        self.param = param

    def fit(self, X, y=None):
        """Fit the estimator (no-op in the base class).

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like, optional
            Target values. Ignored.

        Returns
        -------
        self
        """

    def predict(self, X):
        """Generate predictions (no-op in the base class).

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        None
        """


BaseEstimator = SimpleBaseEstimator

try:
    import sklearn

    BaseEstimator = sklearn.base.BaseEstimator
except ImportError:
    pass
