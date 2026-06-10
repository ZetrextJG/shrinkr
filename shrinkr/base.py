class SimpleBaseEstimator:
    def __init__(self, *, param=1):
        self.param = param

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass


BaseEstimator = SimpleBaseEstimator

try:
    import sklearn

    BaseEstimator = sklearn.base.BaseEstimator
except ImportError:
    pass
