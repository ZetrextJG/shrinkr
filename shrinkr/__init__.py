"""Shrinkr package."""

from . import _native, functional, reference
from .cov import CovarianceEstimator
from .lda import LinearDiscriminantAnalysis

__all__ = [
    "CovarianceEstimator",
    "LinearDiscriminantAnalysis",
    "functional",
    "reference",
    "_native",
]
