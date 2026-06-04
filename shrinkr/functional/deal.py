from typing import Literal

import numpy as np

from shrinkr._native import py_deal, py_deal_objective, py_lw_analytical

EigenvalueOptions = Literal["lw_analytical", "empirical"]


def deal_objective(
    base_evals: np.ndarray,
    surrogate_evals: np.ndarray,
    z_vec: np.ndarray,
    gamma: float,
    n: int,
    start_value: float = 1.0,
):
    p = int(base_evals.shape[0])
    return py_deal_objective(base_evals, surrogate_evals, z_vec, gamma, start_value, n, p)


def deal(
    evals: np.ndarray,
    z_vec: np.ndarray,
    n_eff: int,
    gamma_min: float = 0.02,
    gamma_max: float = 100,
    base_shrinkage: EigenvalueOptions = "lw_analytical",
    surrogate_shrinkage: EigenvalueOptions = "lw_analytical",
    eps=1e-8,
    **kwargs,
):
    """
    The DEAL (Deterministic equivalents for Adaptive LDA) method.

    Args:
        evals - The eigenvales of the empirical covariance matrix
        z_vec - Vector of interest projected to the eigenvector space
        n_eff - the effective number of samples used to compute the empirical covariance matrix
        gamma_min, gamma_max - minimum and maximum values for the gamma grid search / bounded search
        base_shrinkage - which shrinkage to use for the base eigenvalue estimation: lw_analytical, linear, empirical
        surrogate_shrinkage - which shrinkage to use for the surrogate eigenvalue estimation: lw_analytical, linear, empirical
        eps - epsilon value

    Returns:
        - The Adjusted LDA vector,
        - The grid of gamma values used (if optimizer == grid else None),
        - The objective values for each of the grid value (if optimizer == grid else None)

    """
    # Rescale eigenvalues to Trace p
    evals /= np.mean(evals)
    p = evals.shape[0]

    # Compute (non-)linear shrinkages
    orig_evals = evals.copy()
    lw_nl_evals = py_lw_analytical(evals, n_eff, p, eps**2)

    # Select which eigenvalues to use for the base
    if base_shrinkage == "lw_analytical":
        evals = lw_nl_evals
    elif base_shrinkage == "empirical":
        evals = orig_evals
    else:
        raise ValueError("Unknown base shrinkage")

    # ... and the surrogate estimation
    if surrogate_shrinkage == "lw_analytical":
        surrogate_evals = lw_nl_evals
    elif surrogate_shrinkage == "empirical":
        surrogate_evals = evals
    else:
        raise ValueError("Unknown surrogate shrinkage")

    shrinkage = py_deal(evals, surrogate_evals, z_vec, gamma_min, gamma_max, n_eff, p)

    return evals + shrinkage
