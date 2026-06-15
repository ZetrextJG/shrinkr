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
    """Objective function of DEAL.

    Computes the optimization objective using deterministic equivalents.
    Requires solving a fixed-point equation for delta at a given gamma.

    Parameters
    ----------
    base_evals : np.ndarray
        First 1D array of eigenvalues for the objective (Those will be shrunk)
    surrogate_evals : np.ndarray
        Second 1D array of eigenvalues for the objective (Used to compute shrinkage paramters)
    z_vec : np.ndarray
        Vector of interest projected into the eigenvector space.
    gamma : float
        The value of gamma to evaluate. During optimization only this value changes.
    n : int
        Effective number of samples used to compute the empirical covariance matrix.
    start_value : float, optional
        Starting value of delta for the fixed point iteration method used by for the objective.

    See Also
    --------
    [`shrinkr.functional.deal`][]
        function for more information about the DEAL method.

    Returns
    -------
    float
        The DEAL objective estimate.
    """
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
    r"""DEAL (Deterministic Equivalents for Adaptive LDA) shrinkage.

    Parameters
    ----------
    evals : np.ndarray
        Eigenvalues of the empirical covariance matrix.
    z_vec : np.ndarray
        Vector of interest projected into the eigenvector space.
    n_eff : int
        Effective number of samples used to compute the empirical covariance matrix.
    gamma_min : float, optional
        Minimum value for the gamma bounded search. Default is 0.02.
    gamma_max : float, optional
        Maximum value for the gamma bounded search. Default is 100.
    base_shrinkage : {'lw_analytical', 'empirical'}, optional
        Shrinkage method for the base eigenvalue estimation. Default is 'lw_analytical'.
    surrogate_shrinkage : {'lw_analytical', 'empirical'}, optional
        Shrinkage method for the surrogate eigenvalue estimation. Default is 'lw_analytical'.
    eps : float, optional
        Epsilon for numerical stability. Default is 1e-8.

    Notes
    -----
    The DEAL method utilizes the Random Matrix Theory (RMT) [1] to
    construct an estimate and optimize an objective
    defined as an expectation over the data distribution of
    $|| \hat\Sigma^{-1} \mu - \Sigma^{-1} \mu ||_\Sigma^2$ where $\hat\Sigma$
    is an optimal linear correction to any non-directional shrinkage,
    $\Sigma$ is the True Population Covariance, $\mu$ is a constant vector
    of interest and $|| \cdot ||_\Sigma$ is the Mahalanobis distance
    based on the matrix $\Sigma$. More details in a future paper.

    Returns
    -------
    np.ndarray
        Shrinkage-adjusted eigenvalues.

    References
    ----------
    [^1]: Hachem, W., Loubaton, P., Najim, J., & Vallet, P. (2013).
        On bilinear forms based on the resolvent of large random matrices.
        In Annales de l'IHP Probabilités et statistiques (Vol. 49, No. 1, pp. 36-63).
        <https://www.numdam.org/article/AIHPB_2013__49_1_36_0.pdf>
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
