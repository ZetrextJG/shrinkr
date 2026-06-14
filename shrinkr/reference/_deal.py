from typing import Literal

import numpy as np

from ._lw_analytical import ref_lw_analytical


def ref_deal_objective(
    base_evals: np.ndarray,
    surrogate_evals: np.ndarray,
    z_vec: np.ndarray,
    gamma: float,
    n: int,
    start_value: float = 1,
    max_iters: int = 200,
    eps: float = 1e-8,
):
    """Objective function of DEAL (reference implementation).

    Computes the optimization objective using deterministic equivalents.
    Requires solving a fixed-point equation for delta at a given gamma.

    Parameters
    ----------
    base_evals : np.ndarray
        Eigenvalue estimates for the base covariance matrix (those that will be shrunk).
    surrogate_evals : np.ndarray
        Eigenvalue estimates used to compute the shrinkage parameters.
    z_vec : np.ndarray
        Vector of interest projected into the eigenvector space.
    gamma : float
        Scalar resolvent parameter for the matrix S.
    n : int
        Effective number of observations used to compute the sample covariance matrix.
    start_value : float, optional
        Starting value for the fixed-point iteration (supports warm-starts). Default is 1.
    max_iters : int, optional
        Maximum number of fixed-point iterations. Default is 200.
    eps : float, optional
        Convergence tolerance for early stopping. Default is 1e-8.

    Returns
    -------
    obj : float
        The DEAL risk objective estimate.
    delta : float
        The converged delta value from the fixed-point iteration.
    delta_prime : float
        The derivative of delta with respect to gamma.

    See Also
    --------
    [`shrinkr.functional.deal_objective`][]
        Optimized implementation of this method.
        Go there for additional notes and references.
        Functions ref_* are reference implementations intended for validation.

    """
    # Compute delta (and beta) and diagT via fixed point iterations
    # diagT is the vector of the diagonal of the T (deterministic_equivalent) matrix
    # invDiagT is the vector of the inverse diagonal of the T matrix
    delta = start_value
    for _ in range(max_iters):
        beta = 1 / (1 + delta)
        invDiagT = gamma + surrogate_evals * beta
        new_delta = np.sum(surrogate_evals / invDiagT) / n
        if abs(new_delta - delta) < eps:
            break
        delta = new_delta
    else:
        print("WARN: Fixed point method did not coverge.")

    # Compute derivate of delta
    diagT = 1 / invDiagT
    a = np.sum((surrogate_evals) * (diagT**2))
    b = (beta**2) * np.sum((surrogate_evals * diagT) ** 2)
    delta_prime = (-a) / (n - b)

    # Compute intermediate values
    diagT_sq = diagT**2  # T^2
    z_vec_sq = z_vec**2  # z^2_i
    delta_inv_diagT = 1 - delta_prime * surrogate_evals * (
        beta**2
    )  # derivaite of the inverse of T wrt gamma

    # Compute both objective parts
    # ApN_gamma = -np.sum((z_vec_sq ) * diagT_sq * delta_inv_diagT)
    # B_gamma = np.sum(diagT * z_vec_sq / evals)
    # B_gamma_sq = B_gamma ** 2

    # Different formula for Mahalonobis based distance
    z_vec_sq_e = (z_vec_sq) * base_evals
    B_gamma = np.sum(z_vec_sq * diagT)
    B_gamma_sq = B_gamma**2
    ApN_gamma = -np.sum(z_vec_sq_e * diagT_sq * delta_inv_diagT)

    obj = B_gamma_sq / ApN_gamma

    return obj, delta, delta_prime


def golden_section_search(objective_fn, a, b, initial_delta, tol=1e-5):
    """Find the minimum of a 1D function within bounds [a, b] using golden-section search."""
    # Make the opt in log space
    a = np.log(a)
    b = np.log(b)

    def wrapped_fn(log_gamma, delta):
        gamma = np.exp(log_gamma)
        return objective_fn(gamma, delta)

    # The Golden Ratio conjugate (~0.618)
    phi = (np.sqrt(5) - 1) / 2

    # Calculate the two inner test points, c and d
    c = b - phi * (b - a)
    d = a + phi * (b - a)

    # Evaluate the first two points.
    # Notice how we pass delta sequentially so it continuously warm-starts!
    cost_c, delta, _ = wrapped_fn(c, initial_delta)
    cost_d, delta, _ = wrapped_fn(d, delta)

    # Loop until the interval [a, b] is smaller than our tolerance
    iterations = 0
    while abs(b - a) > tol:
        if cost_c < cost_d:
            # The minimum is in the left side [a, d].
            # 'd' becomes our new upper bound 'b'.
            b = d
            d = c
            cost_d = cost_c

            # Calculate a new 'c' and evaluate it
            c = b - phi * (b - a)
            cost_c, delta, _ = wrapped_fn(c, delta)
        else:
            # The minimum is in the right side [c, b].
            # 'c' becomes our new lower bound 'a'.
            a = c
            c = d
            cost_c = cost_d

            # Calculate a new 'd' and evaluate it
            d = a + phi * (b - a)
            cost_d, delta, _ = wrapped_fn(d, delta)

        iterations += 1

    # Return the midpoint of the final tiny interval, and the final delta
    optimal_log_gamma = (b + a) / 2
    cost_f, delta, ratio = wrapped_fn(optimal_log_gamma, delta)
    optimal_gamma = np.exp(optimal_log_gamma)

    return optimal_gamma, delta, ratio


EigenvalueOptions = Literal["lw_analytical", "empirical"]


def ref_deal(
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
    """DEAL (Deterministic Equivalents for Adaptive LDA) shrinkage (reference implementation).

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

    Returns
    -------
    np.ndarray
        Shrinkage-adjusted eigenvalues.

    See Also
    --------
    [`shrinkr.functional.deal`][]
        Optimized implementation of this method.
        Go there for additional notes and references.
        Functions ref_* are reference implementations intended for validation.

    """
    # Rescale eigenvalues to Trace p
    evals /= np.mean(evals)

    # Compute (non-)linear shrinkages
    orig_evals = evals.copy()
    lw_nl_evals = ref_lw_analytical(evals, n_eff, eps**2)

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

    # Optimization
    def objective_fn(gamma, start_delta):
        return ref_deal_objective(
            surrogate_evals=surrogate_evals,
            base_evals=evals,
            z_vec=z_vec,
            n=n_eff,
            gamma=gamma,
            start_value=start_delta,
            eps=eps,
        )

    optimal_gamma, _, ratio = golden_section_search(
        objective_fn, gamma_min, gamma_max, initial_delta=1, tol=eps
    )

    return evals + optimal_gamma
