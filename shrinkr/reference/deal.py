from typing import Literal

import numpy as np

from .lw_analytical import ref_lw_analytical


def deterministic_equivalent_objective(
    base_evals: np.ndarray,
    surrogate_evals: np.ndarray,
    z_vec: np.ndarray,
    gamma: float,
    n: int,
    start_value: float = 1,
    max_iters: int = 200,
    eps: float = 1e-8,
):
    """
    Computes the optimization objective using deterministic equivalents.
    The method requires solving a fixed point equation for delta of gamma.

    Args:
        evals - Estimator for the evals of the underlying covariance matrix
        sevals - Scaled evals as for the variance schedule
        z_vec - The projected direction for the minization objective
        gamma -  gamma (scalar value for the resolvent of the matrix S)
        n - Number of observations used to compute matrix S (effective sample size)
        start_value - Starting value for the fixed point method (can be used for worm-starts)
        max_iters -  Maximum number of iterations for the fixed point method
        mu_noise_scale - The noise scale for trace penalty based on noisy mu
        eps -  Required precision for early stopping of the fixed point method

    Returns:
        - the estimate of the risk objective
        - the delta value

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
    """
    Finds the minimum of a 1D function within bounds [a, b].
    """
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
        return deterministic_equivalent_objective(
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
