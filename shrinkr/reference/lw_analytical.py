# The adapted code from
# https://github.com/matzhaugen/analytic_shrinkage
#

import numpy as np


def ref_lw_analytical_unstable(lam: np.ndarray, n: int, eps: float = 1e-8):
    """Shrink covarince matrix using non-linear shrinkage as described in
    Ledoit and Wolf 2018 http://www.econ.uzh.ch/static/wp/econwp264.pdf .
    The code uses an analytic formula which was previously not available
    and is thus much faster because there is no optimization necessary. The code can
    also handle the high-dimensional setting with p>n .

    Args:
        lam: empirical eigenvalues
        n: effectivate sample size
        eps: Epsilon value to use for eigenvalue filtering

    Returns:
        LW-adjusted eigenvalues

    """
    assert len(lam.shape) == 1

    lam = lam.astype(np.float64)
    p = lam.shape[0]

    # compute analytical nonlinear shrinkage kernel formula
    lam = lam[np.maximum(0, p - n) :]
    if any(lam / sum(lam) < eps):
        raise ValueError("Matrix is singular")

    L = np.tile(lam[:, None], (1, np.minimum(p, n)))
    h = np.power(n, -1 / 3.0)
    # % Equation(4.9)
    H = h * L.T
    x = (L - L.T) / H
    ftilde = (3 / 4.0 / np.sqrt(5)) * np.mean(np.maximum(1 - x**2.0 / 5.0, 0) / H, 1)
    # % Equation(4.7)
    Hftemp = (-3 / 10 / np.pi) * x + (3 / 4.0 / np.sqrt(5) / np.pi) * (1 - x**2.0 / 5.0) * np.log(
        np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x))
    )
    # % Equation(4.8)
    Hftemp[np.abs(x) == np.sqrt(5)] = (-3 / 10 / np.pi) * x[np.abs(x) == np.sqrt(5)]
    Hftilde = np.mean(Hftemp / H, 1)
    if p <= n:
        dtilde = lam / (
            (np.pi * (p / n) * lam * ftilde) ** 2
            + (1 - (p / n) - np.pi * (p / n) * lam * Hftilde) ** 2
        )
    # % Equation(4.3)
    else:
        Hftilde0 = (
            (1 / np.pi)
            * (
                3 / 10.0 / h**2
                + 3
                / 4.0
                / np.sqrt(5)
                / h
                * (1 - 1 / 5.0 / h**2)
                * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))
            )
            * np.mean(1 / lam)
        )
        # % Equation(C.8)
        dtilde0 = 1 / (np.pi * (p - n) / n * Hftilde0)
        # % Equation(C.5)
        dtilde1 = lam / (np.pi**2 * lam**2.0 * (ftilde**2 + Hftilde**2))
        # % Eq. (C.4)
        dtilde = np.concatenate([dtilde0 * np.ones(p - n), dtilde1])

    return dtilde


def ref_lw_analytical(lam: np.ndarray, n: int, eps: float = 1e-8) -> np.ndarray:
    """Shrink covarince matrix using non-linear shrinkage as described in
    Ledoit and Wolf 2018 http://www.econ.uzh.ch/static/wp/econwp264.pdf .
    The code uses an analytic formula which was previously not available
    and is thus much faster because there is no optimization necessary.
    The code can also handle the high-dimensional setting with p>n.

    Args:
        lam: empirical eigenvalues
        n: effectivate sample size
        eps: Epsilon value to use for eigenvalue filtering

    Returns:
        LW-adjusted eigenvalues
    """
    assert len(lam.shape) == 1

    lam = lam.astype(np.float64)
    p = lam.shape[0]
    sqrt5 = np.sqrt(5)

    num_not_effective = int(np.sum(lam[np.maximum(0, p - n) :] < eps))
    if num_not_effective > 0:
        n = n - num_not_effective  # Correct the number of effective samples

    lam = lam[np.maximum(0, p - n) :]  # Use only the effective eigenvalues

    L = np.tile(lam[:, None], (1, np.minimum(p, n)))
    h = np.power(n, -1 / 3.0)
    # % Equation(4.9)

    H = h * L.T
    x = (L - L.T) / H
    abs_x = np.abs(x)

    ftilde = (3 / 4.0 / sqrt5) * np.mean(np.maximum(1 - (x**2) / 5.0, 0) / H, 1)
    # % Equation(4.7)

    Hftemp = np.zeros_like(x)
    # --- Evaluate Regime 1: Singularities ---
    # The log term multiplier (1 - x^2/5) becomes exactly 0, leaving only the linear term.
    singular_mask = np.isclose(abs_x, sqrt5, atol=eps)
    Hftemp[singular_mask] = (-3 / 10 / np.pi) * x[singular_mask]
    # --- Evaluate Regime 2: Large Tails ---
    # Use the asymptotic expansion to avoid subtracting massive numbers.
    # H ~= -1/(pi*x) * [1 + 1/x^2 + 15/(7x^4) + ...]
    large_mask = abs_x > 5.0
    xl = x[large_mask]
    xl2 = xl**2
    Hftemp[large_mask] = (-1 / (np.pi * xl)) * (1 + 1 / xl2 + 15 / (7 * xl2**2))
    # --- Evaluate Regime 3: Normal Domain ---
    normal_mask = ~(singular_mask | large_mask)
    xn = x[normal_mask]
    log_term = np.log(np.abs((sqrt5 - xn) / (sqrt5 + xn)))
    linear_term = (-3 / 10 / np.pi) * xn
    Hftemp[normal_mask] = linear_term + (3 / 4 / sqrt5 / np.pi) * (1 - (xn**2) / 5.0) * log_term

    Hftilde = np.mean(Hftemp / H, 1)
    # % Equation(4.8)

    if p <= n:
        denom1 = (np.pi * (p / n) * lam * ftilde) ** 2
        denom2 = (1 - (p / n) - np.pi * (p / n) * lam * Hftilde) ** 2
        dtilde = lam / (denom1 + denom2)
        # % Equation(4.3)
    else:
        Hftilde0 = (
            (1 / np.pi)
            * (
                3 / 10.0 / h**2
                + 3
                / 4.0
                / sqrt5
                / h
                * (1 - 1 / 5.0 / h**2)
                * np.log((1 + sqrt5 * h) / (1 - sqrt5 * h))
            )
            * np.mean(1 / lam)
        )
        # % Equation(C.8)
        dtilde0 = 1 / (np.pi * (p - n) / n * Hftilde0)
        # % Equation(C.5)
        dtilde1 = lam / (np.pi**2 * lam**2.0 * (ftilde**2 + Hftilde**2))
        # % Eq. (C.4)
        dtilde = np.concatenate([dtilde0 * np.ones(p - n), dtilde1])

    return dtilde
