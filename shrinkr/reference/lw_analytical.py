# The adapted code from
# https://github.com/matzhaugen/analytic_shrinkage
#

import numpy as np


def lw_analytical_shrinkage(lam: np.ndarray, n: int, eps: float = 1e-8) -> np.ndarray:
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
    p = lam.shape[0]

    # compute analytical nonlinear shrinkage kernel formula
    lam = lam[np.maximum(0, p - n):]
    if any(lam / sum(lam) < eps):
        raise ValueError("Matrix is singular")

    L = np.tile(lam[:, None], (1, np.minimum(p, n)))
    h = np.power(n, -1 / 3.)
    # % Equation(4.9)
    H = h * L.T
    x = (L - L.T) / H
    ftilde = (3 / 4. / np.sqrt(5)) * np.mean(np.maximum(
        1 - x ** 2. / 5., 0) / H, 1)
    # % Equation(4.7)
    Hftemp = (-3 / 10 / np.pi) * x + (3 / 4. / np.sqrt(5)
                                      / np.pi) * (1 - x ** 2. / 5.) * np.log(
        np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    # % Equation(4.8)
    Hftemp[np.abs(x) == np.sqrt(5)] = \
        (-3 / 10 / np.pi) * x[np.abs(x) == np.sqrt(5)]
    Hftilde = np.mean(Hftemp / H, 1)
    if p <= n:
        dtilde = lam / ((np.pi * (p / n) * lam * ftilde) ** 2
                        + (1 - (p / n) - np.pi * (p / n) * lam * Hftilde) ** 2)
    # % Equation(4.3)
    else:
        Hftilde0 = (1 / np.pi) * (3 / 10. / h ** 2 + 3 / 4. / np.sqrt(5) / h * (1 - 1 / 5. / h ** 2)
                                  * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))) * np.mean(1 / lam)
        # % Equation(C.8)
        dtilde0 = 1 / (np.pi * (p - n) / n * Hftilde0)
        # % Equation(C.5)
        dtilde1 = lam / (np.pi ** 2 * lam ** 2. * (ftilde ** 2 + Hftilde ** 2))
        # % Eq. (C.4)
        dtilde = np.concatenate([dtilde0 * np.ones((p - n)), dtilde1])
        
    return dtilde

