import numpy as np
from numpy.typing import NDArray

from shrinkr._native import py_lw_analytical


def lw_analytical(
    eigenvalues: np.ndarray, n: int, p: int | None = None, eps: float = 1e-8
) -> np.ndarray:
    """Ledoit-Wolf Analytical (nonlinear) shrinkage of eigenvalues.

    Based on a optimization free formula from Ledoit and Wolf (2020) [1].
    Handles also the high-dimensional setting where $p>n$.

    Parameters
    ----------
    eigenvalues : np.ndarray
        1-D array of eigenvalues of the sample covariance matrix.
    n : int
        Number of observations used to compute the sample covariance.
    p : int, optional
        Number of variables. If None, inferred as ``len(eigenvalues)``.
    eps : float, optional
        Threshold below which eigenvalues are treated as numerically zero.
        Default is 1e-8.

    Returns
    -------
    np.ndarray
        Analytically shrunk eigenvalues of the same shape as ``eigenvalues``.

    References
    ----------
    [^1]: Ledoit, O., & Wolf, M. (2020).
        Analytical nonlinear shrinkage of large-dimensional covariance matrices.
        The Annals of Statistics, 48(5), 3043-3065.
        <http://www.ledoit.net/Analytical_AoS_2020.pdf>
    """
    # Checks
    if len(eigenvalues.shape) != 1:
        raise ValueError("eigenvalues have to be a flat ndarray")

    # Type conversion
    eigenvalues: np.ndarray = np.ascontiguousarray(eigenvalues)
    eigenvalues: NDArray[np.float64] = eigenvalues.astype(np.float64)

    if p is None:
        p = len(eigenvalues)

    # n correction
    num_not_effective = int(np.sum(eigenvalues[np.maximum(0, p - n) :] < eps))
    if num_not_effective > 0:
        n = n - num_not_effective  # Correct the number of effective samples

    return py_lw_analytical(eigenvalues, n, p, eps)
