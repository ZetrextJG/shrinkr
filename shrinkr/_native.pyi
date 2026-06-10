import numpy as np
from numpy.typing import NDArray

def py_lw_analytical(
    lams: NDArray[np.float64], n: int, p: int, eps: float
) -> NDArray[np.float64]: ...
def py_lw_linear(X: NDArray[np.float64], n: int, p: int) -> tuple[NDArray[np.float64], float]: ...
def py_lw_linear_fast(
    X: NDArray[np.float64], sample_cov: NDArray[np.float64], n: int, p: int
) -> tuple[NDArray[np.float64], float]: ...
def py_oas(
    sample_cov: NDArray[np.float64], n: int, p: int
) -> tuple[NDArray[np.float64], float]: ...
def py_deal_objective(
    base_evals: NDArray[np.float64],
    surr_evals: NDArray[np.float64],
    z_vec: NDArray[np.float64],
    gamma: float,
    start_value: float,
    n: int,
    p: int,
) -> NDArray[np.float64]: ...
def py_deal(
    base_evals: NDArray[np.float64],
    surr_evals: NDArray[np.float64],
    z_vec: NDArray[np.float64],
    gamma_min: float,
    gamma_max: float,
    n: int,
    p: int,
) -> NDArray[np.float64]: ...
