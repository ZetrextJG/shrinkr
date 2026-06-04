import numpy as np
from numpy.typing import NDArray

def py_lw_analytical(
    lams: NDArray[np.float64], n: int, p: int, eps: float
) -> NDArray[np.float64]: ...
def py_lw_linear(
    X: NDArray[np.float64], n: int, p: int, block_size: int
) -> NDArray[np.float64]: ...
def py_oas(sample_cov: NDArray[np.float64], n: int, p: int) -> NDArray[np.float64]: ...
