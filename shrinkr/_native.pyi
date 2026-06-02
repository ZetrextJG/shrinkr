import numpy as np
from numpy.typing import NDArray

def py_lw_analytical_shrinkage(
    lams: NDArray[np.float64], n: int, p: int, eps: float
) -> NDArray[np.float64]: ...
