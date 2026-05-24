import numpy as np

def prial(sample_cov: np.ndarray, sigma_hat: np.ndarray, sigma: np.ndarray) -> float:
    """Percentage Relative Improvement in Average Loss

    Args:
        sample_cov: Sample covariance
        sigma_hat: Estimated Covariance
        sigma: True Covariance

    Returns:
        Percentage improvement (between 0,1)
    """

    # Checks
    if len(sample_cov.shape) != 2:
        raise ValueError("Sigma hat has to be a matrix")
    if len(sigma_hat.shape) != 2:
        raise ValueError("Sigma has to be a matrix")
    if len(sigma.shape) != 2:
        raise ValueError("Sigma has to be a matrix")
    if sample_cov.shape != sigma.shape:
        raise ValueError("Sample cov has to have the same shape as Sigma")
    if sigma.shape != sigma_hat.shape:
        raise ValueError("Sigma hat has to have the same shape as Sigma")

    # Logic
    num = loss_mv(sample_cov, sigma) - loss_mv(sigma_hat, sigma)
    sigma_ast = mv_opt_cov(sample_cov, sigma)
    denom = loss_mv(sample_cov, sigma) - loss_mv(sigma_ast, sigma)
    return num / float(denom)


def mv_opt_cov(sample_cov: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Minimal variance optimal rotation equivariant estimator.

    Args:
        sample_cov: Sample covariance
        sigma: True Covariance

    Returns:
        Optimal (under MV) Rotation Equivariant Estimator
    """

    # Checks
    if len(sample_cov.shape) != 2:
        raise ValueError("Sigma hat has to be a matrix")
    if len(sigma.shape) != 2:
        raise ValueError("Sigma has to be a matrix")
    if sample_cov.shape != sigma.shape:
        raise ValueError("Sigma hat has to have the same shape as Sigma")

    # Logic
    lam, u = np.linalg.eigh(sample_cov)
    d_start: np.ndarray = np.einsum("ji, jk, ki -> i", u, sigma, u)
    ud = np.dot(u, np.diag(d_start))
    return np.dot(ud, u.T)


def loss_mv(sigma_hat: np.ndarray, sigma: np.ndarray) -> float:
    """The Minimal Variance (MV) loss 

    Args:
        sample_hat: Estimate of the true covariance
        sigma: True Covariance

    Returns:
        The value of the MV loss
    """

    # Checks
    if len(sigma_hat.shape) != 2:
        raise ValueError("sigma hat has to be a matrix")
    if len(sigma.shape) != 2:
        raise ValueError("sigma has to be a matrix")
    if sigma_hat.shape != sigma.shape:
        raise ValueError("sigma hat has to have the same shape as matrixB")

    # Logic
    n, p = sigma.shape
    omega_hat = np.linalg.inv(sigma_hat)
    num = np.trace(np.dot(np.dot(omega_hat, sigma), omega_hat)) / p
    denom = (np.trace(omega_hat) / p) ** 2
    alpha = (np.trace(np.linalg.inv(sigma)) / p)
    return num / denom - alpha


def loss_fr(matrixA: np.ndarray, matrixB: np.ndarray) -> float:
    "The Frobenius distance between matrices."

    # Checks
    if len(matrixA.shape) != 2:
        raise ValueError("matrixB hat has to be a matrix")
    if len(matrixB.shape) != 2:
        raise ValueError("matrixB has to be a matrix")
    if matrixA.shape != matrixB.shape:
        raise ValueError("matrixB hat has to have the same shape as matrixB")

    # Logic
    n, p = matrixB.shape
    delta = matrixA - matrixB
    return np.sum(delta.reshape(-1) ** 2) / p

