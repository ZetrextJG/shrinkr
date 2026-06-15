import numpy as np


def loss_prial(sample_cov: np.ndarray, sigma_hat: np.ndarray, sigma: np.ndarray) -> float:
    """Percentage Relative Improvement in Average Loss (PRIAL) [1].

    Parameters
    ----------
    sample_cov : np.ndarray
        Sample covariance matrix.
    sigma_hat : np.ndarray
        Estimated covariance matrix.
    sigma : np.ndarray
        True covariance matrix.

    Returns
    -------
    float
        Percentage improvement relative to the oracle, in the range [0, 1].

    References
    ----------
    [^1]: Ledoit, O., & Péché, S. (2011).
        Eigenvectors of some large sample covariance matrix ensembles.
        Probability Theory and Related Fields, 151(1), 233-264.
        <https://link.springer.com/article/10.1007/s00440-010-0298-3>
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

    Oracle estimator derived in [1].

    Parameters
    ----------
    sample_cov : np.ndarray
        Sample covariance matrix.
    sigma : np.ndarray
        True covariance matrix.

    Returns
    -------
    np.ndarray
        Oracle optimal rotation equivariant estimator under the MV loss.

    References
    ----------
    [^1]: Ledoit, O., & Wolf, M. (2020).
        Analytical nonlinear shrinkage of large-dimensional covariance matrices.
        The Annals of Statistics, 48(5), 3043-3065.
        <http://www.ledoit.net/Analytical_AoS_2020.pdf>
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


def loss_fm(v: np.ndarray, sigma: np.ndarray, mu: np.ndarray) -> float:
    r"""Fisher Margin (FM) loss.

    Defined as $FM(v) = -(v^T \mu)^2 / (v^T \Sigma v)$,
    where $\mu$ is the true difference-in-means vector,
    $\Sigma$ is the true Population Covariance matrix
    and the $v$ is the considered LDA vector.

    Minimizing the Fisher Margin leads to an optimal
    Bayesian Classifier on data which admits the LDA
    data assumptions. The loss is scale invariant.

    Parameters
    ----------
    v : np.ndarray
        LDA vector computed from data.
    sigma : np.ndarray
        True covariance matrix.
    mu : np.ndarray
        True difference-in-means vector.

    Notes
    -----
    Practically the Fisher Margin is defined
    without the minus sign. It is there only to turn
    the maximization task in a minimization one
    making it a `loss`.

    Returns
    -------
    float
        Value of the FM loss.
    """
    if len(v.shape) != 1:
        raise ValueError("v has to be a 1D vector")
    if len(mu.shape) != 1:
        raise ValueError("v has to be a 1D vector")
    if len(sigma.shape) != 2:
        raise ValueError("sigma has to be a matrix")

    A = np.dot(v, mu)
    B = v.T @ (sigma @ v)
    return -(A**2) / B


def loss_mv(sigma_hat: np.ndarray, sigma: np.ndarray) -> float:
    """Minimal Variance (MV) loss [1].

    Parameters
    ----------
    sigma_hat : np.ndarray
        Estimated covariance matrix.
    sigma : np.ndarray
        True covariance matrix.

    Returns
    -------
    float
        Value of the MV loss.

    References
    ----------
    [^1]: Ledoit, O., & Wolf, M. (2020).
        Analytical nonlinear shrinkage of large-dimensional covariance matrices.
        The Annals of Statistics, 48(5), 3043-3065.
        <http://www.ledoit.net/Analytical_AoS_2020.pdf>
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
    alpha = np.trace(np.linalg.inv(sigma)) / p
    return num / denom - alpha


def loss_fr(matrixA: np.ndarray, matrixB: np.ndarray) -> float:
    """Frobenius distance between two matrices.

    Parameters
    ----------
    matrixA : np.ndarray
        First matrix.
    matrixB : np.ndarray
        Second matrix.

    Returns
    -------
    float
        Scaled squared Frobenius distance between the matrices.
    """
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


def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy.

    Parameters
    ----------
    y : np.ndarray
        True class labels (1D integer array).
    y_pred : np.ndarray
        Predicted class labels (1D integer array).

    Returns
    -------
    float
        Fraction of correctly classified samples, in the range [0, 1].
    """
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be a 1D array")
    if y.shape != y_pred.shape:
        raise ValueError("y and y_pred must have the same shape")

    return float(np.sum(y == y_pred) / y.shape[0])
