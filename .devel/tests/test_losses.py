import numpy as np
import pytest

from shrinkr.functional import loss_fr, loss_prial, mv_opt_cov


@pytest.mark.unit
def test_fr_loss():
    np.random.seed(42)
    mat = np.random.rand(16)
    mat = mat.reshape(4, 4)
    # Distance between itself should be zero.
    loss = loss_fr(mat, mat)
    assert np.allclose(loss, 0.0)

    loss = loss_fr(
        np.eye(5),  # Identity matrix
        np.zeros((5, 5)),  # Zero matrix
    )
    assert np.allclose(loss, 1.0)


@pytest.mark.unit
def test_prial():
    rng = np.random.default_rng(42)
    mean = np.array([0.0, 1.0, 2.0])
    true_cov = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.3], [0.2, 0.3, 1.5]])
    X: np.ndarray = rng.multivariate_normal(mean, true_cov, size=100)
    sample_cov: np.ndarray = np.cov(X, rowvar=False)

    # No improvement over sample covariance
    loss = loss_prial(sample_cov, sample_cov, true_cov)
    assert np.allclose(loss, 0.0)

    # Max improvement over sample covariance
    # Under rotation equivariance
    sample_ast = mv_opt_cov(sample_cov, true_cov)
    loss = loss_prial(sample_cov, sample_ast, true_cov)
    assert np.allclose(loss, 1.0)


if __name__ == "__main__":
    test_fr_loss()
