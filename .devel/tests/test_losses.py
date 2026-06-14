import numpy as np
import pytest

from shrinkr.functional import loss_fm, loss_fr, loss_mv, loss_prial, mv_opt_cov
from shrinkr.monte_carlo import get_large_sample_cov


@pytest.mark.unit
def test_loss_fr():
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
def test_loss_mv():
    p, n = 40, 80
    X, sc, rc = get_large_sample_cov(p, n)

    # MV loss between empirical sample cov and true cov
    loss1 = loss_mv(sc, rc)

    # MV loss between MV Optimal sample cov and true cov
    opt_sc = mv_opt_cov(sc, rc)
    loss2 = loss_mv(opt_sc, rc)

    # Optimal is better then random sample cov
    assert loss2 <= loss1


@pytest.mark.unit
def test_loss_fm():
    p, n = 40, 80
    mu = np.random.randn(p)
    X, sc, rc = get_large_sample_cov(p, n)

    # Fisher Margin loss between empirical sample cov and true cov
    v1 = np.linalg.solve(sc, mu)
    loss1 = loss_fm(v1, rc, mu)

    # Fisher Margin loss between MV Optimal sample cov and true cov
    opt_sc = mv_opt_cov(sc, rc)
    v2 = np.linalg.solve(opt_sc, mu)
    loss2 = loss_fm(v2, rc, mu)

    # Optimal is better then random sample cov
    assert loss2 <= loss1


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
    test_loss_fr()
    test_prial()
