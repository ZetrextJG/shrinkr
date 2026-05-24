import numpy as np

from shrinkr.functional import loss_fr

def test_fr_loss():
    np.random.seed(42)
    mat = np.random.rand(16)
    mat = mat.reshape(4, 4)
    loss = loss_fr(mat, mat)
    assert np.allclose(loss, 0.0)

    loss = loss_fr(
        np.eye(5),
        np.zeros((5, 5))
    )
    assert np.allclose(loss, 1.0)

if __name__ == "__main__":
    test_fr_loss()
