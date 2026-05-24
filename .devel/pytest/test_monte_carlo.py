import numpy as np

from shrinkr.monte_carlo import get_small_sample_cov

def test_sample_generation():
    n = 100
    X, sc, rc = get_small_sample_cov(n=n)
    assert X.shape == (100, 2)
    assert sc.shape == (2, 2)
    assert rc.shape == (2, 2)


if __name__ == "__main__":
    test_sample_generation()

