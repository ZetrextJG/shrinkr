import numpy as np
from matplotlib import pyplot as plt

from shrinkr.functional import lw_linear
from shrinkr.monte_carlo import get_large_sample_cov

COLOR_BLUE = np.array([0.1882, 0.4118, 0.5961])
COLOR_WHITE = np.array([0.9020, 0.9608, 1.0000])


def to_image(sc_scale, path):
    img_arr = (
        COLOR_WHITE[None, None, :] * (1 - sc_scale[:, :, None])
        + COLOR_BLUE[None, None, :] * sc_scale[:, :, None]
    )

    plt.imshow(img_arr)
    plt.axis("off")
    plt.savefig(path)


def main():
    X, sc, rc = get_large_sample_cov(5, 9, seed=2)
    sc_hat, s = lw_linear(X, assume_centered=False)

    min_value, max_value = sc.min(), sc.max()
    min_value -= (max_value - min_value) * 0.2
    sc = (sc - min_value) / (max_value - min_value)
    sc_hat = (sc_hat - min_value) / (max_value - min_value)

    to_image(sc, "sc.svg")
    to_image(sc_hat, "sc_hat.svg")


if __name__ == "__main__":
    main()
