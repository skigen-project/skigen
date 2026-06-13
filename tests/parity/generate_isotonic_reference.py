"""Reference data for IsotonicRegression parity."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from _common import save, save_params


def main():
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0, 10, size=40))
    y = 0.5 * x + rng.normal(scale=1.0, size=40)  # noisy increasing trend
    iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    yhat = iso.predict(x)
    save("isotonic", x=x, y=y, yhat=yhat)
    save_params("isotonic", {"increasing": True, "out_of_bounds": "clip"})


if __name__ == "__main__":
    main()
