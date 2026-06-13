"""Reference data for BayesianRidge and ARDRegression parity."""

import numpy as np
from sklearn.linear_model import BayesianRidge, ARDRegression

from _common import save, save_params


def main():
    rng = np.random.default_rng(13)
    n, d = 100, 4
    X = rng.normal(size=(n, d))
    w = np.array([1.5, -2.0, 0.0, 3.0])
    y = X @ w + 0.1 * rng.normal(size=n)

    br = BayesianRidge().fit(X, y)
    save("bayesian_ridge", X=X, y=y, coef=br.coef_,
         intercept=np.array([br.intercept_]),
         alpha=np.array([br.alpha_]), lambda_=np.array([br.lambda_]),
         pred=br.predict(X))
    save_params("bayesian_ridge", {"max_iter": 300, "tol": 1e-3})

    ard = ARDRegression().fit(X, y)
    save("ard_regression", X=X, y=y, coef=ard.coef_,
         intercept=np.array([ard.intercept_]),
         alpha=np.array([ard.alpha_]), pred=ard.predict(X))
    save_params("ard_regression", {"max_iter": 300, "threshold_lambda": 1e4})


if __name__ == "__main__":
    main()
