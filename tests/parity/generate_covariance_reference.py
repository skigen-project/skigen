"""Reference data for the Covariance estimators (EmpiricalCovariance,
LedoitWolf, OAS)."""

import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS

from _common import save, save_params


def correlated_data(n=80, d=4, seed=11):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    cov = A @ A.T + np.eye(d)
    L = np.linalg.cholesky(cov)
    X = rng.normal(size=(n, d)) @ L.T + np.array([1.0, -2.0, 0.5, 3.0][:d])
    return X.astype(np.float64)


def main():
    X = correlated_data()

    ec = EmpiricalCovariance().fit(X)
    save("empirical_covariance", X=X, covariance=ec.covariance_,
         location=ec.location_)
    save_params("empirical_covariance", {"assume_centered": False})

    lw = LedoitWolf().fit(X)
    save("ledoit_wolf", X=X, covariance=lw.covariance_,
         shrinkage=lw.shrinkage_, location=lw.location_)
    save_params("ledoit_wolf", {"assume_centered": False})

    oas = OAS().fit(X)
    save("oas", X=X, covariance=oas.covariance_, shrinkage=oas.shrinkage_,
         location=oas.location_)
    save_params("oas", {"assume_centered": False})


if __name__ == "__main__":
    main()
