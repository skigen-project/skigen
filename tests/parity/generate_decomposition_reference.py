"""Reference data for FactorAnalysis parity.

FactorAnalysis loadings are identifiable only up to an orthogonal rotation,
so parity is checked on rotation-invariant quantities: the per-feature noise
variance and the model-implied covariance W Wᵀ + diag(ψ)."""

import numpy as np
from sklearn.decomposition import FactorAnalysis

from _common import save, save_params


def main():
    rng = np.random.default_rng(17)
    n, d, k = 200, 5, 2
    W_true = rng.normal(size=(d, k))
    Z = rng.normal(size=(n, k))
    noise = rng.normal(size=(n, d)) * np.array([0.3, 0.5, 0.2, 0.4, 0.6])
    X = Z @ W_true.T + noise

    fa = FactorAnalysis(n_components=k, max_iter=1000, tol=1e-3,
                        svd_method="lapack").fit(X)
    save("factor_analysis", X=X, noise_variance=fa.noise_variance_,
         implied_cov=fa.get_covariance())
    save_params("factor_analysis", {"n_components": k, "max_iter": 1000,
                                     "tol": 1e-3})


if __name__ == "__main__":
    main()
