"""Reference data for feature-selection univariate scores
(f_classif, f_regression, chi2) and VarianceThreshold."""

import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, VarianceThreshold)

from _common import blobs, regression_data, counts_data, save, save_params


def main():
    # VarianceThreshold variances_
    rng = np.random.default_rng(5)
    Xv = rng.normal(size=(50, 5))
    Xv[:, 2] = 3.0  # constant column -> zero variance
    vt = VarianceThreshold(threshold=0.0).fit(Xv)
    save("variance_threshold", X=Xv, variances=vt.variances_)
    save_params("variance_threshold", {"threshold": 0.0})

    # f_classif
    Xc, yc = blobs(n=60, d=4, centers=3, seed=6)
    Fc, pc = f_classif(Xc, yc)
    save("f_classif", X=Xc, y=yc, scores=Fc, pvalues=pc)
    save_params("f_classif", {"score_func": "f_classif", "k": 2})

    # f_regression
    Xr, yr = regression_data(n=60, d=4, seed=8)
    Fr, pr = f_regression(Xr, yr)
    save("f_regression", X=Xr, y=yr, scores=Fr, pvalues=pr)
    save_params("f_regression", {"score_func": "f_regression", "k": 2})

    # chi2 (non-negative features)
    Xk, yk = counts_data(n=60, d=5, seed=9)
    ch, pch = chi2(Xk, yk)
    save("chi2", X=Xk, y=yk, scores=ch, pvalues=pch)
    save_params("chi2", {"score_func": "chi2", "k": 2})


if __name__ == "__main__":
    main()
