"""Reference data for LocalOutlierFactor, KNeighborsClassifier and
KNeighborsRegressor parity."""

import numpy as np
from sklearn.neighbors import (
    LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor)

from _common import blobs, regression_data, save, save_params


def main():
    # LocalOutlierFactor — deterministic negative_outlier_factor_
    rng = np.random.default_rng(19)
    Xin = rng.normal(size=(60, 3))
    Xout = rng.normal(loc=6.0, size=(5, 3))
    X = np.vstack([Xin, Xout])
    lof = LocalOutlierFactor(n_neighbors=20).fit(X)
    save("local_outlier_factor", X=X,
         negative_outlier_factor=lof.negative_outlier_factor_)
    save_params("local_outlier_factor", {"n_neighbors": 20})

    # KNeighborsClassifier — predictions on a held-out grid
    Xc, yc = blobs(n=60, d=3, centers=3, seed=20)
    Xc_test = Xc[::3]
    knc = KNeighborsClassifier(n_neighbors=5).fit(Xc, yc)
    save("kneighbors_classifier", X=Xc, y=yc, X_test=Xc_test,
         pred=knc.predict(Xc_test))
    save_params("kneighbors_classifier", {"n_neighbors": 5})

    # KNeighborsRegressor — predictions on a held-out grid
    Xr, yr = regression_data(n=60, d=4, seed=21)
    Xr_test = Xr[::3]
    knr = KNeighborsRegressor(n_neighbors=5).fit(Xr, yr)
    save("kneighbors_regressor", X=Xr, y=yr, X_test=Xr_test,
         pred=knr.predict(Xr_test))
    save_params("kneighbors_regressor", {"n_neighbors": 5})


if __name__ == "__main__":
    main()
