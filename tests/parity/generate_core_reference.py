"""Reference data for the v1.0.0 estimators touched by the v1.1.0 §5 refactors
(LinearRegression, Ridge, Lasso, ElasticNet, KMeans) plus StandardScaler.

These have deterministic closed-form or convex solutions, so parity is tight;
KMeans uses a behavioural inertia band because k-means++ seeding differs."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cluster import KMeans

from _common import regression_data, blobs, save, save_params


def main():
    X = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0],
                  [1.0, 1.0, 1.0], [3.0, -1.0, 0.0]])
    sc = StandardScaler().fit(X)
    save("standard_scaler", X=X, mean=sc.mean_, scale=sc.scale_,
         Z=sc.transform(X))
    save_params("standard_scaler", {"with_mean": True, "with_std": True})

    Xr, yr = regression_data(n=80, d=4, seed=80)
    lr = LinearRegression().fit(Xr, yr)
    save("linear_regression", X=Xr, y=yr, coef=lr.coef_,
         intercept=np.array([lr.intercept_]), pred=lr.predict(Xr))
    save_params("linear_regression", {"fit_intercept": True})

    rd = Ridge(alpha=1.0).fit(Xr, yr)
    save("ridge", X=Xr, y=yr, coef=rd.coef_,
         intercept=np.array([rd.intercept_]), pred=rd.predict(Xr))
    save_params("ridge", {"alpha": 1.0, "fit_intercept": True})

    la = Lasso(alpha=0.1).fit(Xr, yr)
    save("lasso", X=Xr, y=yr, coef=la.coef_,
         intercept=np.array([la.intercept_]), pred=la.predict(Xr))
    save_params("lasso", {"alpha": 0.1, "fit_intercept": True})

    en = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(Xr, yr)
    save("elastic_net", X=Xr, y=yr, coef=en.coef_,
         intercept=np.array([en.intercept_]), pred=en.predict(Xr))
    save_params("elastic_net", {"alpha": 0.1, "l1_ratio": 0.5})

    Xk, _ = blobs(n=90, d=3, centers=3, seed=81)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xk)
    save("kmeans", X=Xk, inertia=np.array([km.inertia_]))
    save_params("kmeans", {"n_clusters": 3})


if __name__ == "__main__":
    main()
