"""Reference data for MLPClassifier / MLPRegressor.

Mini-batch ordering and weight initialisation differ from scikit-learn, so
parity is behavioural: a fixed architecture/seed must reach a held-out score
within a band of sklearn's."""

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from _common import save, save_params

HIDDEN = (32,)
MAX_ITER = 500


def split(X, y, frac=0.7):
    k = int(X.shape[0] * frac)
    return X[:k], y[:k], X[k:], y[k:]


def main():
    rng = np.random.default_rng(50)
    n = 200
    Xc = rng.normal(size=(n, 4))
    yc = ((Xc[:, 0] + Xc[:, 1] - Xc[:, 2]) > 0).astype(np.int64)
    Xtr, ytr, Xte, yte = split(Xc, yc)
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER,
                        random_state=0).fit(Xtr, ytr)
    save("mlp_classifier", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
         score=np.array([clf.score(Xte, yte)]))
    save_params("mlp_classifier", {"hidden_layer_sizes": list(HIDDEN),
                                   "max_iter": MAX_ITER, "random_state": 0})

    rng = np.random.default_rng(51)
    Xr = rng.normal(size=(n, 4))
    yr = Xr @ np.array([1.5, -2.0, 0.5, 1.0]) + 0.1 * rng.normal(size=n)
    Xtr, ytr, Xte, yte = split(Xr, yr)
    reg = MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER,
                       random_state=0).fit(Xtr, ytr)
    save("mlp_regressor", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
         score=np.array([reg.score(Xte, yte)]))
    save_params("mlp_regressor", {"hidden_layer_sizes": list(HIDDEN),
                                  "max_iter": MAX_ITER, "random_state": 0})


if __name__ == "__main__":
    main()
