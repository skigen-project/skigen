"""Reference data for the SVM estimators.

Skigen's SMO/coordinate-descent solvers differ in iteration detail from
libsvm/liblinear, so parity is behavioural: held-out accuracy (classifiers)
or R^2 (regressors) within a band of scikit-learn's. The nu-SVM family and
OneClassSVM ship as placeholders and are not covered here."""

import numpy as np
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

from _common import save, save_params


def split(X, y, frac=0.7):
    k = int(X.shape[0] * frac)
    return X[:k], y[:k], X[k:], y[k:]


def clf_data(seed):
    rng = np.random.default_rng(seed)
    n = 120
    X = rng.normal(size=(n, 3))
    y = ((X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2]) > 0).astype(np.int64)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def reg_data(seed):
    rng = np.random.default_rng(seed)
    n = 120
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=n)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def emit(name, model, data_fn, seed):
    X, y = data_fn(seed)
    Xtr, ytr, Xte, yte = split(X, y)
    model.fit(Xtr, ytr)
    save(name, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
         score=np.array([model.score(Xte, yte)]))
    save_params(name, {"note": "default params; behavioural parity"})


def main():
    emit("svc", SVC(), clf_data, 40)
    emit("linear_svc", LinearSVC(max_iter=5000), clf_data, 41)
    emit("svr", SVR(), reg_data, 42)
    emit("linear_svr", LinearSVR(max_iter=5000), reg_data, 43)


if __name__ == "__main__":
    main()
