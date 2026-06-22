"""Reference data for the ensemble estimators.

Tree ensembles use bootstrap sampling and feature subsampling whose RNG stream
differs from scikit-learn's, so fitted trees never match bit-for-bit. Parity
is asserted behaviourally: Skigen must reach a held-out score within a band of
sklearn's, on the same train/test split."""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor)

from _common import save, save_params


def split(X, y, frac=0.7):
    n = X.shape[0]
    k = int(n * frac)
    return X[:k], y[:k], X[k:], y[k:]


def clf_data(seed):
    rng = np.random.default_rng(seed)
    n = 120
    X = rng.normal(size=(n, 4))
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(np.int64)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def reg_data(seed):
    rng = np.random.default_rng(seed)
    n = 120
    X = rng.normal(size=(n, 4))
    y = X @ np.array([2.0, -1.0, 0.5, 1.5]) + 0.3 * rng.normal(size=n)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def clf3_data(seed):
    """Three-class fixture for multiclass boosting parity."""
    rng = np.random.default_rng(seed)
    n = 150
    X = rng.normal(size=(n, 4))
    score = X @ np.array([1.5, -1.0, 0.8, 0.5])
    y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3])).astype(np.int64)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def emit_clf3(name, model, seed):
    X, y = clf3_data(seed)
    Xtr, ytr, Xte, yte = split(X, y)
    model.fit(Xtr, ytr)
    save(name, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
         score=np.array([model.score(Xte, yte)]))


def emit_clf(name, model, seed):
    X, y = clf_data(seed)
    Xtr, ytr, Xte, yte = split(X, y)
    model.fit(Xtr, ytr)
    save(name, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
         score=np.array([model.score(Xte, yte)]))


def emit_reg(name, model, seed):
    X, y = reg_data(seed)
    Xtr, ytr, Xte, yte = split(X, y)
    model.fit(Xtr, ytr)
    save(name, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
         score=np.array([model.score(Xte, yte)]))


def main():
    emit_clf("random_forest_classifier",
             RandomForestClassifier(random_state=0), 30)
    emit_reg("random_forest_regressor",
             RandomForestRegressor(random_state=0), 31)
    emit_clf("gradient_boosting_classifier",
             GradientBoostingClassifier(random_state=0), 32)
    emit_reg("gradient_boosting_regressor",
             GradientBoostingRegressor(random_state=0), 33)
    emit_clf("hist_gradient_boosting_classifier",
             HistGradientBoostingClassifier(random_state=0), 34)
    emit_reg("hist_gradient_boosting_regressor",
             HistGradientBoostingRegressor(random_state=0), 35)

    # --- Non-default parameter branches (v1.2.0 §4.8 / §4.11) ---
    # GradientBoostingRegressor alternative losses.
    emit_reg("gradient_boosting_regressor_absolute_error",
             GradientBoostingRegressor(loss="absolute_error",
                                       random_state=0), 36)
    emit_reg("gradient_boosting_regressor_huber",
             GradientBoostingRegressor(loss="huber", random_state=0), 37)
    emit_reg("gradient_boosting_regressor_quantile",
             GradientBoostingRegressor(loss="quantile", alpha=0.5,
                                       random_state=0), 38)
    # Stochastic gradient boosting (subsample < 1).
    emit_reg("gradient_boosting_regressor_subsample",
             GradientBoostingRegressor(subsample=0.7, random_state=0), 39)
    # Multiclass boosting.
    emit_clf3("gradient_boosting_classifier_multiclass",
              GradientBoostingClassifier(random_state=0), 40)
    emit_clf3("hist_gradient_boosting_classifier_multiclass",
              HistGradientBoostingClassifier(random_state=0), 41)
    # HistGB leaf-wise growth bound and L2.
    emit_reg("hist_gradient_boosting_regressor_leafwise",
             HistGradientBoostingRegressor(max_leaf_nodes=8,
                                           l2_regularization=1.0,
                                           random_state=0), 42)

    for n in ["random_forest_classifier", "gradient_boosting_classifier",
              "hist_gradient_boosting_classifier",
              "gradient_boosting_classifier_multiclass",
              "hist_gradient_boosting_classifier_multiclass"]:
        save_params(n, {"note": "behavioural parity on accuracy"})
    for n in ["random_forest_regressor", "gradient_boosting_regressor",
              "hist_gradient_boosting_regressor",
              "gradient_boosting_regressor_absolute_error",
              "gradient_boosting_regressor_huber",
              "gradient_boosting_regressor_quantile",
              "gradient_boosting_regressor_subsample",
              "hist_gradient_boosting_regressor_leafwise"]:
        save_params(n, {"note": "behavioural parity on R^2"})


if __name__ == "__main__":
    main()
