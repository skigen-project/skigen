"""Reference data for GridSearchCV / RandomizedSearchCV.

CV fold boundaries differ from scikit-learn, so parity is behavioural: the
selected best estimator's held-out R^2 must be within a band of sklearn's, and
the best alpha should land in the same neighbourhood of the grid."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from _common import save, save_params

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def main():
    rng = np.random.default_rng(70)
    n = 150
    X = rng.normal(size=(n, 5))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0, 1.5]) + 0.3 * rng.normal(size=n)
    k = int(n * 0.7)
    Xtr, ytr, Xte, yte = X[:k], y[:k], X[k:], y[k:]

    gs = GridSearchCV(Ridge(), {"alpha": ALPHAS}, cv=5, scoring="r2")
    gs.fit(Xtr, ytr)
    test_r2 = gs.best_estimator_.score(Xte, yte)
    save("grid_search_cv", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
         best_score=np.array([gs.best_score_]),
         test_r2=np.array([test_r2]),
         best_alpha=np.array([gs.best_params_["alpha"]]),
         alphas=np.array(ALPHAS))
    save_params("grid_search_cv", {"alphas": ALPHAS, "cv": 5, "scoring": "r2"})

    # RandomizedSearchCV shares the same selection target.
    save("randomized_search_cv", X_train=Xtr, y_train=ytr, X_test=Xte,
         y_test=yte, test_r2=np.array([test_r2]), alphas=np.array(ALPHAS))
    save_params("randomized_search_cv", {"alphas": ALPHAS, "cv": 5})


if __name__ == "__main__":
    main()
