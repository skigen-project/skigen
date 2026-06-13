"""Reference data for CalibratedClassifierCV (binary, sigmoid).

Parity is behavioural: held-out accuracy and Brier score within a band of
scikit-learn's CalibratedClassifierCV wrapping the same base estimator."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss

from _common import save, save_params


def main():
    rng = np.random.default_rng(60)
    n = 200
    X = rng.normal(size=(n, 4))
    y = ((X[:, 0] - X[:, 1] + 0.5 * X[:, 2]) > 0).astype(np.int64)
    k = int(n * 0.7)
    Xtr, ytr, Xte, yte = X[:k], y[:k], X[k:], y[k:]

    cc = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5)
    cc.fit(Xtr, ytr)
    proba = cc.predict_proba(Xte)[:, 1]
    save("calibrated_classifier_cv", X_train=Xtr, y_train=ytr,
         X_test=Xte, y_test=yte,
         accuracy=np.array([cc.score(Xte, yte)]),
         brier=np.array([brier_score_loss(yte, proba)]))
    save_params("calibrated_classifier_cv",
                {"method": "sigmoid", "cv": 5, "base": "GaussianNB"})


if __name__ == "__main__":
    main()
