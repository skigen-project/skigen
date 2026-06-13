"""Reference data for the Naive Bayes parity tests (GaussianNB,
MultinomialNB, BernoulliNB)."""

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from _common import blobs, counts_data, save, save_params


def main():
    # --- GaussianNB ---------------------------------------------------------
    Xg, yg = blobs(n=60, d=3, centers=2, seed=1)
    g = GaussianNB().fit(Xg, yg)
    save(
        "gaussian_nb",
        X=Xg,
        y=yg,
        theta=g.theta_,
        var=g.var_,
        class_prior=g.class_prior_,
        proba=g.predict_proba(Xg),
    )
    save_params("gaussian_nb", {"var_smoothing": 1e-9})

    # --- MultinomialNB ------------------------------------------------------
    Xc, yc = counts_data(n=50, d=5, seed=2)
    m = MultinomialNB().fit(Xc, yc)
    save(
        "multinomial_nb",
        X=Xc,
        y=yc,
        feature_log_prob=m.feature_log_prob_,
        class_log_prior=m.class_log_prior_,
        proba=m.predict_proba(Xc),
    )
    save_params("multinomial_nb", {"alpha": 1.0, "fit_prior": True})

    # --- BernoulliNB --------------------------------------------------------
    rng = np.random.default_rng(3)
    Xb = (rng.random((50, 6)) > 0.5).astype(np.float64)
    yb = (Xb[:, 0] + Xb[:, 1] > Xb[:, 3] + Xb[:, 4]).astype(np.int64)
    b = BernoulliNB().fit(Xb, yb)
    save(
        "bernoulli_nb",
        X=Xb,
        y=yb,
        feature_log_prob=b.feature_log_prob_,
        class_log_prior=b.class_log_prior_,
        proba=b.predict_proba(Xb),
    )
    save_params("bernoulli_nb", {"alpha": 1.0, "binarize": 0.0, "fit_prior": True})


if __name__ == "__main__":
    main()
