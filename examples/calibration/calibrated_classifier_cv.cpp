// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// calibrated_classifier_cv.cpp — wrap a GaussianNB binary classifier with
// CalibratedClassifierCV (sigmoid Platt scaling).
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.calibration import CalibratedClassifierCV
//   from sklearn.naive_bayes import GaussianNB
//   import numpy as np
//   rng = np.random.default_rng(42)
//   X = np.vstack([rng.normal(loc=-1.5, size=(100, 2)),
//                  rng.normal(loc= 1.5, size=(100, 2))])
//   y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
//   cc = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5)
//   cc.fit(X, y)
//   print("training accuracy =", cc.score(X, y))
//   print("first row probas  =", cc.predict_proba(X[:1]))

#include <Skigen/Calibration>
#include <Skigen/NaiveBayes>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 200;
    std::mt19937_64 rng(42);
    std::normal_distribution<double> nz(0.0, 1.0);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.5 : 1.5;
        X(i, 0) = cls + nz(rng);
        X(i, 1) = cls + nz(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }

    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Sigmoid, /*cv=*/5,
        /*n_jobs=*/1, /*ensemble=*/true,
        std::optional<uint64_t>(42));
    cc.fit(X, y);

    auto pred = cc.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (pred(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / y.size();

    Eigen::MatrixXd P_first = cc.predict_proba(X.topRows(3));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== CalibratedClassifierCV (sigmoid, 5-fold) ===\n";
    std::cout << "  base                = GaussianNB\n";
    std::cout << "  n_estimators_fitted = " << cc.n_estimators_fitted() << "\n";
    std::cout << "  n_classes           = " << cc.n_classes() << "\n";
    std::cout << "  classes             = ["
              << cc.classes()(0) << ", " << cc.classes()(1) << "]\n";
    std::cout << "  training accuracy   = " << acc << "\n";
    std::cout << "  first 3 rows predict_proba:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "    row " << i << ": ["
                  << P_first(i, 0) << ", " << P_first(i, 1) << "]\n";
    }
    return 0;
}
