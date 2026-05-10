// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// linear_svc.cpp — LinearSVC on a binary 2-D classification problem.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.svm import LinearSVC
//   import numpy as np
//   rng = np.random.default_rng(7)
//   X = np.vstack([rng.normal(loc=-1.5, scale=0.5, size=(60, 2)),
//                  rng.normal(loc= 1.5, scale=0.5, size=(60, 2))])
//   y = np.concatenate([np.zeros(60), np.ones(60)]).astype(int)
//   clf = LinearSVC(C=1.0).fit(X, y)
//   print("training accuracy =", clf.score(X, y))

#include <Skigen/SVM>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 120;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nz(0.0, 0.5);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.5 : 1.5;
        X(i, 0) = cls + nz(rng);
        X(i, 1) = cls + nz(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }

    Skigen::LinearSVC<double> clf(
        /*C=*/1.0,
        Skigen::LinearSVC<double>::Loss::SquaredHinge,
        /*tol=*/1e-4, /*max_iter=*/200, /*fit_intercept=*/true,
        std::optional<uint64_t>(7));
    clf.fit(X, y);

    auto pred = clf.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (pred(i) == y(i)) ++correct;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== LinearSVC ===\n";
    std::cout << "  training accuracy = "
              << static_cast<double>(correct) / n << "\n";
    std::cout << "  coef shape        = " << clf.coef().rows() << " x "
              << clf.coef().cols() << "\n";
    return 0;
}
