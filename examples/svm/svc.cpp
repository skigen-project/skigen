// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// svc.cpp — kernel SVC on a 2-cluster Gaussian dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.svm import SVC
//   import numpy as np
//   rng = np.random.default_rng(11)
//   X = np.vstack([rng.normal(loc=-1, scale=0.5, size=(100, 2)),
//                  rng.normal(loc= 1, scale=0.5, size=(100, 2))])
//   y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
//   clf = SVC(C=1.0, kernel="rbf", gamma=0.5)
//   clf.fit(X, y)
//   print("training accuracy =", clf.score(X, y))
//   print("n_support_        =", clf.n_support_)

#include <Skigen/SVM>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 200;
    std::mt19937_64 rng(11);
    std::normal_distribution<double> nz(0.0, 0.5);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.0 : 1.0;
        X(i, 0) = cls + nz(rng);
        X(i, 1) = cls + nz(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }

    using K = Skigen::SVC<double>::Kernel;
    Skigen::SVC<double> clf(/*C=*/1.0, K::RBF, /*degree=*/3, /*gamma=*/0.5);
    clf.fit(X, y);

    auto pred = clf.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (pred(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / n;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== SVC (RBF kernel) ===\n";
    std::cout << "  training accuracy = " << acc << "\n";
    std::cout << "  n_support         = " << clf.n_support() << "\n";
    return 0;
}
