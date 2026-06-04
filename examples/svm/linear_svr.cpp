// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// linear_svr.cpp — LinearSVR on a 1-D noisy linear dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.svm import LinearSVR
//   import numpy as np
//   rng = np.random.default_rng(0)
//   X = np.linspace(0, 10, 100).reshape(-1, 1)
//   y = 2.0 * X.ravel() + 1.0 + rng.normal(scale=0.5, size=100)
//   reg = LinearSVR(C=10.0, epsilon=0.1, max_iter=5000, random_state=0)
//   reg.fit(X, y)
//   print("R^2 =", reg.score(X, y))

#include <Skigen/SVM>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 100;
    std::mt19937_64 rng(0);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = 10.0 * static_cast<double>(i) / (n - 1);
        y(i)    = 2.0 * X(i, 0) + 1.0 + noise(rng);
    }

    Skigen::LinearSVR<double> reg(
        /*C=*/10.0, /*epsilon=*/0.1,
        Skigen::LinearSVR<double>::Loss::SquaredEpsilonInsensitive,
        /*tol=*/1e-6, /*max_iter=*/5000, /*fit_intercept=*/true,
        std::optional<uint64_t>(0));
    reg.fit(X, y);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== LinearSVR ===\n";
    std::cout << "  R^2 (training) = " << reg.score(X, y) << "\n";
    std::cout << "  coef           = " << reg.coef() << "\n";
    std::cout << "  intercept      = " << reg.intercept() << "\n";
    return 0;
}
