// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// kernel_ridge.cpp — Kernel ridge regression on a noisy sine curve.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.kernel_ridge import KernelRidge
//   import numpy as np
//   rng = np.random.default_rng(7)
//   X = np.linspace(-3, 3, 80).reshape(-1, 1)
//   y = np.sin(X).ravel() + rng.normal(scale=0.08, size=80)
//   model = KernelRidge(alpha=0.05, kernel="rbf", gamma=1.0)
//   model.fit(X, y)
//   print(model.score(X, y))

#include <Skigen/KernelRidge>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 80;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> noise(0.0, 0.08);

    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / (n - 1);
        X(i, 0) = x;
        y(i) = std::sin(x) + noise(rng);
    }

    //! [example_kernel_ridge]
    Skigen::KernelRidge<double> model(
        /*alpha=*/0.05,
        Skigen::KernelRidge<double>::Kernel::RBF,
        /*gamma=*/1.0);
    model.fit(X, y);
    Eigen::VectorXd y_pred = model.predict(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== KernelRidge (RBF) ===\n";
    std::cout << "  R^2 (training) = " << model.score(X, y) << "\n";
    std::cout << "  dual_coef size = " << model.dual_coef().size() << "\n";
    std::cout << "  first prediction = " << y_pred(0) << "\n";
    //! [example_kernel_ridge]

    return 0;
}
