// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// svr.cpp — kernel epsilon-SVR on a sinusoidal regression task.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.svm import SVR
//   import numpy as np
//   rng = np.random.default_rng(0)
//   X = np.linspace(-3, 3, 80).reshape(-1, 1)
//   y = np.sin(X.ravel()) + 0.1 * rng.standard_normal(80)
//   reg = SVR(C=1.0, kernel="rbf", gamma=0.5, epsilon=0.1)
//   reg.fit(X, y)
//   print("R^2 =", reg.score(X, y), " n_support =", len(reg.support_))

#include <Skigen/SVM>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 80;
    std::mt19937_64 rng(0);
    std::normal_distribution<double> nz(0.0, 0.1);

    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / (n - 1);
        X(i, 0) = x;
        y(i)    = std::sin(x) + nz(rng);
    }

    using K = Skigen::SVR<double>::Kernel;
    Skigen::SVR<double> reg(/*C=*/1.0, K::RBF, /*degree=*/3, /*gamma=*/0.5,
                            /*coef0=*/0.0, /*epsilon=*/0.1);
    reg.fit(X, y);

    auto yh = reg.predict(X);
    const double ss_res = (yh - y).squaredNorm();
    const double ss_tot = (y.array() - y.mean()).square().sum();
    const double r2     = 1.0 - ss_res / ss_tot;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== SVR (RBF kernel, epsilon-insensitive) ===\n";
    std::cout << "  R^2        = " << r2 << "\n";
    std::cout << "  n_support  = " << reg.n_support() << "\n";
    return 0;
}
