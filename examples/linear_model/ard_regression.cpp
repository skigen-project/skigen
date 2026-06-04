// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// ard_regression.cpp — Bayesian ARD regression with feature pruning.
#include <Skigen/LinearModel>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Synthetic data: only feature 0 carries signal; features 1..4 are noise.
    constexpr int n = 80;
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.2);

    Eigen::MatrixXd X(n, 5);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 5; ++j) X(i, j) = noise(rng);
        X(i, 0) = static_cast<double>(i) / 10.0;
        y(i) = 1.5 * X(i, 0) + 0.5 + noise(rng);
    }

    std::cout << std::fixed << std::setprecision(4);

    //! [example_ard_regression]
    Skigen::ARDRegression<double> ard(
        /*max_iter=*/300, /*tol=*/1e-3,
        /*alpha_1=*/1e-6, /*alpha_2=*/1e-6,
        /*lambda_1=*/1e-6, /*lambda_2=*/1e-6,
        /*compute_score=*/false,
        /*threshold_lambda=*/1e4);
    ard.fit(X, y);

    std::cout << "=== ARDRegression ===\n";
    std::cout << "  alpha    = " << ard.alpha() << "\n";
    std::cout << "  lambda_  = " << ard.lambda_() << "\n";
    std::cout << "  coef     = " << ard.coef() << "\n";
    std::cout << "  n_iter   = " << ard.n_iter() << "\n";
    std::cout << "  R^2 (train) = " << ard.score(X, y) << "\n";

    // Predictions with confidence intervals on first 5 samples.
    Eigen::MatrixXd X_test = X.topRows(5);
    auto [y_mean, y_std] = ard.predict(X_test, Skigen::with_std);
    std::cout << "\nPredictions with 95% CI (first 5 rows):\n";
    for (Eigen::Index i = 0; i < y_mean.size(); ++i) {
        std::cout << "  y_pred = " << std::setw(7) << y_mean(i)
                  << "  +/- " << std::setw(6) << 1.96 * y_std(i)
                  << "  (true=" << y(i) << ")\n";
    }
    //! [example_ard_regression]
    return 0;
}
