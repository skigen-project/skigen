// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// bayesian_ridge.cpp — Bayesian Ridge regression with predictive std.
#include <Skigen/LinearModel>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <optional>
#include <random>

int main() {
    // Tiny diabetes-style synthetic dataset.
    constexpr int n = 60;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.3);

    Eigen::MatrixXd X(n, 3);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        X(i, 1) = std::sin(static_cast<double>(i) / 7.0);
        X(i, 2) = noise(rng);  // pure noise feature
        y(i) = 2.0 * X(i, 0) + 0.5 * X(i, 1) + 1.0 + noise(rng);
    }

    std::cout << std::fixed << std::setprecision(4);

    //! [example_bayesian_ridge]
    Skigen::BayesianRidge<double> model(
        /*max_iter=*/300, /*tol=*/1e-3,
        /*alpha_1=*/1e-6, /*alpha_2=*/1e-6,
        /*lambda_1=*/1e-6, /*lambda_2=*/1e-6,
        /*alpha_init=*/std::nullopt, /*lambda_init=*/std::nullopt,
        /*compute_score=*/true);
    model.fit(X, y);

    std::cout << "=== BayesianRidge ===\n";
    std::cout << "  alpha (noise precision)  = " << model.alpha() << "\n";
    std::cout << "  lambda (weight precision)= " << model.lambda_() << "\n";
    std::cout << "  n_iter = " << model.n_iter() << "\n";
    std::cout << "  coef   = " << model.coef() << "\n";
    std::cout << "  intercept = " << model.intercept() << "\n";
    std::cout << "  R^2 (train) = " << model.score(X, y) << "\n";

    // Predict with confidence intervals on the first 5 samples.
    Eigen::MatrixXd X_test = X.topRows(5);
    auto [y_mean, y_std] = model.predict(X_test, Skigen::with_std);
    std::cout << "\nPredictions with 95% CI (first 5 rows):\n";
    for (Eigen::Index i = 0; i < y_mean.size(); ++i) {
        std::cout << "  y_pred = " << std::setw(7) << y_mean(i)
                  << "  +/- " << std::setw(6) << 1.96 * y_std(i)
                  << "  (true=" << y(i) << ")\n";
    }
    //! [example_bayesian_ridge]
    return 0;
}
