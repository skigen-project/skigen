// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// ridge.cpp — Ridge regression: L2-regularized least squares
#include <Skigen/LinearModel>
#include <Skigen/Preprocessing>
#include <Skigen/ModelSelection>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate collinear data where regularization matters
    constexpr int n = 100;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, 4);
    Eigen::VectorXd y(n);

    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        X(i, 1) = X(i, 0) + noise(rng) * 0.1;   // near-collinear with x0
        X(i, 2) = std::sin(static_cast<double>(i) / 15.0);
        X(i, 3) = noise(rng);                     // pure noise feature
        y(i) = 3.0 * X(i, 0) + 0.5 * X(i, 2) + 1.0 + noise(rng);
    }

    auto split = Skigen::train_test_split(X, y, 0.2, 42);

    Skigen::StandardScaler<double> scaler;
    scaler.fit(split.X_train);
    auto X_tr = scaler.transform(split.X_train);
    auto X_te = scaler.transform(split.X_test);

    std::cout << std::fixed << std::setprecision(4);

    // Compare different alpha values
    std::cout << "=== Ridge: alpha sweep ===\n";
    for (double alpha : {0.01, 0.1, 1.0, 10.0, 100.0}) {
        Skigen::Ridge<double> model(alpha);
        model.fit(X_tr, split.y_train);
        std::cout << "  alpha=" << std::setw(6) << alpha
                  << "  R²=" << model.score(X_te, split.y_test)
                  << "  coef=" << model.coef() << "\n";
    }

    std::cout << "\n=== OLS comparison ===\n";
    Skigen::LinearRegression<double> ols;
    ols.fit(X_tr, split.y_train);
    std::cout << "  OLS R²=" << ols.score(X_te, split.y_test)
              << "  coef=" << ols.coef() << "\n";

    return 0;
}
