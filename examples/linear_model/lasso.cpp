// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// lasso.cpp — Lasso regression: L1-regularized least squares (feature selection)
#include <Skigen/LinearModel>
#include <Skigen/Preprocessing>
#include <Skigen/ModelSelection>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate data where only 2 out of 6 features are informative
    constexpr int n = 100;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.3);

    Eigen::MatrixXd X(n, 6);
    Eigen::VectorXd y(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 6; ++j) X(i, j) = noise(rng) * 3.0;
        // Only features 0 and 2 are informative
        y(i) = 5.0 * X(i, 0) - 3.0 * X(i, 2) + 1.0 + noise(rng);
    }

    auto split = Skigen::train_test_split(X, y, 0.2, 42);

    Skigen::StandardScaler<double> scaler;
    scaler.fit(split.X_train);
    auto X_tr = scaler.transform(split.X_train);
    auto X_te = scaler.transform(split.X_test);

    std::cout << std::fixed << std::setprecision(4);

    // Lasso with varying alpha — higher alpha = more sparsity
    std::cout << "=== Lasso: feature selection via L1 ===\n";
    for (double alpha : {0.001, 0.01, 0.1, 0.5}) {
        Skigen::Lasso<double> model(alpha);
        model.fit(X_tr, split.y_train);

        int nonzero = (model.coef().array().abs() > 1e-10).count();
        std::cout << "  alpha=" << std::setw(5) << alpha
                  << "  non-zero=" << nonzero
                  << "  R²=" << model.score(X_te, split.y_test)
                  << "  coef=" << model.coef() << "\n";
    }

    return 0;
}
