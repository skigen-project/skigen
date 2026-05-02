// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// model_selection.cpp — Train/test split and cross-validation
#include <Skigen/ModelSelection>
#include <Skigen/LinearModel>
#include <Skigen/Metrics>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate regression data
    constexpr int n = 120;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, 3);
    Eigen::VectorXd y(n);

    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        X(i, 1) = static_cast<double>(i % 15) / 5.0;
        X(i, 2) = std::sin(static_cast<double>(i) / 20.0);
        y(i) = 3.0 * X(i, 0) - 2.0 * X(i, 1) + X(i, 2) + noise(rng);
    }

    std::cout << std::fixed << std::setprecision(4);

    // -----------------------------------------------------------------------
    // 1. Train/test split
    // -----------------------------------------------------------------------
    auto split = Skigen::train_test_split(X, y, /*test_size=*/0.25, /*random_state=*/42);

    std::cout << "=== Train/Test Split ===\n";
    std::cout << "Total:  " << n << " samples\n";
    std::cout << "Train:  " << split.X_train.rows() << " samples\n";
    std::cout << "Test:   " << split.X_test.rows()  << " samples\n\n";

    Skigen::LinearRegression<double> model;
    model.fit(split.X_train, split.y_train);
    std::cout << "Train R²: " << model.score(split.X_train, split.y_train) << "\n";
    std::cout << "Test  R²: " << model.score(split.X_test, split.y_test)   << "\n\n";

    // -----------------------------------------------------------------------
    // 2. Cross-validation
    // -----------------------------------------------------------------------
    auto cv5 = Skigen::cross_val_score(
        Skigen::LinearRegression<double>(), X, y, /*cv=*/5);

    std::cout << "=== 5-Fold Cross-Validation ===\n";
    std::cout << "Fold scores: " << cv5.transpose() << "\n";
    std::cout << "Mean R²:     " << cv5.mean() << "\n";
    std::cout << "Std:         " << std::sqrt((cv5.array() - cv5.mean()).square().mean()) << "\n\n";

    // 10-fold
    auto cv10 = Skigen::cross_val_score(
        Skigen::LinearRegression<double>(), X, y, /*cv=*/10);

    std::cout << "=== 10-Fold Cross-Validation ===\n";
    std::cout << "Mean R²: " << cv10.mean() << "\n";

    return 0;
}
