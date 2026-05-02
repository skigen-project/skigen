// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// pipeline.cpp — Pipeline: chain transformers and estimators
#include <Skigen/Pipeline>
#include <Skigen/Preprocessing>
#include <Skigen/Decomposition>
#include <Skigen/LinearModel>
#include <Skigen/Metrics>
#include <Skigen/ModelSelection>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate regression data
    constexpr int n = 100;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, 5);
    Eigen::VectorXd y(n);

    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        X(i, 1) = static_cast<double>(i % 20) / 5.0;
        X(i, 2) = std::sin(static_cast<double>(i) / 15.0);
        X(i, 3) = noise(rng);
        X(i, 4) = noise(rng);
        y(i) = 3.0 * X(i, 0) - 2.0 * X(i, 1) + 0.5 * X(i, 2) + 1.0 + noise(rng);
    }

    auto split = Skigen::train_test_split(X, y, 0.2, 42);

    std::cout << std::fixed << std::setprecision(4);

    // -----------------------------------------------------------------------
    // Pipeline 1: StandardScaler → LinearRegression
    // -----------------------------------------------------------------------
    auto pipe1 = Skigen::make_pipeline(
        Skigen::StandardScaler<double>(),
        Skigen::LinearRegression<double>());

    pipe1.fit(split.X_train, split.y_train);

    std::cout << "=== Pipeline: Scaler → OLS ===\n";
    std::cout << "R²: " << pipe1.score(split.X_test, split.y_test) << "\n\n";

    // Access fitted scaler
    auto& scaler = pipe1.get<0>();
    std::cout << "Fitted mean:  " << scaler.mean()  << "\n";
    std::cout << "Fitted scale: " << scaler.scale() << "\n\n";

    // -----------------------------------------------------------------------
    // Pipeline 2: StandardScaler → PCA → Ridge
    // -----------------------------------------------------------------------
    auto pipe2 = Skigen::make_pipeline(
        Skigen::StandardScaler<double>(),
        Skigen::PCA<double>(3),
        Skigen::Ridge<double>(1.0));

    pipe2.fit(split.X_train, split.y_train);

    std::cout << "=== Pipeline: Scaler → PCA(3) → Ridge ===\n";
    std::cout << "R²: " << pipe2.score(split.X_test, split.y_test) << "\n\n";

    // -----------------------------------------------------------------------
    // Pipeline 3: PolynomialFeatures → StandardScaler → Lasso
    // -----------------------------------------------------------------------
    auto pipe3 = Skigen::make_pipeline(
        Skigen::PolynomialFeatures<double>(2, false),
        Skigen::StandardScaler<double>(),
        Skigen::Lasso<double>(0.01));

    pipe3.fit(split.X_train, split.y_train);

    std::cout << "=== Pipeline: Poly(2) → Scaler → Lasso ===\n";
    std::cout << "R²: " << pipe3.score(split.X_test, split.y_test) << "\n";

    return 0;
}
