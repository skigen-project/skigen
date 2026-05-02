// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// elastic_net.cpp — ElasticNet: combined L1 + L2 regularization
#include <Skigen/LinearModel>
#include <Skigen/Preprocessing>
#include <Skigen/ModelSelection>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate data with correlated features
    constexpr int n = 100;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, 5);
    Eigen::VectorXd y(n);

    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        X(i, 1) = X(i, 0) + noise(rng) * 0.2;   // correlated with x0
        X(i, 2) = std::cos(static_cast<double>(i) / 20.0);
        X(i, 3) = noise(rng);
        X(i, 4) = noise(rng);
        y(i) = 2.0 * X(i, 0) + 1.5 * X(i, 1) + X(i, 2) + noise(rng);
    }

    auto split = Skigen::train_test_split(X, y, 0.2, 42);

    Skigen::StandardScaler<double> scaler;
    scaler.fit(split.X_train);
    auto X_tr = scaler.transform(split.X_train);
    auto X_te = scaler.transform(split.X_test);

    std::cout << std::fixed << std::setprecision(4);

    //! [example_elasticnet]
    // Compare L1/L2 ratio effects
    std::cout << "=== ElasticNet: l1_ratio sweep (alpha=0.1) ===\n";
    for (double ratio : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        Skigen::ElasticNet<double> model(0.1, ratio);
        model.fit(X_tr, split.y_train);

        int nonzero = (model.coef().array().abs() > 1e-10).count();
        std::cout << "  l1_ratio=" << ratio
                  << "  non-zero=" << nonzero
                  << "  R²=" << model.score(X_te, split.y_test) << "\n";
    }
    //! [example_elasticnet]

    std::cout << "\n=== Comparison: OLS vs Ridge vs Lasso vs ElasticNet ===\n";

    Skigen::LinearRegression<double> ols;
    ols.fit(X_tr, split.y_train);
    std::cout << "  OLS:        R²=" << ols.score(X_te, split.y_test) << "\n";

    Skigen::Ridge<double> ridge(0.1);
    ridge.fit(X_tr, split.y_train);
    std::cout << "  Ridge:      R²=" << ridge.score(X_te, split.y_test) << "\n";

    Skigen::Lasso<double> lasso(0.1);
    lasso.fit(X_tr, split.y_train);
    std::cout << "  Lasso:      R²=" << lasso.score(X_te, split.y_test) << "\n";

    Skigen::ElasticNet<double> enet(0.1, 0.5);
    enet.fit(X_tr, split.y_train);
    std::cout << "  ElasticNet: R²=" << enet.score(X_te, split.y_test) << "\n";

    return 0;
}
