// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// regression_pipeline.cpp — End-to-end regression: scale → fit → predict → evaluate
#include <Skigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate synthetic regression data: y = 3*x1 - 2*x2 + 0.5*x3 + 1
    constexpr int n_samples = 100;
    constexpr int n_features = 3;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n_samples, n_features);
    Eigen::VectorXd y(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        X(i, 1) = static_cast<double>(i % 20) / 5.0;
        X(i, 2) = std::sin(static_cast<double>(i) / 15.0);
        y(i) = 3.0 * X(i, 0) - 2.0 * X(i, 1) + 0.5 * X(i, 2) + 1.0 + noise(rng);
    }

    auto split = Skigen::train_test_split(X, y, 0.2, 42);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Train: " << split.X_train.rows() << " samples\n";
    std::cout << "Test:  " << split.X_test.rows()  << " samples\n\n";

    // Pipeline: StandardScaler → LinearRegression
    auto pipe = Skigen::make_pipeline(
        Skigen::StandardScaler<double>(),
        Skigen::LinearRegression<double>());

    pipe.fit(split.X_train, split.y_train);

    Eigen::VectorXd y_pred = pipe.predict(split.X_test);
    double r2 = pipe.score(split.X_test, split.y_test);

    std::cout << "=== Regression Pipeline Results ===\n";
    std::cout << "  R²:   " << r2 << "\n";
    std::cout << "  MSE:  " << Skigen::Metrics::mean_squared_error(split.y_test, y_pred) << "\n";
    std::cout << "  RMSE: " << Skigen::Metrics::root_mean_squared_error(split.y_test, y_pred) << "\n";
    std::cout << "  MAE:  " << Skigen::Metrics::mean_absolute_error(split.y_test, y_pred) << "\n\n";

    // Compare models
    std::cout << "=== Model Comparison (R² on test set) ===\n";

    Skigen::StandardScaler<double> scaler;
    scaler.fit(split.X_train);
    auto X_train_s = scaler.transform(split.X_train);
    auto X_test_s  = scaler.transform(split.X_test);

    Skigen::LinearRegression<double> ols;
    ols.fit(X_train_s, split.y_train);
    std::cout << "  OLS:        " << ols.score(X_test_s, split.y_test) << "\n";

    Skigen::Ridge<double> ridge(1.0);
    ridge.fit(X_train_s, split.y_train);
    std::cout << "  Ridge(1.0): " << ridge.score(X_test_s, split.y_test) << "\n";

    Skigen::Lasso<double> lasso(0.01);
    lasso.fit(X_train_s, split.y_train);
    std::cout << "  Lasso(0.01):" << lasso.score(X_test_s, split.y_test) << "\n";

    Skigen::ElasticNet<double> enet(0.01, 0.5);
    enet.fit(X_train_s, split.y_train);
    std::cout << "  ElasticNet: " << enet.score(X_test_s, split.y_test) << "\n\n";

    // Cross-validation
    auto cv_scores = Skigen::cross_val_score(
        Skigen::LinearRegression<double>(), X_train_s, split.y_train, 5);
    std::cout << "=== 5-Fold Cross-Validation (OLS) ===\n";
    std::cout << "  Fold scores: " << cv_scores.transpose() << "\n";
    std::cout << "  Mean R²:     " << cv_scores.mean() << "\n";

    return 0;
}
