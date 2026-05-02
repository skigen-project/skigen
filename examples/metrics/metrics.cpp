// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// metrics.cpp — Regression and classification metrics
#include <Skigen/Metrics>
#include <Skigen/LinearModel>
#include <Skigen/ModelSelection>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // -----------------------------------------------------------------------
    // Regression metrics
    // -----------------------------------------------------------------------
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    constexpr int n = 50;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXd y(n);

    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        X(i, 1) = std::sin(static_cast<double>(i) / 8.0);
        y(i) = 2.0 * X(i, 0) + X(i, 1) + noise(rng);
    }

    auto split_reg = Skigen::train_test_split(X, y, 0.2, 42);

    Skigen::LinearRegression<double> model;
    model.fit(split_reg.X_train, split_reg.y_train);
    auto y_pred = model.predict(split_reg.X_test);

    std::cout << "=== Regression Metrics ===\n";
    std::cout << "MSE:  " << Skigen::Metrics::mean_squared_error(split_reg.y_test, y_pred) << "\n";
    std::cout << "RMSE: " << Skigen::Metrics::root_mean_squared_error(split_reg.y_test, y_pred) << "\n";
    std::cout << "MAE:  " << Skigen::Metrics::mean_absolute_error(split_reg.y_test, y_pred) << "\n";
    std::cout << "R²:   " << Skigen::Metrics::r2_score(split_reg.y_test, y_pred) << "\n\n";

    // -----------------------------------------------------------------------
    // Classification metrics
    // -----------------------------------------------------------------------
    constexpr int nc = 30;
    Eigen::MatrixXd Xc(nc * 2, 2);
    Eigen::VectorXi yc(nc * 2);

    for (int i = 0; i < nc; ++i) {
        Xc(i, 0) = -2.0 + noise(rng);
        Xc(i, 1) = -2.0 + noise(rng);
        yc(i) = 0;
        Xc(nc + i, 0) = 2.0 + noise(rng);
        Xc(nc + i, 1) = 2.0 + noise(rng);
        yc(nc + i) = 1;
    }

    auto split_cls = Skigen::train_test_split(Xc, yc, 0.3, 42);

    Skigen::LogisticRegression<double> clf(1.0);
    clf.fit(split_cls.X_train, split_cls.y_train);
    auto yc_pred = clf.predict(split_cls.X_test);

    std::cout << "=== Classification Metrics ===\n";
    std::cout << "Accuracy:  " << Skigen::Metrics::accuracy_score(split_cls.y_test, yc_pred) << "\n";
    std::cout << "Precision: " << Skigen::Metrics::precision_score(split_cls.y_test, yc_pred) << "\n";
    std::cout << "Recall:    " << Skigen::Metrics::recall_score(split_cls.y_test, yc_pred) << "\n";
    std::cout << "F1:        " << Skigen::Metrics::f1_score(split_cls.y_test, yc_pred) << "\n\n";

    std::cout << "Confusion Matrix:\n"
              << Skigen::Metrics::confusion_matrix(split_cls.y_test, yc_pred) << "\n";

    return 0;
}
