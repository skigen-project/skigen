// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// logistic_regression.cpp — Logistic Regression for classification
#include <Skigen/LinearModel>
#include <Skigen/Metrics>
#include <Skigen/ModelSelection>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate a linearly separable binary dataset
    constexpr int n_per_class = 50;
    constexpr int n = n_per_class * 2;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.8);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);

    // Class 0: centered at (-2, -2)
    for (int i = 0; i < n_per_class; ++i) {
        X(i, 0) = -2.0 + noise(rng);
        X(i, 1) = -2.0 + noise(rng);
        y(i) = 0;
    }
    // Class 1: centered at (2, 2)
    for (int i = 0; i < n_per_class; ++i) {
        X(n_per_class + i, 0) = 2.0 + noise(rng);
        X(n_per_class + i, 1) = 2.0 + noise(rng);
        y(n_per_class + i) = 1;
    }

    auto split = Skigen::train_test_split(X, y, 0.3, 42);

    std::cout << std::fixed << std::setprecision(4);

    // Train logistic regression
    Skigen::LogisticRegression<double> model(/*C=*/1.0);
    model.fit(split.X_train, split.y_train);

    auto y_pred = model.predict(split.X_test);

    std::cout << "=== Logistic Regression ===\n";
    std::cout << "Accuracy:  " << Skigen::Metrics::accuracy_score(split.y_test, y_pred) << "\n";
    std::cout << "Precision: " << Skigen::Metrics::precision_score(split.y_test, y_pred) << "\n";
    std::cout << "Recall:    " << Skigen::Metrics::recall_score(split.y_test, y_pred) << "\n";
    std::cout << "F1 Score:  " << Skigen::Metrics::f1_score(split.y_test, y_pred) << "\n\n";

    // Confusion matrix
    auto cm = Skigen::Metrics::confusion_matrix(split.y_test, y_pred);
    std::cout << "Confusion Matrix:\n" << cm << "\n\n";

    // Compare different C values
    std::cout << "=== Regularization sweep ===\n";
    for (double C : {0.01, 0.1, 1.0, 10.0, 100.0}) {
        Skigen::LogisticRegression<double> lr(C);
        lr.fit(split.X_train, split.y_train);
        auto pred = lr.predict(split.X_test);
        std::cout << "  C=" << std::setw(6) << C
                  << "  accuracy=" << Skigen::Metrics::accuracy_score(split.y_test, pred) << "\n";
    }

    return 0;
}
