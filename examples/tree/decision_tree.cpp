// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// decision_tree.cpp — Decision Tree classification
#include <Skigen/Tree>
#include <Skigen/Metrics>
#include <Skigen/ModelSelection>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate a non-linear dataset (XOR-like)
    constexpr int n = 200;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.3);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);

    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i % 20) / 10.0 - 1.0 + noise(rng);
        X(i, 1) = static_cast<double>(i / 20) / 5.0 - 1.0 + noise(rng);
        // XOR-like: positive in quadrants I and III
        y(i) = ((X(i, 0) > 0) == (X(i, 1) > 0)) ? 1 : 0;
    }

    auto split = Skigen::train_test_split(X, y, 0.3, 42);

    std::cout << std::fixed << std::setprecision(4);

    // Compare different max_depth values
    std::cout << "=== Decision Tree: depth sweep ===\n";
    for (int depth : {1, 2, 3, 5, 10}) {
        Skigen::DecisionTreeClassifier<double> tree(depth);
        tree.fit(split.X_train, split.y_train);
        auto pred = tree.predict(split.X_test);
        std::cout << "  depth=" << std::setw(2) << depth
                  << "  accuracy=" << Skigen::Metrics::accuracy_score(split.y_test, pred)
                  << "  F1=" << Skigen::Metrics::f1_score(split.y_test, pred) << "\n";
    }

    // Best model
    Skigen::DecisionTreeClassifier<double> best(5);
    best.fit(split.X_train, split.y_train);
    auto best_pred = best.predict(split.X_test);

    std::cout << "\n=== Confusion Matrix (depth=5) ===\n";
    auto cm = Skigen::Metrics::confusion_matrix(split.y_test, best_pred);
    std::cout << cm << "\n";

    return 0;
}
