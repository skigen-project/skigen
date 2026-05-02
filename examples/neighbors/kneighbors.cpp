// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// kneighbors.cpp — K-Nearest Neighbors classification
#include <Skigen/Neighbors>
#include <Skigen/Metrics>
#include <Skigen/ModelSelection>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate 3-class data
    constexpr int n_per = 40;
    constexpr int n = n_per * 3;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.6);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);

    for (int i = 0; i < n_per; ++i) {
        X(i, 0) = -3.0 + noise(rng);
        X(i, 1) = -3.0 + noise(rng);
        y(i) = 0;
    }
    for (int i = 0; i < n_per; ++i) {
        X(n_per + i, 0) = 3.0 + noise(rng);
        X(n_per + i, 1) = -3.0 + noise(rng);
        y(n_per + i) = 1;
    }
    for (int i = 0; i < n_per; ++i) {
        X(2 * n_per + i, 0) = 0.0 + noise(rng);
        X(2 * n_per + i, 1) = 4.0 + noise(rng);
        y(2 * n_per + i) = 2;
    }

    auto split = Skigen::train_test_split(X, y, 0.3, 42);

    std::cout << std::fixed << std::setprecision(4);

    //! [example_kneighbors_classifier]
    // Compare different k values
    std::cout << "=== KNN: varying k ===\n";
    for (int k : {1, 3, 5, 7, 11}) {
        Skigen::KNeighborsClassifier<double> knn(k);
        knn.fit(split.X_train, split.y_train);
        auto pred = knn.predict(split.X_test);
        std::cout << "  k=" << std::setw(2) << k
                  << "  accuracy=" << Skigen::Metrics::accuracy_score(split.y_test, pred)
                  << "  F1=" << Skigen::Metrics::f1_score(split.y_test, pred) << "\n";
    }
    //! [example_kneighbors_classifier]

    //! [example_kneighbors_regressor]
    // KNN for regression
    Eigen::VectorXd y_reg(split.X_train.rows());
    for (Eigen::Index i = 0; i < y_reg.size(); ++i)
        y_reg(i) = split.X_train(i, 0) + 0.5 * split.X_train(i, 1);

    Skigen::KNeighborsRegressor<double> knn_reg(5);
    knn_reg.fit(split.X_train, y_reg);

    Eigen::VectorXd y_reg_test(split.X_test.rows());
    for (Eigen::Index i = 0; i < y_reg_test.size(); ++i)
        y_reg_test(i) = split.X_test(i, 0) + 0.5 * split.X_test(i, 1);

    std::cout << "\n=== KNeighborsRegressor (k=5) ===\n";
    std::cout << "R²: " << knn_reg.score(split.X_test, y_reg_test) << "\n";
    //! [example_kneighbors_regressor]

    return 0;
}
