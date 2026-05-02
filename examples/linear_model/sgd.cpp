// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// sgd.cpp — SGD classifier and regressor with online learning
#include <Skigen/LinearModel>
#include <Skigen/Metrics>
#include <Skigen/ModelSelection>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate 3-class dataset
    constexpr int n_per = 60;
    constexpr int n = n_per * 3;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.7);

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

    //! [example_sgd_classifier]
    // SGD with hinge loss (SVM-like)
    Skigen::SGDClassifier<double> svm(Skigen::SGDClassifier<double>::Loss::Hinge);
    svm.fit(split.X_train, split.y_train);
    auto svm_pred = svm.predict(split.X_test);

    std::cout << "=== SGD Classifier (Hinge Loss) ===\n";
    std::cout << "Accuracy: " << Skigen::Metrics::accuracy_score(split.y_test, svm_pred) << "\n\n";

    // SGD with log loss (logistic regression-like)
    Skigen::SGDClassifier<double> log_clf(Skigen::SGDClassifier<double>::Loss::Log);
    log_clf.fit(split.X_train, split.y_train);
    auto log_pred = log_clf.predict(split.X_test);

    std::cout << "=== SGD Classifier (Log Loss) ===\n";
    std::cout << "Accuracy: " << Skigen::Metrics::accuracy_score(split.y_test, log_pred) << "\n\n";
    //! [example_sgd_classifier]

    // SGD Regressor
    Eigen::VectorXd y_reg(n);
    for (int i = 0; i < n; ++i)
        y_reg(i) = 2.0 * X(i, 0) - X(i, 1) + 1.0 + noise(rng);

    auto split_reg = Skigen::train_test_split(X, y_reg, 0.3, 42);

    //! [example_sgd_regressor]
    Skigen::SGDRegressor<double> regressor;
    regressor.fit(split_reg.X_train, split_reg.y_train);

    std::cout << "=== SGD Regressor ===\n";
    std::cout << "R²:   " << regressor.score(split_reg.X_test, split_reg.y_test) << "\n";
    std::cout << "Coef: " << regressor.coef() << "\n";
    //! [example_sgd_regressor]

    return 0;
}
