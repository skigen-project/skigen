// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// classification_workflow.cpp — Full classification workflow comparing classifiers
#include <Skigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate synthetic 3-class data with clear cluster structure
    constexpr int n_per_class = 50;
    constexpr int n_samples = n_per_class * 3;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.8);

    Eigen::MatrixXd X(n_samples, 2);
    Eigen::VectorXi y(n_samples);

    for (int i = 0; i < n_per_class; ++i) {
        X(i, 0) = -3.0 + noise(rng);
        X(i, 1) = -3.0 + noise(rng);
        y(i) = 0;
    }
    for (int i = 0; i < n_per_class; ++i) {
        X(n_per_class + i, 0) = 3.0 + noise(rng);
        X(n_per_class + i, 1) = -3.0 + noise(rng);
        y(n_per_class + i) = 1;
    }
    for (int i = 0; i < n_per_class; ++i) {
        X(2 * n_per_class + i, 0) = 0.0 + noise(rng);
        X(2 * n_per_class + i, 1) = 4.0 + noise(rng);
        y(2 * n_per_class + i) = 2;
    }

    auto split = Skigen::train_test_split(X, y, 0.3, 42);
    std::cout << "Dataset: " << n_samples << " samples, 3 classes\n";
    std::cout << "Train: " << split.X_train.rows() << "  Test: " << split.X_test.rows() << "\n\n";
    std::cout << std::fixed << std::setprecision(4);

    // Logistic Regression
    Skigen::LogisticRegression<double> logreg(10.0, true, 500);
    logreg.fit(split.X_train, split.y_train);
    auto lr_pred = logreg.predict(split.X_test);

    std::cout << "=== Logistic Regression ===\n";
    std::cout << "  Accuracy:  " << Skigen::Metrics::accuracy_score(split.y_test, lr_pred) << "\n";
    std::cout << "  F1:        " << Skigen::Metrics::f1_score(split.y_test, lr_pred) << "\n\n";

    // KNN
    Skigen::KNeighborsClassifier<double> knn(5);
    knn.fit(split.X_train, split.y_train);
    auto knn_pred = knn.predict(split.X_test);

    std::cout << "=== KNN (k=5) ===\n";
    std::cout << "  Accuracy:  " << Skigen::Metrics::accuracy_score(split.y_test, knn_pred) << "\n";
    std::cout << "  F1:        " << Skigen::Metrics::f1_score(split.y_test, knn_pred) << "\n\n";

    // Decision Tree
    Skigen::DecisionTreeClassifier<double> dt(5);
    dt.fit(split.X_train, split.y_train);
    auto dt_pred = dt.predict(split.X_test);

    std::cout << "=== Decision Tree (depth=5) ===\n";
    std::cout << "  Accuracy:  " << Skigen::Metrics::accuracy_score(split.y_test, dt_pred) << "\n";
    std::cout << "  F1:        " << Skigen::Metrics::f1_score(split.y_test, dt_pred) << "\n\n";

    // SGD
    Skigen::SGDClassifier<double> sgd(Skigen::SGDClassifier<double>::Loss::Log);
    sgd.fit(split.X_train, split.y_train);
    auto sgd_pred = sgd.predict(split.X_test);

    std::cout << "=== SGD (Log Loss) ===\n";
    std::cout << "  Accuracy:  " << Skigen::Metrics::accuracy_score(split.y_test, sgd_pred) << "\n";
    std::cout << "  F1:        " << Skigen::Metrics::f1_score(split.y_test, sgd_pred) << "\n\n";

    // Confusion matrix for best
    std::cout << "=== Confusion Matrix (Logistic Regression) ===\n";
    std::cout << Skigen::Metrics::confusion_matrix(split.y_test, lr_pred) << "\n";

    return 0;
}
