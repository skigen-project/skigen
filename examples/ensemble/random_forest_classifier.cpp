// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// random_forest_classifier.cpp — RandomForestClassifier on a small XOR dataset.
#include <Skigen/Ensemble>
#include <Skigen/Metrics>
#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 200;
    std::mt19937_64 rng(42);
    std::normal_distribution<double> noise(0.0, 0.3);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i % 20) / 10.0 - 1.0 + noise(rng);
        X(i, 1) = static_cast<double>(i / 20) / 5.0 - 1.0 + noise(rng);
        y(i) = ((X(i, 0) > 0) == (X(i, 1) > 0)) ? 1 : 0;
    }

    using RFC = Skigen::RandomForestClassifier<double>;
    RFC rf(50, RFC::CriterionClf::Gini, std::nullopt, 2, 1, 0.0,
           RFC::MaxFeaturesMode::Sqrt, std::nullopt, std::nullopt, 0.0,
           true, /*oob_score=*/true, 1, std::optional<uint64_t>(7));
    rf.fit(X, y);

    auto pred = rf.predict(X);
    double acc = Skigen::Metrics::accuracy_score(y, pred);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== RandomForestClassifier ===\n";
    std::cout << "  n_estimators       = " << rf.estimators().size() << "\n";
    std::cout << "  training accuracy  = " << acc << "\n";
    std::cout << "  oob_score_         = " << rf.oob_score() << "\n";
    std::cout << "  feature_importances_ = ["
              << rf.feature_importances()(0) << ", "
              << rf.feature_importances()(1) << "]\n";
    return 0;
}
