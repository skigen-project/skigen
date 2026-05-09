// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// gradient_boosting_regressor.cpp — GradientBoostingRegressor on a noisy
// non-linear 1-D dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.ensemble import GradientBoostingRegressor
//   import numpy as np
//   rng = np.random.default_rng(42)
//   X = np.linspace(-3, 3, 200).reshape(-1, 1)
//   y = np.sin(X).ravel() + rng.normal(scale=0.1, size=200)
//   gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
//                                  max_depth=3, random_state=42)
//   gb.fit(X, y)
//   print("R^2 (train) =", gb.score(X, y))
//   print("init_       =", gb.init_.constant_)
//   print("first/last train MSE =", gb.train_score_[0], gb.train_score_[-1])

#include <Skigen/Ensemble>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 200;
    std::mt19937_64 rng(42);
    std::normal_distribution<double> noise(0.0, 0.1);

    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / (n - 1);
        X(i, 0) = x;
        y(i)    = std::sin(x) + noise(rng);
    }

    using GBR = Skigen::GradientBoostingRegressor<double>;
    GBR gb(GBR::Loss::SquaredError,
           /*learning_rate=*/0.05,
           /*n_estimators=*/200,
           /*subsample=*/1.0,
           GBR::CriterionGB::FriedmanMSE,
           /*min_samples_split=*/2,
           /*min_samples_leaf=*/1,
           /*min_weight_fraction_leaf=*/0.0,
           /*max_depth=*/3,
           /*min_impurity_decrease=*/0.0,
           std::optional<uint64_t>(42));
    gb.fit(X, y);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== GradientBoostingRegressor ===\n";
    std::cout << "  n_estimators        = " << gb.n_estimators_fitted() << "\n";
    std::cout << "  init_               = " << gb.init() << "\n";
    std::cout << "  R^2 (training)      = " << gb.score(X, y) << "\n";
    std::cout << "  first stage MSE     = " << gb.train_score()(0) << "\n";
    std::cout << "  final stage MSE     = "
              << gb.train_score()(gb.train_score().size() - 1) << "\n";
    std::cout << "  feature_importances = ["
              << gb.feature_importances()(0) << "]\n";
    return 0;
}
