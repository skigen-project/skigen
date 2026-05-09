// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// random_forest_regressor.cpp — RandomForestRegressor on a noisy non-linear
// 1-D dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.ensemble import RandomForestRegressor
//   import numpy as np
//   rng = np.random.default_rng(7)
//   X = np.linspace(-3, 3, 200).reshape(-1, 1)
//   y = np.sin(X).ravel() + rng.normal(scale=0.1, size=200)
//   rf = RandomForestRegressor(n_estimators=50, random_state=7, oob_score=True)
//   rf.fit(X, y)
//   print("R^2 (train)  =", rf.score(X, y))
//   print("oob_score_   =", rf.oob_score_)
//   print("feat. import =", rf.feature_importances_)

#include <Skigen/Ensemble>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 200;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> noise(0.0, 0.1);

    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / (n - 1);
        X(i, 0) = x;
        y(i)    = std::sin(x) + noise(rng);
    }

    using RFR = Skigen::RandomForestRegressor<double>;
    RFR rf(/*n_estimators=*/50,
           RFR::CriterionReg::SquaredError,
           std::nullopt,                 // max_depth
           2, 1, 0.0,
           RFR::MaxFeaturesMode::OneThird,
           std::nullopt, std::nullopt, 0.0,
           /*bootstrap=*/true,
           /*oob_score=*/true,
           /*n_jobs=*/1,
           std::optional<uint64_t>(7));
    rf.fit(X, y);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== RandomForestRegressor ===\n";
    std::cout << "  n_estimators         = " << rf.estimators().size() << "\n";
    std::cout << "  R^2 (training)       = " << rf.score(X, y) << "\n";
    std::cout << "  oob_score_           = " << rf.oob_score()  << "\n";
    std::cout << "  feature_importances_ = [" << rf.feature_importances()(0) << "]\n";
    return 0;
}
