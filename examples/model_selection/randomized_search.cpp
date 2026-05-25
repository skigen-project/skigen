// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// randomized_search.cpp — RandomizedSearchCV samples a parameter distribution.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.linear_model import Ridge
//   from sklearn.model_selection import RandomizedSearchCV
//   import numpy as np
//   rng = np.random.default_rng(0)
//   X = rng.standard_normal((80, 3))
//   y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * rng.standard_normal(80)
//   dist = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
//   rs = RandomizedSearchCV(Ridge(), dist, n_iter=4, cv=5, random_state=42)
//   rs.fit(X, y)
//   print(rs.best_params_, rs.best_score_)

#include <Skigen/ModelSelection>
#include <Skigen/LinearModel>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 80;
    std::mt19937_64 rng(0);
    std::normal_distribution<double> nz(0.0, 0.1);

    Eigen::MatrixXd X(n, 3);
    Eigen::VectorXd y(n);
    const Eigen::Vector3d w{2.0, -1.0, 0.5};
    for (int i = 0; i < n; ++i) {
        X(i, 0) = nz(rng) * 5.0;
        X(i, 1) = nz(rng) * 5.0;
        X(i, 2) = nz(rng) * 5.0;
        y(i)    = X.row(i) * w + nz(rng);
    }

    Skigen::Ridge<double> base;
    Skigen::ParameterGrid dist(Skigen::ParameterGrid::Grid{
        {"alpha", {Skigen::ParameterValue(0.001),
                   Skigen::ParameterValue(0.01),
                   Skigen::ParameterValue(0.1),
                   Skigen::ParameterValue(1.0),
                   Skigen::ParameterValue(10.0),
                   Skigen::ParameterValue(100.0)}}});

    Skigen::RandomizedSearchCV<Skigen::Ridge<double>> rs(
        base, dist, /*n_iter=*/4, /*cv=*/5,
        /*replace=*/true, /*random_state=*/42);
    rs.fit(X, y);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== RandomizedSearchCV (Ridge / alpha) ===\n";
    std::cout << "  best alpha   = "
              << std::get<double>(rs.best_params().at("alpha")) << "\n";
    std::cout << "  best score   = " << rs.best_score() << "\n";
    std::cout << "  samples      = " << rs.cv_results_params().size() << "\n";
    return 0;
}
