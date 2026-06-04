// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// grid_search.cpp — GridSearchCV over Ridge's alpha hyperparameter.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.linear_model import Ridge
//   from sklearn.model_selection import GridSearchCV
//   import numpy as np
//   rng = np.random.default_rng(0)
//   X = rng.standard_normal((80, 3))
//   y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * rng.standard_normal(80)
//   grid = {"alpha": [0.01, 0.1, 1.0, 10.0]}
//   gs = GridSearchCV(Ridge(), grid, cv=5)
//   gs.fit(X, y)
//   print(gs.best_params_, gs.best_score_)

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
    Skigen::ParameterGrid grid(Skigen::ParameterGrid::Grid{
        {"alpha", {Skigen::ParameterValue(0.01),
                   Skigen::ParameterValue(0.1),
                   Skigen::ParameterValue(1.0),
                   Skigen::ParameterValue(10.0)}}});

    Skigen::GridSearchCV<Skigen::Ridge<double>> gs(base, grid, /*cv=*/5);
    gs.fit(X, y);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== GridSearchCV (Ridge / alpha) ===\n";
    std::cout << "  best alpha   = "
              << std::get<double>(gs.best_params().at("alpha")) << "\n";
    std::cout << "  best score   = " << gs.best_score() << "\n";
    std::cout << "  grid size    = " << gs.cv_results_params().size() << "\n";
    return 0;
}
