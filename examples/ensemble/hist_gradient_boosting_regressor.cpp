// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// hist_gradient_boosting_regressor.cpp — quantile-binned gradient
// boosting on a noisy non-linear 1-D dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.ensemble import HistGradientBoostingRegressor
//   import numpy as np
//   rng = np.random.default_rng(42)
//   X = np.linspace(-3, 3, 200).reshape(-1, 1)
//   y = np.sin(X).ravel() + rng.normal(scale=0.1, size=200)
//   hgb = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1,
//                                       max_bins=64, random_state=42)
//   hgb.fit(X, y)
//   print("R^2 (train) =", hgb.score(X, y))

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

    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR hgb(HGBR::Loss::SquaredError,
             /*learning_rate=*/0.1,
             /*max_iter=*/100,
             /*max_leaf_nodes=*/31,
             /*max_depth=*/std::nullopt,
             /*min_samples_leaf=*/2,
             /*l2=*/0.0,
             /*max_bins=*/64,
             std::nullopt, std::nullopt, false, 1e-7,
             std::optional<uint64_t>(42));
    hgb.fit(X, y);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== HistGradientBoostingRegressor ===\n";
    std::cout << "  n_iter            = " << hgb.n_iter() << "\n";
    std::cout << "  init_             = " << hgb.init() << "\n";
    std::cout << "  R^2 (training)    = " << hgb.score(X, y) << "\n";
    std::cout << "  first stage MSE   = " << hgb.train_score()(0) << "\n";
    std::cout << "  final stage MSE   = "
              << hgb.train_score()(hgb.train_score().size() - 1) << "\n";
    std::cout << "  bin_edges[0] size = " << hgb.bin_edges()[0].size()
              << " thresholds\n";
    return 0;
}
