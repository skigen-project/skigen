// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// quantile_regressor.cpp — predicting conditional quantiles via LP.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.linear_model import QuantileRegressor
//   import numpy as np
//   rng = np.random.default_rng(0)
//   X = np.linspace(0, 1, 200).reshape(-1, 1)
//   y = 2.0 * X.ravel() + 1.0 + rng.normal(0, 0.3, 200)
//   for q in (0.1, 0.5, 0.9):
//       m = QuantileRegressor(quantile=q, alpha=0.0).fit(X, y)
//       print(q, m.coef_, m.intercept_)

#include <Skigen/LinearModel>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    std::mt19937_64 rng(0);
    std::normal_distribution<double> noise(0.0, 0.3);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / n;
        y(i)    = 2.0 * X(i, 0) + 1.0 + noise(rng);
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== QuantileRegressor ===\n";

    //! [example_quantile_regressor]
    // Fit the 10th, 50th (median), and 90th conditional quantiles.
    for (double q : {0.1, 0.5, 0.9}) {
        Skigen::QuantileRegressor<double> model(/*quantile=*/q,
                                                /*alpha=*/0.0);
        model.fit(X, y);
        std::cout << "  q=" << q
                  << "  slope=" << model.coef()(0)
                  << "  intercept=" << model.intercept()
                  << "  n_iter=" << model.n_iter() << "\n";
    }
    //! [example_quantile_regressor]

    return 0;
}
