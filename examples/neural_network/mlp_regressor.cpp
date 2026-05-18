// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// mlp_regressor.cpp — MLPRegressor on a noisy non-linear 1-D dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.neural_network import MLPRegressor
//   import numpy as np
//   rng = np.random.default_rng(0)
//   X = np.linspace(-3, 3, 200).reshape(-1, 1)
//   y = np.sin(X).ravel() + rng.normal(scale=0.1, size=200)
//   reg = MLPRegressor(hidden_layer_sizes=(32,), activation="tanh",
//                      solver="sgd", learning_rate_init=0.05,
//                      max_iter=500, random_state=0)
//   reg.fit(X, y)
//   print("R^2 =", reg.score(X, y))

#include <Skigen/NeuralNetwork>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 200;
    std::mt19937_64 rng(0);
    std::normal_distribution<double> noise(0.0, 0.1);

    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / (n - 1);
        X(i, 0) = x;
        y(i)    = std::sin(x) + noise(rng);
    }

    Skigen::MLPRegressor<double> reg(
        /*hidden_layer_sizes=*/{32},
        Skigen::MLPActivation::Tanh,
        Skigen::MLPSolver::SGD,
        /*alpha=*/1e-4, /*lr=*/0.05, /*max_iter=*/500,
        /*tol=*/1e-7, /*batch_size=*/0,
        std::optional<uint64_t>(0));
    reg.fit(X, y);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== MLPRegressor ===\n";
    std::cout << "  R^2 (training)     = " << reg.score(X, y) << "\n";
    std::cout << "  n_iter             = " << reg.n_iter_run() << "\n";
    std::cout << "  final loss         = " << reg.loss() << "\n";
    std::cout << "  number of layers   = " << reg.coefs().size() << "\n";
    return 0;
}
