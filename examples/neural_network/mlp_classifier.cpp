// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// mlp_classifier.cpp — MLPClassifier on a binary 2-D Gaussian dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.neural_network import MLPClassifier
//   import numpy as np
//   rng = np.random.default_rng(7)
//   X = np.vstack([rng.normal(loc=-1.5, scale=0.5, size=(100, 2)),
//                  rng.normal(loc= 1.5, scale=0.5, size=(100, 2))])
//   y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
//   clf = MLPClassifier(hidden_layer_sizes=(16,), activation="relu",
//                       solver="sgd", random_state=7)
//   clf.fit(X, y)
//   print("training accuracy =", clf.score(X, y))

#include <Skigen/NeuralNetwork>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 200;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nz(0.0, 0.5);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.5 : 1.5;
        X(i, 0) = cls + nz(rng);
        X(i, 1) = cls + nz(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }

    Skigen::MLPClassifier<double> mlp(
        /*hidden_layer_sizes=*/{16},
        Skigen::MLPActivation::ReLU,
        /*alpha=*/1e-4, /*lr=*/0.05, /*max_iter=*/300,
        /*tol=*/1e-6, /*batch_size=*/0,
        std::optional<uint64_t>(7));
    mlp.fit(X, y);

    auto pred = mlp.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (pred(i) == y(i)) ++correct;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== MLPClassifier ===\n";
    std::cout << "  training accuracy = "
              << static_cast<double>(correct) / n << "\n";
    std::cout << "  n_iter            = " << mlp.n_iter_run() << "\n";
    std::cout << "  final loss        = " << mlp.loss() << "\n";
    return 0;
}
