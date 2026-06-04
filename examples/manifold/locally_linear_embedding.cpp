// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// locally_linear_embedding.cpp — Standard LLE on a small circular manifold.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.manifold import LocallyLinearEmbedding
//   import numpy as np
//   t = np.linspace(0.0, 2 * np.pi, 30)
//   X = np.column_stack([np.cos(t), np.sin(t), 0.2 * t])
//   lle = LocallyLinearEmbedding(n_neighbors=5, n_components=2)
//   Y = lle.fit_transform(X)
//   print(Y.shape, lle.reconstruction_error_)

#include <Skigen/Manifold>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>

int main() {
    constexpr int n = 30;
    Eigen::MatrixXd X(n, 3);
    for (int i = 0; i < n; ++i) {
        const double t = 2.0 * std::numbers::pi * static_cast<double>(i) / (n - 1);
        X(i, 0) = std::cos(t);
        X(i, 1) = std::sin(t);
        X(i, 2) = 0.2 * t;
    }

    Skigen::LocallyLinearEmbedding<double> lle(/*n_components=*/2, /*n_neighbors=*/5);
    auto Y = lle.fit_transform(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Locally Linear Embedding (standard) ===\n";
    std::cout << "  embedding shape       = " << Y.rows() << " x " << Y.cols() << "\n";
    std::cout << "  reconstruction error  = " << lle.reconstruction_error() << "\n";
    return 0;
}
