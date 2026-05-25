// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// isomap.cpp — Isometric Mapping (Isomap) on a small swiss-roll-like dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.manifold import Isomap
//   import numpy as np
//   t = np.linspace(0.0, 2 * np.pi, 40)
//   X = np.column_stack([np.cos(t), np.sin(t), 0.1 * t])
//   iso = Isomap(n_neighbors=5, n_components=2)
//   Y = iso.fit_transform(X)
//   print(Y.shape, iso.dist_matrix_.shape)

#include <Skigen/Manifold>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>

int main() {
    constexpr int n = 40;
    Eigen::MatrixXd X(n, 3);
    for (int i = 0; i < n; ++i) {
        const double t = 2.0 * std::numbers::pi * static_cast<double>(i) / (n - 1);
        X(i, 0) = std::cos(t);
        X(i, 1) = std::sin(t);
        X(i, 2) = 0.1 * t;
    }

    Skigen::Isomap<double> iso(/*n_components=*/2, /*n_neighbors=*/5);
    auto Y = iso.fit_transform(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Isomap ===\n";
    std::cout << "  embedding shape    = " << Y.rows() << " x " << Y.cols() << "\n";
    std::cout << "  geodesic dist[0,1] = " << iso.dist_matrix()(0, 1) << "\n";
    std::cout << "  Y[0]               = (" << Y(0, 0) << ", " << Y(0, 1) << ")\n";
    return 0;
}
