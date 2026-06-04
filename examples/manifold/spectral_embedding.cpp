// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// spectral_embedding.cpp — Laplacian Eigenmaps on a tiny two-cluster set.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.manifold import SpectralEmbedding
//   import numpy as np
//   X = np.vstack([np.random.default_rng(0).normal(-2, 0.1, (15, 2)),
//                  np.random.default_rng(1).normal( 2, 0.1, (15, 2))])
//   se = SpectralEmbedding(n_components=2, n_neighbors=5, random_state=0)
//   Y = se.fit_transform(X)
//   print(Y.shape, se.affinity_matrix_.shape)

#include <Skigen/Manifold>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 30;
    Eigen::MatrixXd X(n, 2);
    std::mt19937_64 rng(0);
    std::normal_distribution<double> nz(0.0, 0.1);
    for (int i = 0; i < n; ++i) {
        const double mu = (i < n / 2) ? -2.0 : 2.0;
        X(i, 0) = mu + nz(rng);
        X(i, 1) = mu + nz(rng);
    }

    Skigen::SpectralEmbedding<double> se(/*n_components=*/2, /*n_neighbors=*/5);
    auto Y = se.fit_transform(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== SpectralEmbedding (Laplacian Eigenmaps) ===\n";
    std::cout << "  embedding shape  = " << Y.rows() << " x " << Y.cols() << "\n";
    std::cout << "  affinity shape   = "
              << se.affinity_matrix().rows() << " x "
              << se.affinity_matrix().cols() << "\n";
    return 0;
}
