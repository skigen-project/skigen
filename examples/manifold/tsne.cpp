// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// tsne.cpp — exact t-SNE on a 3-cluster Gaussian mixture.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.manifold import TSNE
//   import numpy as np
//   rng = np.random.default_rng(7)
//   X = np.vstack([rng.normal(c, 0.3, (20, 4)) for c in (-3.0, 0.0, 3.0)])
//   tsne = TSNE(n_components=2, perplexity=5.0, learning_rate=200.0,
//               n_iter=500, random_state=7)
//   Y = tsne.fit_transform(X)
//   print(Y.shape, tsne.kl_divergence_)

#include <Skigen/Manifold>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n_per = 20;
    constexpr int n = 3 * n_per;
    constexpr int d = 4;

    std::mt19937_64 rng(7);
    std::normal_distribution<double> nz(0.0, 0.3);

    Eigen::MatrixXd X(n, d);
    const double centers[3] = {-3.0, 0.0, 3.0};
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < n_per; ++i) {
            for (int j = 0; j < d; ++j) {
                X(c * n_per + i, j) = centers[c] + nz(rng);
            }
        }
    }

    Skigen::TSNE<double> tsne(/*n_components=*/2, /*perplexity=*/5.0,
                              /*learning_rate=*/200.0, /*n_iter=*/500,
                              /*random_state=*/7);
    auto Y = tsne.fit_transform(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== t-SNE (exact) ===\n";
    std::cout << "  embedding shape = " << Y.rows() << " x " << Y.cols() << "\n";
    std::cout << "  KL divergence   = " << tsne.kl_divergence() << "\n";
    return 0;
}
