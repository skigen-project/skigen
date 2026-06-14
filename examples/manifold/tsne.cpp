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

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
#include <skigen/plot/figure.h>
#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
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

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
    //! [example_tsne_plot]
    Eigen::VectorXi labels(n);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < n_per; ++i) labels(c * n_per + i) = c;

    Skigen::Plot::Figure fig;
    fig.title("t-SNE embedding")
       .caption("Three 4-D Gaussian clusters embedded into 2-D by exact Skigen::TSNE")
       .xlabel("t-SNE 1")
       .ylabel("t-SNE 2")
       .scatter(Y, labels);
    return argc > 1 ? (fig.saveThemed(argv[1]) ? 0 : 1) : fig.show();
    //! [example_tsne_plot]
#else
    return 0;
#endif
}
