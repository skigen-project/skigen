// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// umap.cpp — UMAP on a 3-cluster Gaussian mixture.
//
// Equivalent umap-learn snippet:
//
//   import umap, numpy as np
//   rng = np.random.default_rng(0)
//   X = np.vstack([rng.normal(c, 0.3, (15, 3)) for c in (-2.0, 0.0, 2.0)])
//   reducer = umap.UMAP(n_neighbors=5, n_components=2, min_dist=0.1,
//                       learning_rate=0.01, n_epochs=100, random_state=0)
//   Y = reducer.fit_transform(X)
//   print(Y.shape)

#include <Skigen/Manifold>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
#include <skigen/plot/figure.h>
#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    constexpr int n_per = 15;
    constexpr int n = 3 * n_per;
    constexpr int d = 3;

    std::mt19937_64 rng(0);
    std::normal_distribution<double> nz(0.0, 0.3);
    Eigen::MatrixXd X(n, d);
    const double centers[3] = {-2.0, 0.0, 2.0};
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < n_per; ++i) {
            for (int j = 0; j < d; ++j) {
                X(c * n_per + i, j) = centers[c] + nz(rng);
            }
        }
    }

    Skigen::UMAP<double> umap(/*n_components=*/2, /*n_neighbors=*/5,
                              /*min_dist=*/0.1, /*learning_rate=*/0.01,
                              /*n_epochs=*/100, /*negative_sample_rate=*/5,
                              /*random_state=*/0);
    auto Y = umap.fit_transform(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== UMAP ===\n";
    std::cout << "  embedding shape = " << Y.rows() << " x " << Y.cols() << "\n";
    std::cout << "  Y[0]            = (" << Y(0, 0) << ", " << Y(0, 1) << ")\n";
#ifdef SKIGEN_EXAMPLE_WITH_PLOT
    //! [example_umap_plot]
    Eigen::VectorXi labels(n);
    for (int cluster = 0; cluster < 3; ++cluster) {
        for (int sample = 0; sample < n_per; ++sample) {
            labels(cluster * n_per + sample) = cluster;
        }
    }

    Skigen::Plot::Figure fig;
    fig.title("UMAP Embedding")
       .caption("Three 3-D Gaussian clusters embedded into 2-D by Skigen::UMAP")
       .xlabel("UMAP 1")
       .ylabel("UMAP 2")
       .scatter(Y, labels);

    return argc > 1 ? (fig.saveThemed(argv[1]) ? 0 : 1) : fig.show();
    //! [example_umap_plot]
#else
    return 0;
#endif
}
