// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// pca_clustering.cpp — PCA dimensionality reduction → KMeans clustering
#include <Skigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
#include <skigen/plot/figure.h>
#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    // Generate high-dimensional data with 3 hidden clusters
    constexpr int n_per = 60;
    constexpr int n = n_per * 3;
    constexpr int d = 10;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, d);

    for (int i = 0; i < n_per; ++i) {
        for (int j = 0; j < d; ++j) X(i, j) = noise(rng);
        X(i, 0) += -5.0;
        X(i, 1) += -5.0;
    }
    for (int i = 0; i < n_per; ++i) {
        int idx = n_per + i;
        for (int j = 0; j < d; ++j) X(idx, j) = noise(rng);
        X(idx, 0) += 5.0;
        X(idx, 1) += -5.0;
    }
    for (int i = 0; i < n_per; ++i) {
        int idx = 2 * n_per + i;
        for (int j = 0; j < d; ++j) X(idx, j) = noise(rng);
        X(idx, 0) += 0.0;
        X(idx, 1) += 5.0;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Original data: " << n << " x " << d << "\n\n";

    // Standardize → PCA → KMeans
    Skigen::StandardScaler<double> scaler;
    Eigen::MatrixXd X_scaled = scaler.fit_transform(X);

    Skigen::PCA<double> pca(2);
    pca.fit(X_scaled);
    Eigen::MatrixXd X_pca = pca.transform(X_scaled);

    std::cout << "=== PCA (10D → 2D) ===\n";
    std::cout << "  Explained variance ratio: "
              << pca.explained_variance_ratio().transpose() << "\n";
    std::cout << "  Total captured: "
              << pca.explained_variance_ratio().sum() * 100.0 << "%\n\n";

    Skigen::KMeans<double> km(3, 300, 10, 42);
    km.fit(X_pca);

    std::cout << "=== KMeans on PCA ===\n";
    std::cout << "  Inertia:    " << km.inertia() << "\n";
    std::cout << "  Iterations: " << km.n_iter() << "\n";
    std::cout << "  Centers:\n" << km.cluster_centers() << "\n\n";

    // Compare with MiniBatchKMeans
    Skigen::MiniBatchKMeans<double> mbk(3, 30, 100, 42);
    mbk.fit(X_pca);

    std::cout << "=== MiniBatchKMeans ===\n";
    std::cout << "  Inertia: " << mbk.inertia() << "\n";

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
    //! [example_pca_clustering_plot]
    Skigen::Plot::Figure fig;
    fig.title("PCA → KMeans")
       .caption("10-D Gaussian clusters projected to 2-D by Skigen::PCA and grouped by Skigen::KMeans")
       .xlabel("PC 1")
       .ylabel("PC 2")
       .scatter(X_pca, km.predict(X_pca))
       .scatter(km.cluster_centers(), km.predict(km.cluster_centers()),
                {.pointSize = 18.0f, .hollow = true});
    return argc > 1 ? (fig.saveThemed(argv[1]) ? 0 : 1) : fig.show();
    //! [example_pca_clustering_plot]
#else
    return 0;
#endif
}
