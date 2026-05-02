// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// kmeans.cpp — KMeans and MiniBatchKMeans clustering
#include <Skigen/Cluster>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate 3-cluster data in 2D
    constexpr int n_per = 60;
    constexpr int n = n_per * 3;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, 2);

    // Cluster A: centered at (-4, 0)
    for (int i = 0; i < n_per; ++i) {
        X(i, 0) = -4.0 + noise(rng);
        X(i, 1) = 0.0 + noise(rng);
    }
    // Cluster B: centered at (4, 0)
    for (int i = 0; i < n_per; ++i) {
        X(n_per + i, 0) = 4.0 + noise(rng);
        X(n_per + i, 1) = 0.0 + noise(rng);
    }
    // Cluster C: centered at (0, 5)
    for (int i = 0; i < n_per; ++i) {
        X(2 * n_per + i, 0) = 0.0 + noise(rng);
        X(2 * n_per + i, 1) = 5.0 + noise(rng);
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Data: " << n << " samples, 2 features, 3 clusters\n\n";

    //! [example_kmeans]
    // KMeans
    Skigen::KMeans<double> km(3, /*max_iter=*/300, /*n_init=*/10, /*random_state=*/42);
    km.fit(X);

    std::cout << "=== KMeans (k=3) ===\n";
    std::cout << "Inertia:    " << km.inertia() << "\n";
    std::cout << "Iterations: " << km.n_iter() << "\n";
    std::cout << "Centers:\n" << km.cluster_centers() << "\n\n";

    // Predict on new points
    Eigen::MatrixXd X_new(3, 2);
    X_new << -4.0, 0.0,
              4.0, 0.0,
              0.0, 5.0;
    auto labels = km.predict(X_new);
    std::cout << "New point labels: " << labels.transpose() << "\n\n";
    //! [example_kmeans]

    //! [example_mini_batch_kmeans]
    // MiniBatchKMeans — faster for large datasets
    Skigen::MiniBatchKMeans<double> mbk(3, /*batch_size=*/30, /*max_iter=*/100, /*random_state=*/42);
    mbk.fit(X);

    std::cout << "=== MiniBatchKMeans (k=3, batch=30) ===\n";
    std::cout << "Inertia:    " << mbk.inertia() << "\n";
    std::cout << "Centers:\n" << mbk.cluster_centers() << "\n";
    //! [example_mini_batch_kmeans]

    return 0;
}
