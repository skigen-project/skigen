// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// pca_clustering.cpp — PCA dimensionality reduction → KMeans clustering
#include <Skigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate high-dimensional data with 3 hidden clusters
    constexpr int n_per_cluster = 60;
    constexpr int n_samples = n_per_cluster * 3;
    constexpr int n_features = 10;  // high-dimensional

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n_samples, n_features);
    Eigen::VectorXi true_labels(n_samples);

    // Cluster 0: strong signal in first 2 components
    for (int i = 0; i < n_per_cluster; ++i) {
        for (int j = 0; j < n_features; ++j) X(i, j) = noise(rng);
        X(i, 0) += -5.0;
        X(i, 1) += -5.0;
        true_labels(i) = 0;
    }
    // Cluster 1
    for (int i = 0; i < n_per_cluster; ++i) {
        int idx = n_per_cluster + i;
        for (int j = 0; j < n_features; ++j) X(idx, j) = noise(rng);
        X(idx, 0) += 5.0;
        X(idx, 1) += -5.0;
        true_labels(idx) = 1;
    }
    // Cluster 2
    for (int i = 0; i < n_per_cluster; ++i) {
        int idx = 2 * n_per_cluster + i;
        for (int j = 0; j < n_features; ++j) X(idx, j) = noise(rng);
        X(idx, 0) += 0.0;
        X(idx, 1) += 5.0;
        true_labels(idx) = 2;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Original data: " << n_samples << " x " << n_features << "\n\n";

    // -----------------------------------------------------------------------
    // 1. Standardize
    // -----------------------------------------------------------------------
    Skigen::StandardScaler<double> scaler;
    Eigen::MatrixXd X_scaled = scaler.fit_transform(X);

    // -----------------------------------------------------------------------
    // 2. PCA — reduce to 2 components
    // -----------------------------------------------------------------------
    Skigen::PCA<double> pca(2);
    pca.fit(X_scaled);
    Eigen::MatrixXd X_pca = pca.transform(X_scaled);

    std::cout << "=== PCA (10D → 2D) ===\n";
    std::cout << "  Explained variance ratio: "
              << pca.explained_variance_ratio().transpose() << "\n";
    std::cout << "  Total variance captured:  "
              << pca.explained_variance_ratio().sum() * 100.0 << "%\n";
    std::cout << "  Singular values:          "
              << pca.singular_values().transpose() << "\n\n";

    // -----------------------------------------------------------------------
    // 3. KMeans on PCA-reduced data
    // -----------------------------------------------------------------------
    Skigen::KMeans<double> km(3, 300, 10, 42);
    km.fit(X_pca);

    std::cout << "=== KMeans (k=3) on PCA components ===\n";
    std::cout << "  Inertia: " << km.inertia() << "\n";
    std::cout << "  Iterations: " << km.n_iter() << "\n";
    std::cout << "  Cluster centers:\n" << km.cluster_centers() << "\n\n";

    // -----------------------------------------------------------------------
    // 4. Compare KMeans vs MiniBatchKMeans
    // -----------------------------------------------------------------------
    Skigen::MiniBatchKMeans<double> mbk(3, 30, 100, 42);
    mbk.fit(X_pca);

    std::cout << "=== MiniBatchKMeans (k=3, batch=30) ===\n";
    std::cout << "  Inertia: " << mbk.inertia() << "\n\n";

    // -----------------------------------------------------------------------
    // 5. Cluster purity (simple metric against true labels)
    // -----------------------------------------------------------------------
    auto km_labels = km.labels();
    int agree = 0;
    // Count how many neighbors agree (proxy for cluster quality)
    for (Eigen::Index i = 0; i < n_samples; ++i) {
        for (Eigen::Index j = i + 1; j < n_samples; ++j) {
            bool same_true = (true_labels(i) == true_labels(j));
            bool same_pred = (km_labels(i) == km_labels(j));
            if (same_true == same_pred) ++agree;
        }
    }
    int total_pairs = n_samples * (n_samples - 1) / 2;
    double rand_index = static_cast<double>(agree) / static_cast<double>(total_pairs);
    std::cout << "=== Clustering Quality ===\n";
    std::cout << "  Rand Index: " << rand_index << "\n";

    // -----------------------------------------------------------------------
    // 6. TruncatedSVD comparison (no centering)
    // -----------------------------------------------------------------------
    Skigen::TruncatedSVD<double> tsvd(2);
    tsvd.fit(X);  // raw data, no centering
    Eigen::MatrixXd X_tsvd = tsvd.transform(X);

    std::cout << "\n=== TruncatedSVD (no centering) ===\n";
    std::cout << "  Explained variance ratio: "
              << tsvd.explained_variance_ratio().transpose() << "\n";
    std::cout << "  Total variance captured:  "
              << tsvd.explained_variance_ratio().sum() * 100.0 << "%\n";

    return 0;
}
