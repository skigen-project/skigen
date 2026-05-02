// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// pca.cpp — Principal Component Analysis for dimensionality reduction
#include <Skigen/Decomposition>
#include <Skigen/Preprocessing>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate 10-dimensional data with structure in 2 components
    constexpr int n = 100;
    constexpr int d = 10;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.3);

    Eigen::MatrixXd X(n, d);
    for (int i = 0; i < n; ++i) {
        double t1 = static_cast<double>(i) / 10.0;
        double t2 = std::sin(static_cast<double>(i) / 15.0);
        for (int j = 0; j < d; ++j) {
            X(i, j) = t1 * (j % 3 == 0 ? 1.0 : 0.1)
                     + t2 * (j % 3 == 1 ? 1.0 : 0.1)
                     + noise(rng);
        }
    }

    // Standardize before PCA
    Skigen::StandardScaler<double> scaler;
    Eigen::MatrixXd X_scaled = scaler.fit_transform(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Original data: " << n << " x " << d << "\n\n";

    // Reduce to 3 components
    Skigen::PCA<double> pca(3);
    pca.fit(X_scaled);

    Eigen::MatrixXd X_reduced = pca.transform(X_scaled);

    std::cout << "=== PCA (10D → 3D) ===\n";
    std::cout << "Explained variance ratio: "
              << pca.explained_variance_ratio().transpose() << "\n";
    std::cout << "Total variance captured:  "
              << pca.explained_variance_ratio().sum() * 100.0 << "%\n";
    std::cout << "Singular values:          "
              << pca.singular_values().transpose() << "\n";
    std::cout << "Reduced shape: " << X_reduced.rows() << " x "
              << X_reduced.cols() << "\n\n";

    // Inverse transform — reconstruct approximate original
    Eigen::MatrixXd X_approx = pca.inverse_transform(X_reduced);
    double reconstruction_error =
        (X_scaled - X_approx).squaredNorm() / static_cast<double>(n * d);
    std::cout << "Mean reconstruction error: " << reconstruction_error << "\n";

    return 0;
}
