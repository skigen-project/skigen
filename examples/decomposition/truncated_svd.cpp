// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// truncated_svd.cpp — TruncatedSVD: dimensionality reduction without centering
#include <Skigen/Decomposition>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate data — TruncatedSVD works on un-centered data (unlike PCA)
    constexpr int n = 80;
    constexpr int d = 8;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.2);

    Eigen::MatrixXd X(n, d);
    for (int i = 0; i < n; ++i) {
        double base = static_cast<double>(i) / 10.0;
        for (int j = 0; j < d; ++j) {
            X(i, j) = base * (j < 2 ? 2.0 : 0.3) + noise(rng) + 5.0;
        }
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Original data: " << n << " x " << d << "\n\n";

    // Reduce to 2 components
    Skigen::TruncatedSVD<double> svd(2);
    svd.fit(X);

    Eigen::MatrixXd X_reduced = svd.transform(X);

    std::cout << "=== TruncatedSVD (8D → 2D) ===\n";
    std::cout << "Explained variance ratio: "
              << svd.explained_variance_ratio().transpose() << "\n";
    std::cout << "Total variance captured:  "
              << svd.explained_variance_ratio().sum() * 100.0 << "%\n";
    std::cout << "Singular values:          "
              << svd.singular_values().transpose() << "\n";
    std::cout << "Reduced shape: " << X_reduced.rows() << " x "
              << X_reduced.cols() << "\n\n";

    // Compare with 5 components
    Skigen::TruncatedSVD<double> svd5(5);
    svd5.fit(X);
    std::cout << "=== 5 components ===\n";
    std::cout << "Explained variance ratio: "
              << svd5.explained_variance_ratio().transpose() << "\n";
    std::cout << "Total variance captured:  "
              << svd5.explained_variance_ratio().sum() * 100.0 << "%\n";

    return 0;
}
