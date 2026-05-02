// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// local_outlier_factor.cpp — Outlier detection with LOF
#include <Skigen/Neighbors>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate 2D inlier cluster + a few outliers
    constexpr int n_inliers = 50;
    constexpr int n_outliers = 5;
    constexpr int n = n_inliers + n_outliers;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, 2);

    // Inliers: tight cluster around (0, 0)
    for (int i = 0; i < n_inliers; ++i) {
        X(i, 0) = noise(rng);
        X(i, 1) = noise(rng);
    }

    // Outliers: far from the cluster
    X(n_inliers + 0, 0) = 10.0;  X(n_inliers + 0, 1) =  0.0;
    X(n_inliers + 1, 0) = -8.0;  X(n_inliers + 1, 1) =  6.0;
    X(n_inliers + 2, 0) =  0.0;  X(n_inliers + 2, 1) = 12.0;
    X(n_inliers + 3, 0) =  7.0;  X(n_inliers + 3, 1) = -9.0;
    X(n_inliers + 4, 0) = -6.0;  X(n_inliers + 4, 1) = -7.0;

    std::cout << std::fixed << std::setprecision(4);

    // ---------------------------------------------------------------
    // Fit LocalOutlierFactor
    // ---------------------------------------------------------------
    //! [example_lof]
    Skigen::LocalOutlierFactor<double> lof(20);
    lof.fit(X);

    auto scores = lof.negative_outlier_factor();
    auto labels = lof.fit_predict_labels();

    std::cout << "=== Local Outlier Factor ===\n"
              << "Neighbors used: " << lof.n_neighbors_used() << "\n"
              << "Offset:         " << lof.offset() << "\n\n";
    //! [example_lof]

    // Show the outlier samples
    std::cout << "Outlier samples (label = -1):\n";
    for (Eigen::Index i = 0; i < n; ++i) {
        if (labels(i) == -1) {
            std::cout << "  Sample " << i
                      << "  (" << X(i, 0) << ", " << X(i, 1) << ")"
                      << "  LOF score: " << -scores(i) << "\n";
        }
    }

    // Summary
    int n_detected = (labels.array() == -1).count();
    std::cout << "\nDetected " << n_detected << " outliers out of "
              << n << " samples.\n";

    return 0;
}
