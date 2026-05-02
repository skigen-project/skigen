// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// factor_analysis.cpp — Factor Analysis for latent variable estimation
#include <Skigen/Decomposition>
#include <Skigen/Preprocessing>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate data with known factor structure: X = Z * W^T + noise
    constexpr int n = 500;
    constexpr int p = 6;
    constexpr int k = 2;

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Latent factors
    Eigen::MatrixXd Z(n, k);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j)
            Z(i, j) = dist(rng);

    // True loading matrix (p × k)
    Eigen::MatrixXd W_true(p, k);
    W_true << 2.0, 0.0,
              1.5, 0.5,
              0.0, 2.0,
              0.3, 1.5,
              1.0, 0.0,
              0.0, 1.0;

    // Observations = Z * W^T + noise
    Eigen::MatrixXd noise(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            noise(i, j) = dist(rng) * 0.3;

    Eigen::MatrixXd X = Z * W_true.transpose() + noise;

    std::cout << std::fixed << std::setprecision(4);

    // ---------------------------------------------------------------
    // Fit FactorAnalysis
    // ---------------------------------------------------------------
    //! [example_factor_analysis]
    Skigen::FactorAnalysis<double> fa(k);
    fa.fit(X);

    std::cout << "=== Factor Analysis ===\n"
              << "Components (loading matrix, " << fa.components().rows()
              << " x " << fa.components().cols() << "):\n"
              << fa.components() << "\n\n"
              << "Noise variances: "
              << fa.noise_variance().transpose() << "\n"
              << "Log-likelihood: " << fa.log_likelihood() << "\n"
              << "EM iterations:  " << fa.n_iter() << "\n\n";
    //! [example_factor_analysis]

    // Verify covariance recovery
    auto cov_est = fa.covariance();
    Eigen::MatrixXd cov_true = W_true * W_true.transpose();
    for (int j = 0; j < p; ++j)
        cov_true(j, j) += 0.3 * 0.3;

    std::cout << "Estimated covariance diagonal: ";
    for (int j = 0; j < p; ++j)
        std::cout << cov_est(j, j) << " ";
    std::cout << "\nTrue covariance diagonal:      ";
    for (int j = 0; j < p; ++j)
        std::cout << cov_true(j, j) << " ";
    std::cout << "\n\n";

    // Score the data
    double ll = fa.score(X);
    std::cout << "Score (avg. log-likelihood per sample): " << ll << "\n";

    return 0;
}
