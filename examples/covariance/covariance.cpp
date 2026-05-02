// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// covariance.cpp — Covariance estimation: EmpiricalCovariance, LedoitWolf, OAS
#include <Skigen/Covariance>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    // Generate correlated data: X = Z * L^T
    constexpr int n = 200;
    constexpr int p = 4;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 1.0);

    Eigen::MatrixXd Z(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            Z(i, j) = noise(rng);

    // Introduce correlation via a lower-triangular factor
    Eigen::MatrixXd L(p, p);
    L << 2.0, 0.0, 0.0, 0.0,
         0.5, 1.5, 0.0, 0.0,
         0.3, 0.7, 1.0, 0.0,
         0.1, 0.2, 0.4, 0.8;
    Eigen::MatrixXd X = Z * L.transpose();

    std::cout << std::fixed << std::setprecision(4);

    // ---------------------------------------------------------------
    // 1. EmpiricalCovariance
    // ---------------------------------------------------------------
    //! [example_empirical_covariance]
    Skigen::EmpiricalCovariance<double> emp;
    emp.fit(X);

    std::cout << "=== EmpiricalCovariance ===\n"
              << "Covariance (4x4):\n" << emp.covariance() << "\n\n"
              << "Location: " << emp.location() << "\n\n";
    //! [example_empirical_covariance]

    // ---------------------------------------------------------------
    // 2. LedoitWolf
    // ---------------------------------------------------------------
    //! [example_ledoit_wolf]
    Skigen::LedoitWolf<double> lw;
    lw.fit(X);

    std::cout << "=== LedoitWolf ===\n"
              << "Shrinkage coefficient: " << lw.shrinkage() << "\n"
              << "Covariance (4x4):\n" << lw.covariance() << "\n\n";
    //! [example_ledoit_wolf]

    // ---------------------------------------------------------------
    // 3. OAS
    // ---------------------------------------------------------------
    //! [example_oas]
    Skigen::OAS<double> oas;
    oas.fit(X);

    std::cout << "=== OAS ===\n"
              << "Shrinkage coefficient: " << oas.shrinkage() << "\n"
              << "Covariance (4x4):\n" << oas.covariance() << "\n\n";
    //! [example_oas]

    // Compare diagonal elements
    std::cout << "=== Diagonal comparison ===\n";
    for (int j = 0; j < p; ++j) {
        std::cout << "Feature " << j << ":  Emp=" << emp.covariance()(j, j)
                  << "  LW=" << lw.covariance()(j, j)
                  << "  OAS=" << oas.covariance()(j, j) << "\n";
    }

    return 0;
}
