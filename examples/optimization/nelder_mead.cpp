// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// nelder_mead.cpp — derivative-free minimization with Nelder-Mead.

#include <Skigen/Optimization>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    //! [example_nelder_mead]
    Eigen::VectorXd x0(2);
    x0 << -3.0, 4.0;

    Skigen::NelderMead<double> optimizer(
        /*max_iter=*/500,
        /*xatol=*/1e-9,
        /*fatol=*/1e-12,
        /*initial_step=*/0.2);

    auto result = optimizer.minimize(x0, [](const Eigen::VectorXd& x) {
        return std::pow(x(0) - 1.5, 2) + std::pow(x(1) + 2.0, 2);
    });

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== NelderMead ===\n";
    std::cout << "success: " << result.success << "\n";
    std::cout << "x: " << result.x.transpose() << "\n";
    std::cout << "fun: " << result.fun << "\n";
    std::cout << "iterations: " << result.nit << "\n";
    //! [example_nelder_mead]

    return 0;
}
