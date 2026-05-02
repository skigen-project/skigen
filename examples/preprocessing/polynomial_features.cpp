// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// polynomial_features.cpp — Generate polynomial and interaction features
#include <Skigen/Preprocessing>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>

int main() {
    // 2 features
    Eigen::MatrixXd X(3, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Original (" << X.rows() << "x" << X.cols() << "):\n"
              << X << "\n\n";

    // Degree 2 with bias: 1, x1, x2, x1², x1·x2, x2²
    Skigen::PolynomialFeatures<double> poly2(2);
    Eigen::MatrixXd X2 = poly2.fit_transform(X);

    std::cout << "Degree 2 with bias (" << X2.rows() << "x" << X2.cols()
              << "):\n" << X2 << "\n\n";

    // Degree 2 without bias: x1, x2, x1², x1·x2, x2²
    Skigen::PolynomialFeatures<double> poly2_no_bias(2, false);
    Eigen::MatrixXd X2nb = poly2_no_bias.fit_transform(X);

    std::cout << "Degree 2 no bias (" << X2nb.rows() << "x" << X2nb.cols()
              << "):\n" << X2nb << "\n\n";

    // Interaction only: 1, x1, x2, x1·x2 (no x1², x2²)
    Skigen::PolynomialFeatures<double> poly_inter(2, true, true);
    Eigen::MatrixXd Xi = poly_inter.fit_transform(X);

    std::cout << "Interaction only (" << Xi.rows() << "x" << Xi.cols()
              << "):\n" << Xi << "\n\n";

    // Degree 3 — more features
    Skigen::PolynomialFeatures<double> poly3(3, false);
    Eigen::MatrixXd X3 = poly3.fit_transform(X);
    std::cout << "Degree 3 no bias: " << X3.cols() << " output features\n";

    return 0;
}
