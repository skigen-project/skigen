// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// basic_pipeline.cpp — Demonstrates StandardScaler usage
#include <Skigen/Dense>
#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X(5, 3);
    X << 1.0, -1.0,  2.0,
         2.0,  0.0,  0.0,
         0.0,  1.0, -1.0,
         1.0,  1.0,  1.0,
         3.0, -1.0,  0.0;

    std::cout << "Original data:\n" << X << "\n\n";

    // Fit and transform
    Skigen::StandardScaler scaler;
    Eigen::MatrixXd Z = scaler.fit_transform(X);

    std::cout << "Standardized:\n" << Z << "\n\n";
    std::cout << "Mean:  " << scaler.mean()  << "\n";
    std::cout << "Scale: " << scaler.scale() << "\n\n";

    // Round-trip
    Eigen::MatrixXd X_back = scaler.inverse_transform(Z);
    std::cout << "Recovered:\n" << X_back << "\n\n";

    // In-place transform
    Eigen::MatrixXd X_copy = X;
    scaler.transform_inplace(X_copy);
    std::cout << "In-place standardized:\n" << X_copy << "\n";

    return 0;
}
