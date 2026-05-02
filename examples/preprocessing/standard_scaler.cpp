// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// standard_scaler.cpp — StandardScaler: zero-mean, unit-variance scaling
#include <Skigen/Preprocessing>
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

    //! [example_standard_scaler]
    // Fit and transform
    Skigen::StandardScaler<double> scaler;
    Eigen::MatrixXd Z = scaler.fit_transform(X);

    std::cout << "Standardized:\n" << Z << "\n\n";
    std::cout << "Mean:  " << scaler.mean()  << "\n";
    std::cout << "Scale: " << scaler.scale() << "\n\n";

    // Round-trip: inverse_transform recovers the original data
    Eigen::MatrixXd X_back = scaler.inverse_transform(Z);
    std::cout << "Recovered:\n" << X_back << "\n\n";
    //! [example_standard_scaler]

    // In-place transform — avoids allocations
    Eigen::MatrixXd X_copy = X;
    scaler.transform_inplace(X_copy);
    std::cout << "In-place standardized:\n" << X_copy << "\n\n";

    // float specialization — 2x SIMD density
    Eigen::MatrixXf Xf = Eigen::MatrixXf::Random(1000, 100);
    Skigen::StandardScaler<float> scalerf;
    Eigen::MatrixXf Zf = scalerf.fit_transform(Xf);
    std::cout << "Float scaler: " << Xf.rows() << "x" << Xf.cols()
              << " -> mean(0)=" << scalerf.mean()(0) << "\n";

    return 0;
}
