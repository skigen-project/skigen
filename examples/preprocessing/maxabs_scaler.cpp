// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// maxabs_scaler.cpp — MaxAbsScaler: scale each feature by its max absolute value
#include <Skigen/Preprocessing>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>

int main() {
    Eigen::MatrixXd X(4, 3);
    X <<  1.0, -2.0,  3.0,
         -4.0,  5.0, -6.0,
          2.0, -3.0,  1.0,
          0.5,  1.0, -0.5;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Original data:\n" << X << "\n\n";

    //! [example_max_abs_scaler]
    Skigen::MaxAbsScaler<double> scaler;
    Eigen::MatrixXd Z = scaler.fit_transform(X);

    std::cout << "MaxAbs scaled (range [-1, 1]):\n" << Z << "\n\n";
    std::cout << "Max absolute values: " << scaler.max_abs() << "\n";
    std::cout << "Scale factors:       " << scaler.scale()   << "\n\n";

    // Round-trip
    Eigen::MatrixXd X_back = scaler.inverse_transform(Z);
    std::cout << "Recovered:\n" << X_back << "\n";
    //! [example_max_abs_scaler]

    return 0;
}
