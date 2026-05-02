// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// minmax_scaler.cpp — MinMaxScaler: scale features to a given range
#include <Skigen/Preprocessing>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>

int main() {
    Eigen::MatrixXd X(5, 3);
    X << -1.0,  2.0,  0.0,
          0.0,  0.0,  1.0,
          1.0, -1.0,  2.0,
          2.0,  3.0, -1.0,
          3.0,  1.0,  0.5;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Original data:\n" << X << "\n\n";

    // Default range [0, 1]
    Skigen::MinMaxScaler<double> scaler;
    Eigen::MatrixXd Z = scaler.fit_transform(X);

    std::cout << "Scaled to [0, 1]:\n" << Z << "\n\n";
    std::cout << "Data min:   " << scaler.data_min()   << "\n";
    std::cout << "Data max:   " << scaler.data_max()   << "\n";
    std::cout << "Data range: " << scaler.data_range() << "\n\n";

    // Custom range [-1, 1]
    Skigen::MinMaxScaler<double> scaler2({-1.0, 1.0});
    Eigen::MatrixXd Z2 = scaler2.fit_transform(X);
    std::cout << "Scaled to [-1, 1]:\n" << Z2 << "\n\n";

    // Round-trip
    Eigen::MatrixXd X_back = scaler.inverse_transform(Z);
    std::cout << "Recovered:\n" << X_back << "\n";

    return 0;
}
