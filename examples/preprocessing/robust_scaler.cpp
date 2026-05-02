// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// robust_scaler.cpp — RobustScaler: scaling using median and IQR (outlier-robust)
#include <Skigen/Preprocessing>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>

int main() {
    // Data with outliers in the first feature
    Eigen::MatrixXd X(6, 2);
    X <<  1.0, 2.0,
          2.0, 3.0,
          3.0, 4.0,
          4.0, 5.0,
          5.0, 6.0,
        100.0, 7.0;   // outlier in feature 0

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Original data (note outlier in col 0):\n" << X << "\n\n";

    // RobustScaler uses median and IQR — resistant to outliers
    Skigen::RobustScaler<double> robust;
    Eigen::MatrixXd Z_robust = robust.fit_transform(X);

    std::cout << "Robust-scaled:\n" << Z_robust << "\n\n";
    std::cout << "Center (median): " << robust.center() << "\n";
    std::cout << "Scale  (IQR):    " << robust.scale()  << "\n\n";

    // Compare with StandardScaler — outlier distorts the result
    Skigen::StandardScaler<double> standard;
    Eigen::MatrixXd Z_std = standard.fit_transform(X);

    std::cout << "Standard-scaled (outlier-sensitive):\n" << Z_std << "\n\n";

    // Round-trip
    Eigen::MatrixXd X_back = robust.inverse_transform(Z_robust);
    std::cout << "Recovered:\n" << X_back << "\n";

    return 0;
}
