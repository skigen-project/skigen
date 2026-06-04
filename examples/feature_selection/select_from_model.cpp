// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// select_from_model.cpp — Threshold features by an estimator's coefficients.
#include <Skigen/Dense>
#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X(8, 3);
    X << 1, 0.1,  0.0,
         2, 0.2,  0.1,
         3, 0.3, -0.1,
         4, 0.1,  0.2,
         5, 0.2, -0.2,
         6, 0.3,  0.0,
         7, 0.1,  0.1,
         8, 0.2, -0.1;
    Eigen::VectorXd y = X.col(0) * 5.0;  // strong signal in column 0

    std::cout << "Input shape: " << X.rows() << " x " << X.cols() << "\n";

    Skigen::Ridge<double> ridge(0.01);
    Skigen::SelectFromModel<Skigen::Ridge<double>> sfm(
        ridge, std::string("mean"));
    sfm.fit(X, y);

    Eigen::MatrixXd Xs = sfm.transform(X);
    std::cout << "Coefficients: " << sfm.estimator().coef() << "\n";
    std::cout << "Threshold:    " << sfm.threshold_value() << "\n";
    std::cout << "Selected: ";
    auto idx = sfm.get_support_indices();
    for (Eigen::Index i = 0; i < idx.size(); ++i) std::cout << idx(i) << ' ';
    std::cout << "\nOutput shape: " << Xs.rows() << " x " << Xs.cols() << "\n";
    return 0;
}
