// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// variance_threshold.cpp — Drop low-variance (constant) features.
#include <Skigen/FeatureSelection>
#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X(5, 4);
    X << 1.0, 5.0, 0.1, 2.0,
         2.0, 5.0, 0.2, 4.0,
         3.0, 5.0, 0.3, 6.0,
         4.0, 5.0, 0.4, 8.0,
         5.0, 5.0, 0.5, 10.0;

    std::cout << "Input shape: " << X.rows() << " x " << X.cols() << "\n";

    Skigen::VarianceThreshold<double> vt(/*threshold=*/0.0);
    Eigen::MatrixXd Xs = vt.fit(X).transform(X);

    std::cout << "Variances: " << vt.variances() << "\n";
    std::cout << "Selected: ";
    auto idx = vt.get_support_indices();
    for (Eigen::Index i = 0; i < idx.size(); ++i) std::cout << idx(i) << ' ';
    std::cout << "\n";
    std::cout << "Output shape: " << Xs.rows() << " x " << Xs.cols() << "\n";
    return 0;
}
