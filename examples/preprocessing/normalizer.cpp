// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// normalizer.cpp — Normalizer: scale each sample to unit norm
#include <Skigen/Preprocessing>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>

int main() {
    Eigen::MatrixXd X(4, 3);
    X << 4.0, 1.0, 2.0,
         1.0, 3.0, 9.0,
         5.0, 7.0, 5.0,
         0.0, 0.0, 0.0;   // zero row — stays zero

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Original data:\n" << X << "\n\n";

    // L2 normalization (default) — each row has unit Euclidean norm
    Skigen::Normalizer<double> l2_norm(Skigen::Norm::L2);
    Eigen::MatrixXd Z_l2 = l2_norm.fit_transform(X);

    std::cout << "L2 normalized:\n" << Z_l2 << "\n";
    std::cout << "Row norms: ";
    for (Eigen::Index i = 0; i < Z_l2.rows(); ++i)
        std::cout << Z_l2.row(i).norm() << " ";
    std::cout << "\n\n";

    // L1 normalization — each row sums to 1 (in absolute value)
    Skigen::Normalizer<double> l1_norm(Skigen::Norm::L1);
    Eigen::MatrixXd Z_l1 = l1_norm.fit_transform(X);

    std::cout << "L1 normalized:\n" << Z_l1 << "\n";
    std::cout << "Row abs sums: ";
    for (Eigen::Index i = 0; i < Z_l1.rows(); ++i)
        std::cout << Z_l1.row(i).cwiseAbs().sum() << " ";
    std::cout << "\n\n";

    // Max normalization — each row's max absolute value is 1
    Skigen::Normalizer<double> max_norm(Skigen::Norm::Max);
    Eigen::MatrixXd Z_max = max_norm.fit_transform(X);

    std::cout << "Max normalized:\n" << Z_max << "\n";

    return 0;
}
