// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// rfe.cpp — Recursive feature elimination wrapper around Ridge.
#include <Skigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <variant>

int main() {
    Eigen::MatrixXd X(10, 4);
    X << 1.0, 0.1, -2.0,  0.05,
         2.0, 0.2, -4.0, -0.05,
         3.0, 0.3, -6.0,  0.10,
         4.0, 0.1, -8.0, -0.10,
         5.0, 0.2,-10.0,  0.05,
         6.0, 0.3,-12.0, -0.05,
         7.0, 0.1,-14.0,  0.10,
         8.0, 0.2,-16.0, -0.10,
         9.0, 0.3,-18.0,  0.05,
        10.0, 0.1,-20.0, -0.05;
    Eigen::VectorXd y = X.col(0) * 3.0 + X.col(2) * (-1.5);

    std::cout << "Input shape: " << X.rows() << " x " << X.cols() << "\n";

    Skigen::Ridge<double> ridge(0.01);
    using NF = std::variant<int, double>;
    Skigen::RFE<Skigen::Ridge<double>> rfe(
        ridge, std::optional<NF>{NF{2}}, NF{1});
    rfe.fit(X, y);

    std::cout << "Ranking: " << rfe.ranking().transpose() << "\n";
    std::cout << "Selected: ";
    auto idx = rfe.get_support_indices();
    for (Eigen::Index i = 0; i < idx.size(); ++i) std::cout << idx(i) << ' ';
    std::cout << "\n";

    Eigen::MatrixXd Xs = rfe.transform(X);
    std::cout << "Output shape: " << Xs.rows() << " x " << Xs.cols() << "\n";
    return 0;
}
