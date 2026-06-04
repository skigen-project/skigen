// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// select_k_best.cpp — Pick the k features with the highest univariate score.
#include <Skigen/FeatureSelection>
#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X(6, 3);
    X << 1.0, 0.3, 5.0,
         2.0, 0.5, 5.5,
         1.5, 0.1, 5.2,
        10.0, 0.4, 50.0,
        11.0, 0.2, 51.0,
        10.5, 0.6, 50.5;
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    std::cout << "Input shape: " << X.rows() << " x " << X.cols() << "\n";

    Skigen::SelectKBest<double> sel(
        Skigen::feature_selection::FClassif<double>{}, /*k=*/2);
    sel.fit(X, y);
    Eigen::MatrixXd Xs = sel.transform(X);

    std::cout << "Scores:  " << sel.scores() << "\n";
    std::cout << "p-vals:  " << sel.pvalues() << "\n";
    std::cout << "Selected: ";
    auto idx = sel.get_support_indices();
    for (Eigen::Index i = 0; i < idx.size(); ++i) std::cout << idx(i) << ' ';
    std::cout << "\nOutput shape: " << Xs.rows() << " x " << Xs.cols() << "\n";
    return 0;
}
