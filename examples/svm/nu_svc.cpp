// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// nu_svc.cpp — NuSVC is currently a placeholder; fit() throws.
//
// The constructor signature mirrors
// https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
// so user code can be written against the API surface, but the nu-SVM
// SMO variant is not yet implemented. This example demonstrates the
// expected exception so users can plan migrations.

#include <Skigen/SVM>

#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 2);
    Eigen::VectorXi y(20);
    for (int i = 0; i < 20; ++i) y(i) = (i < 10) ? 0 : 1;

    using K = Skigen::NuSVC<double>::Kernel;
    Skigen::NuSVC<double> clf(/*nu=*/0.5, K::RBF);

    std::cout << "=== NuSVC (placeholder) ===\n";
    try {
        clf.fit(X, y);
        std::cout << "  unexpectedly succeeded\n";
    } catch (const std::exception& e) {
        std::cout << "  fit() threw as documented: " << e.what() << "\n";
    }
    return 0;
}
