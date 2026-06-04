// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// nu_svr.cpp — NuSVR is currently a placeholder; fit() throws.
//
// The constructor signature mirrors
// https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
// but the nu-SVM solver is not yet implemented. Use Skigen::SVR instead
// for kernel regression in this release.

#include <Skigen/SVM>

#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 2);
    Eigen::VectorXd y = Eigen::VectorXd::Random(20);

    using K = Skigen::NuSVR<double>::Kernel;
    Skigen::NuSVR<double> reg(/*nu=*/0.5, /*C=*/1.0, K::RBF);

    std::cout << "=== NuSVR (placeholder) ===\n";
    try {
        reg.fit(X, y);
        std::cout << "  unexpectedly succeeded\n";
    } catch (const std::exception& e) {
        std::cout << "  fit() threw as documented: " << e.what() << "\n";
    }
    return 0;
}
