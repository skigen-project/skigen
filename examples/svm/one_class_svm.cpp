// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// one_class_svm.cpp — OneClassSVM is currently a placeholder; fit() throws.
//
// The constructor signature mirrors
// https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
// but the nu-SVM solver that backs this estimator is not yet
// implemented. Use Skigen::LocalOutlierFactor for unsupervised outlier
// detection in this release.

#include <Skigen/SVM>

#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 2);

    using K = Skigen::OneClassSVM<double>::Kernel;
    Skigen::OneClassSVM<double> det(K::RBF);

    std::cout << "=== OneClassSVM (placeholder) ===\n";
    try {
        det.fit(X);
        std::cout << "  unexpectedly succeeded\n";
    } catch (const std::exception& e) {
        std::cout << "  fit() threw as documented: " << e.what() << "\n";
    }
    return 0;
}
