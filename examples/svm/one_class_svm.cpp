// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// one_class_svm.cpp — unsupervised outlier detection with a one-class SVM.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.svm import OneClassSVM
//   import numpy as np
//   X = np.vstack([np.random.randn(10, 2) * 0.2, [[6, 6], [-6, 5]]])
//   oc = OneClassSVM(kernel="rbf", nu=0.2).fit(X)
//   labels = oc.predict(X)   # +1 inlier, -1 outlier

#include <Skigen/SVM>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    //! [example_one_class_svm]
    Eigen::MatrixXd X(12, 2);
    for (int i = 0; i < 10; ++i) {
        X(i, 0) = 0.2 * std::sin(static_cast<double>(i));
        X(i, 1) = 0.2 * std::cos(static_cast<double>(i));
    }
    X(10, 0) = 6.0;  X(10, 1) = 6.0;
    X(11, 0) = -6.0; X(11, 1) = 5.0;

    using K = Skigen::OneClassSVM<double>::Kernel;
    Skigen::OneClassSVM<double> det(K::RBF, 3, 0.0, 0.0, /*nu=*/0.2);
    det.fit(X);
    const Eigen::VectorXi labels = det.predict(X);
    //! [example_one_class_svm]

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== OneClassSVM ===\n";
    std::cout << "support vectors: " << det.n_support() << "\n";
    std::cout << "offset: " << det.offset() << "\n";
    std::cout << "labels: " << labels.transpose() << "\n";
    return 0;
}
