// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// nu_svc.cpp — nu-parameterised support vector classification.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.svm import NuSVC
//   import numpy as np
//   X = np.vstack([np.random.randn(10, 2) - 2, np.random.randn(10, 2) + 2])
//   y = np.array([0] * 10 + [1] * 10)
//   clf = NuSVC(nu=0.5, kernel="rbf").fit(X, y)
//   labels = clf.predict(X)

#include <Skigen/SVM>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    //! [example_nu_svc]
    Eigen::MatrixXd X(20, 2);
    Eigen::VectorXi y(20);
    for (int i = 0; i < 20; ++i) {
        const double cls = (i < 10) ? -2.0 : 2.0;
        X(i, 0) = cls + 0.1 * std::sin(0.7 * i);
        X(i, 1) = cls + 0.1 * std::cos(0.7 * i);
        y(i)    = (cls > 0) ? 1 : 0;
    }

    using K = Skigen::NuSVC<double>::Kernel;
    Skigen::NuSVC<double> clf(/*nu=*/0.5, K::RBF);
    clf.fit(X, y);
    const Eigen::VectorXi labels = clf.predict(X);
    //! [example_nu_svc]

    int correct = 0;
    for (int i = 0; i < 20; ++i)
        if (labels(i) == y(i)) ++correct;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== NuSVC ===\n";
    std::cout << "support vectors: " << clf.n_support() << "\n";
    std::cout << "intercept: " << clf.intercept() << "\n";
    std::cout << "accuracy: " << static_cast<double>(correct) / 20.0 << "\n";
    return 0;
}
