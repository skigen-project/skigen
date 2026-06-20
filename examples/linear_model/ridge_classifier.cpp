// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// ridge_classifier.cpp — RidgeClassifier on a small separable dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.linear_model import RidgeClassifier
//   import numpy as np
//   X = np.array([[-2, -1], [-1.5, -1.2], [-1.2, -1.8], [1, 1.2], [1.5, 1.7], [2, 1.3]])
//   y = np.array([2, 2, 2, 9, 9, 9])
//   clf = RidgeClassifier(alpha=1.0).fit(X, y)
//   print(clf.predict(X))

#include <Skigen/LinearModel>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    Eigen::MatrixXd X(6, 2);
    X << -2.0, -1.0,
         -1.5, -1.2,
         -1.2, -1.8,
          1.0,  1.2,
          1.5,  1.7,
          2.0,  1.3;
    Eigen::VectorXi y(6);
    y << 2, 2, 2, 9, 9, 9;

    //! [example_ridge_classifier]
    Skigen::RidgeClassifier<double> clf(/*alpha=*/1.0);
    clf.fit(X, y);
    Eigen::VectorXi pred = clf.predict(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== RidgeClassifier ===\n";
    std::cout << "classes: " << clf.classes().transpose() << "\n";
    std::cout << "coef: " << clf.coef() << "\n";
    std::cout << "predictions: " << pred.transpose() << "\n";
    //! [example_ridge_classifier]

    return 0;
}
