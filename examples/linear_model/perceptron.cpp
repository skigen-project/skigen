// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// perceptron.cpp — Perceptron on a linearly separable dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.linear_model import Perceptron
//   import numpy as np
//   X = np.array([[-2, -1], [-1.5, -1.2], [-1.2, -1.8], [1, 1.2], [1.5, 1.7], [2, 1.3]])
//   y = np.array([2, 2, 2, 9, 9, 9])
//   clf = Perceptron(max_iter=50, eta0=1.0, random_state=0).fit(X, y)
//   print(clf.predict(X))

#include <Skigen/LinearModel>

#include <Eigen/Core>
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

    //! [example_perceptron]
    Skigen::Perceptron<double> clf(/*max_iter=*/50, /*tol=*/0.0,
                                   /*eta0=*/1.0, /*random_state=*/0);
    clf.fit(X, y);
    Eigen::VectorXi pred = clf.predict(X);

    std::cout << "=== Perceptron ===\n";
    std::cout << "coef:\n" << clf.coef() << "\n";
    std::cout << "predictions: " << pred.transpose() << "\n";
    //! [example_perceptron]

    return 0;
}
