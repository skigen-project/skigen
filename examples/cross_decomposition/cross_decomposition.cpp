// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// cross_decomposition.cpp — PLSRegression and CCA on paired dense matrices.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.cross_decomposition import PLSRegression, CCA
//   import numpy as np
//   X = np.array([[0.0, 0.0, 1.0], [1.0, 0.5, 0.2], [2.0, 1.0, -0.1],
//                 [3.0, 1.5, -0.5], [4.0, 2.0, -1.0], [5.0, 2.5, -1.4]])
//   Y = np.column_stack([1.5 * X[:, 0] - 0.4 * X[:, 2],
//                        -0.2 * X[:, 0] + 0.8 * X[:, 1] + 0.3 * X[:, 2]])
//   pls = PLSRegression(n_components=2).fit(X, Y)
//   cca = CCA(n_components=2).fit(X, Y)

#include <Skigen/CrossDecomposition>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    Eigen::MatrixXd X(6, 3);
    X << 0.0, 0.0,  1.0,
         1.0, 0.5,  0.2,
         2.0, 1.0, -0.1,
         3.0, 1.5, -0.5,
         4.0, 2.0, -1.0,
         5.0, 2.5, -1.4;
    Eigen::MatrixXd Y(6, 2);
    Y.col(0) = (1.5 * X.col(0).array() - 0.4 * X.col(2).array()).matrix();
    Y.col(1) = (-0.2 * X.col(0).array() + 0.8 * X.col(1).array() + 0.3 * X.col(2).array()).matrix();

    std::cout << std::fixed << std::setprecision(4);

    //! [example_pls_regression]
    Skigen::PLSRegression<double> pls(/*n_components=*/2);
    pls.fit(X, Y);
    Eigen::MatrixXd pls_prediction = pls.predict(X);

    std::cout << "=== PLSRegression ===\n";
    std::cout << "R2: " << pls.score(X, Y) << "\n";
    std::cout << "first prediction: " << pls_prediction.row(0) << "\n\n";
    //! [example_pls_regression]

    //! [example_cca]
    Skigen::CCA<double> cca(/*n_components=*/2);
    cca.fit(X, Y);
    auto [x_scores, y_scores] = cca.transform(X, Y);

    std::cout << "=== CCA ===\n";
    std::cout << "x score head: " << x_scores.row(0) << "\n";
    std::cout << "y score head: " << y_scores.row(0) << "\n";
    //! [example_cca]

    return 0;
}
