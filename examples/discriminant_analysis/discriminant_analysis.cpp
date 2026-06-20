// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// discriminant_analysis.cpp — LDA and QDA on a small dense dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
//   import numpy as np
//   X = np.array([[-2.0, -1.0], [-1.6, -1.5], [-1.2, -0.8], [-1.8, -1.1],
//                 [ 1.0,  1.7], [ 1.4,  1.1], [ 1.8,  1.6], [ 1.2,  1.3]])
//   y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
//   lda = LinearDiscriminantAnalysis().fit(X, y)
//   qda = QuadraticDiscriminantAnalysis(reg_param=1e-6).fit(X, y)

#include <Skigen/DiscriminantAnalysis>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    Eigen::MatrixXd X(8, 2);
    X << -2.0, -1.0,
         -1.6, -1.5,
         -1.2, -0.8,
         -1.8, -1.1,
          1.0,  1.7,
          1.4,  1.1,
          1.8,  1.6,
          1.2,  1.3;
    Eigen::VectorXi y(8);
    y << 0, 0, 0, 0, 1, 1, 1, 1;

    std::cout << std::fixed << std::setprecision(4);

    //! [example_lda]
    Skigen::LinearDiscriminantAnalysis<double> lda;
    lda.fit(X, y);
    Eigen::VectorXi lda_pred = lda.predict(X);

    std::cout << "=== LinearDiscriminantAnalysis ===\n";
    std::cout << "priors: " << lda.priors().transpose() << "\n";
    std::cout << "first prediction: " << lda_pred(0) << "\n\n";
    //! [example_lda]

    //! [example_qda]
    Skigen::QuadraticDiscriminantAnalysis<double> qda(/*reg_param=*/1e-6);
    qda.fit(X, y);
    Eigen::VectorXi qda_pred = qda.predict(X);

    std::cout << "=== QuadraticDiscriminantAnalysis ===\n";
    std::cout << "priors: " << qda.priors().transpose() << "\n";
    std::cout << "first prediction: " << qda_pred(0) << "\n";
    //! [example_qda]

    return 0;
}
