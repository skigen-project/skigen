// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// nu_svr.cpp — nu-parameterised support vector regression.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.svm import NuSVR
//   import numpy as np
//   X = np.linspace(0, 1, 30).reshape(-1, 1)
//   y = 1.5 * X.ravel() + 0.2
//   reg = NuSVR(nu=0.5, C=5.0, kernel="linear").fit(X, y)
//   pred = reg.predict(X)

#include <Skigen/SVM>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    //! [example_nu_svr]
    Eigen::MatrixXd X(30, 1);
    Eigen::VectorXd y(30);
    for (int i = 0; i < 30; ++i) {
        X(i, 0) = static_cast<double>(i) / 30.0;
        y(i)    = 1.5 * X(i, 0) + 0.2;
    }

    using K = Skigen::NuSVR<double>::Kernel;
    Skigen::NuSVR<double> reg(/*nu=*/0.5, /*C=*/5.0, K::Linear);
    reg.fit(X, y);
    const Eigen::VectorXd pred = reg.predict(X);
    //! [example_nu_svr]

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== NuSVR ===\n";
    std::cout << "support vectors: " << reg.n_support() << "\n";
    std::cout << "fitted epsilon: " << reg.epsilon_fitted() << "\n";
    std::cout << "R^2: " << reg.score(X, y) << "\n";
    return 0;
}
