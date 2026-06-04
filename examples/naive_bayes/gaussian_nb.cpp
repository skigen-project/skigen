// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// gaussian_nb.cpp — GaussianNB classifier on a tiny 2D dataset
//
// Equivalent sklearn snippet:
//
//     from sklearn.naive_bayes import GaussianNB
//     X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
//     y = [0, 0, 0, 1, 1, 1]
//     clf = GaussianNB()
//     clf.fit(X, y)
//     print(clf.predict([[-0.8, -1]]))

#include <Skigen/Dense>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    Eigen::MatrixXd X(6, 2);
    X << -2, -1,
         -1, -1,
         -1, -2,
          1,  1,
          1,  2,
          2,  1;
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    Skigen::GaussianNB<double> nb;
    nb.fit(X, y);

    Eigen::MatrixXd Xtest(2, 2);
    Xtest << -0.8, -1.0,
              0.5,  0.5;
    auto pred = nb.predict(Xtest);
    Eigen::MatrixXd proba = nb.predict_proba(Xtest);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== GaussianNB ===\n";
    std::cout << "theta_:\n" << nb.theta() << "\n";
    std::cout << "var_:\n"   << nb.var()   << "\n";
    std::cout << "predictions: " << pred.transpose() << "\n";
    std::cout << "predict_proba:\n" << proba << "\n";
    return 0;
}
