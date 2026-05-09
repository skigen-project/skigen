// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// multinomial_nb.cpp — MultinomialNB on a tiny count dataset
//
// Equivalent sklearn snippet:
//
//     from sklearn.naive_bayes import MultinomialNB
//     X = [[2, 1, 0], [3, 0, 1], [1, 2, 0],
//          [0, 1, 3], [0, 0, 4], [1, 0, 5]]
//     y = [0, 0, 0, 1, 1, 1]
//     clf = MultinomialNB(alpha=1.0)
//     clf.fit(X, y)
//     print(clf.predict_proba(X))

#include <Skigen/Dense>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    Eigen::MatrixXd X(6, 3);
    X << 2, 1, 0,
         3, 0, 1,
         1, 2, 0,
         0, 1, 3,
         0, 0, 4,
         1, 0, 5;
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    Skigen::MultinomialNB<double> nb(/*alpha=*/1.0);
    nb.fit(X, y);

    auto pred = nb.predict(X);
    Eigen::MatrixXd proba = nb.predict_proba(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== MultinomialNB ===\n";
    std::cout << "feature_count_:\n"    << nb.feature_count()    << "\n";
    std::cout << "feature_log_prob_:\n" << nb.feature_log_prob() << "\n";
    std::cout << "class_log_prior_: "   << nb.class_log_prior().transpose() << "\n";
    std::cout << "predictions:     "    << pred.transpose() << "\n";
    std::cout << "predict_proba:\n"     << proba << "\n";
    return 0;
}
