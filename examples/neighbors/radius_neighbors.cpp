// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// radius_neighbors.cpp — radius-neighbor classification and regression.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor
//   import numpy as np
//   X = np.array([[0.0], [0.2], [0.4], [2.0], [2.2]])
//   y = np.array([0, 0, 1, 1, 1])
//   clf = RadiusNeighborsClassifier(radius=0.35).fit(X, y)
//   print(clf.predict([[0.1], [2.1]]))

#include <Skigen/Neighbors>

#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X(5, 1);
    X << 0.0, 0.2, 0.4, 2.0, 2.2;
    Eigen::VectorXi y_class(5);
    y_class << 0, 0, 1, 1, 1;
    Eigen::VectorXd y_reg(5);
    y_reg << 1.0, 3.0, 5.0, 10.0, 14.0;
    Eigen::MatrixXd Q(2, 1);
    Q << 0.1, 2.1;

    //! [example_radius_neighbors_classifier]
    Skigen::RadiusNeighborsClassifier<double> clf(/*radius=*/0.35);
    clf.fit(X, y_class);
    Eigen::VectorXi labels = clf.predict(Q);

    std::cout << "=== RadiusNeighborsClassifier ===\n";
    std::cout << "labels: " << labels.transpose() << "\n\n";
    //! [example_radius_neighbors_classifier]

    //! [example_radius_neighbors_regressor]
    Skigen::RadiusNeighborsRegressor<double> reg(/*radius=*/0.35);
    reg.fit(X, y_reg);
    Eigen::VectorXd values = reg.predict(Q);

    std::cout << "=== RadiusNeighborsRegressor ===\n";
    std::cout << "values: " << values.transpose() << "\n";
    //! [example_radius_neighbors_regressor]

    return 0;
}
