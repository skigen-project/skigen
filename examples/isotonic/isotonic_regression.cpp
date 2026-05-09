// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// IsotonicRegression — fit a non-decreasing piecewise-linear function.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.isotonic import IsotonicRegression
//   import numpy as np
//   X = np.arange(10).reshape(-1, 1)
//   y = np.array([1, 3, 2, 4, 6, 5, 7, 8, 9, 11])
//   ir = IsotonicRegression()
//   y_iso = ir.fit_transform(X.ravel(), y)
//   print(y_iso)        # monotone non-decreasing
//   print(ir.predict([[2.5]]))  # interpolated value

#include <Skigen/Isotonic>

#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X(10, 1);
    for (int i = 0; i < 10; ++i) X(i, 0) = i;

    Eigen::VectorXd y(10);
    y << 1, 3, 2, 4, 6, 5, 7, 8, 9, 11;

    Skigen::IsotonicRegression<double> iso;
    iso.fit(X, y);

    Eigen::VectorXd y_iso = iso.predict(X);

    std::cout << "X            y_raw   y_iso\n";
    std::cout << "----------------------------\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "  " << X(i, 0) << "         "
                  << y(i) << "      " << y_iso(i) << "\n";
    }

    Eigen::MatrixXd T(1, 1);
    T(0, 0) = 2.5;
    std::cout << "\nInterpolated at x=2.5: "
              << iso.predict(T)(0) << "\n";
    std::cout << "R^2 on training data: "
              << iso.score(X, y) << "\n";
    return 0;
}
