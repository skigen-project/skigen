// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// simple_imputer.cpp — dense numeric missing-value imputation.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.impute import SimpleImputer, MissingIndicator
//   import numpy as np
//   X = np.array([[1.0, np.nan, 3.0],
//                 [2.0, 4.0, np.nan],
//                 [np.nan, 6.0, 9.0]])
//   Z = SimpleImputer(strategy="mean").fit_transform(X)
//   M = MissingIndicator(features="all").fit_transform(X)

#include <Skigen/Impute>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <limits>

int main() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXd X(3, 3);
    X << 1.0, nan, 3.0,
         2.0, 4.0, nan,
         nan, 6.0, 9.0;

    //! [example_simple_imputer]
    Skigen::SimpleImputer<double> imputer;
    Eigen::MatrixXd Z = imputer.fit_transform(X);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== SimpleImputer (mean) ===\n";
    std::cout << "statistics: " << imputer.statistics() << "\n";
    std::cout << "transformed:\n" << Z << "\n\n";
    //! [example_simple_imputer]

    //! [example_missing_indicator]
    Skigen::MissingIndicator<double> indicator(
        nan, Skigen::MissingIndicatorFeatures::All);
    Eigen::MatrixXd M = indicator.fit_transform(X);

    std::cout << "=== MissingIndicator (all features) ===\n";
    std::cout << M << "\n";
    //! [example_missing_indicator]

    return 0;
}
