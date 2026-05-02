// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// linear_regression.cpp — Ordinary Least Squares regression
#include <Skigen/LinearModel>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>

int main() {
    // Simple 2-feature dataset: y = 2*x1 + 3*x2 + 1
    Eigen::MatrixXd X(6, 2);
    X << 1, 1,
         1, 2,
         2, 2,
         2, 3,
         3, 3,
         3, 4;
    Eigen::VectorXd y(6);
    y << 6, 9, 8, 11, 10, 13;

    Skigen::LinearRegression<double> model;
    model.fit(X, y);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Linear Regression ===\n";
    std::cout << "Coefficients: " << model.coef() << "\n";
    std::cout << "Intercept:    " << model.intercept() << "\n";
    std::cout << "R² (train):   " << model.score(X, y) << "\n\n";

    // Predict on new data
    Eigen::MatrixXd X_new(2, 2);
    X_new << 4, 5,
             5, 6;
    Eigen::VectorXd y_pred = model.predict(X_new);
    std::cout << "Predictions for [[4,5],[5,6]]:\n" << y_pred.transpose() << "\n";

    return 0;
}
