// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>
#include <Eigen/Core>
#include <chrono>
#include <iostream>

int main() {
    constexpr int rows = 1000;
    constexpr int cols = 20;
    constexpr int runs = 5;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(rows, cols);

    auto bench = [](const std::string& name, auto fn) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < runs; ++r) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
        std::cout << "  " << name << ": " << ms << " ms\n";
    };

    std::cout << "MLP benchmarks (" << rows << "x" << cols << ")\n";

    {
        Eigen::VectorXd y(rows);
        for (int i = 0; i < rows; ++i)
            y(i) = X.row(i).sum() + 0.1 * static_cast<double>(i % 5);

        Skigen::MLPRegressor<double> reg(
            {64, 32}, Skigen::MLPActivation::ReLU, Skigen::MLPSolver::Adam,
            1e-4, 1e-3, 100);
        bench("MLPRegressor(Adam,[64,32]).fit", [&]() {
            reg.fit(X, y);
        });
        bench("MLPRegressor(Adam,[64,32]).predict", [&]() {
            auto yh = reg.predict(X);
            static_cast<void>(yh);
        });

        Skigen::MLPRegressor<double> reg_sgd(
            {64, 32}, Skigen::MLPActivation::ReLU, Skigen::MLPSolver::SGD,
            1e-4, 1e-3, 100);
        bench("MLPRegressor(SGD,[64,32]).fit", [&]() {
            reg_sgd.fit(X, y);
        });
    }

    {
        Eigen::VectorXi y(rows);
        for (int i = 0; i < rows; ++i) y(i) = i % 3;

        Skigen::MLPClassifier<double> clf(
            {64, 32}, Skigen::MLPActivation::ReLU, Skigen::MLPSolver::Adam,
            1e-4, 1e-3, 100);
        bench("MLPClassifier(Adam,[64,32]).fit", [&]() {
            clf.fit(X, y);
        });
        bench("MLPClassifier(Adam,[64,32]).predict", [&]() {
            auto yh = clf.predict(X);
            static_cast<void>(yh);
        });
    }

    return 0;
}
