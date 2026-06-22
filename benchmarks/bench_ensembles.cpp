// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>
#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <iostream>

int main() {
    constexpr int rows = 2000;
    constexpr int cols = 20;
    constexpr int runs = 5;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(rows, cols);
    Eigen::VectorXi y(rows);
    for (int i = 0; i < rows; ++i) y(i) = i % 3;

    auto bench = [](const std::string& name, auto fn) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < runs; ++r) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
        std::cout << "  " << name << ": " << ms << " ms\n";
    };

    std::cout << "Ensemble benchmarks (" << rows << "x" << cols << ")\n";

    {
        Skigen::RandomForestClassifier<double> rf(50);
        bench("RandomForestClassifier(50).fit", [&]() {
            rf.fit(X, y);
        });
        bench("RandomForestClassifier(50).predict", [&]() {
            auto yh = rf.predict(X);
            static_cast<void>(yh);
        });
    }

    {
        Eigen::VectorXd yr(rows);
        for (int i = 0; i < rows; ++i) yr(i) = static_cast<double>(i) / rows;

        Skigen::GradientBoostingClassifier<double> gbc;
        bench("GradientBoostingClassifier(100).fit", [&]() {
            Eigen::VectorXi yb(rows);
            for (int i = 0; i < rows; ++i) yb(i) = i % 2;
            gbc.fit(X, yb);
        });
        bench("GradientBoostingClassifier(100).predict", [&]() {
            auto yh = gbc.predict(X);
            static_cast<void>(yh);
        });
    }

    {
        // Histogram gradient boosting (native leaf-wise histogram grower).
        Eigen::VectorXd yr(rows);
        for (int i = 0; i < rows; ++i)
            yr(i) = std::sin(0.01 * i) + 0.001 * (i % 7);

        Skigen::HistGradientBoostingRegressor<double> hgb(
            Skigen::HistGradientBoostingRegressor<double>::Loss::SquaredError,
            0.1, 100);
        bench("HistGradientBoostingRegressor(100).fit", [&]() {
            hgb.fit(X, yr);
        });
        bench("HistGradientBoostingRegressor(100).predict", [&]() {
            auto yh = hgb.predict(X);
            static_cast<void>(yh);
        });

        Eigen::VectorXi yb(rows);
        for (int i = 0; i < rows; ++i) yb(i) = i % 2;
        Skigen::HistGradientBoostingClassifier<double> hgc(
            Skigen::HistGradientBoostingClassifier<double>::Loss::LogLoss,
            0.1, 100);
        bench("HistGradientBoostingClassifier(100).fit", [&]() {
            hgc.fit(X, yb);
        });
        bench("HistGradientBoostingClassifier(100).predict", [&]() {
            auto yh = hgc.predict(X);
            static_cast<void>(yh);
        });
    }

    return 0;
}
