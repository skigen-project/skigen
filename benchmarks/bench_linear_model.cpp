// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// bench_linear_model.cpp — Throughput benchmarks for linear models
#include <Skigen/Dense>
#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <iomanip>

template <typename F>
double bench(F&& fn, int runs) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;
}

int main() {
    constexpr int runs = 50;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "=============================================================\n";
    std::cout << "  Linear Model Benchmarks (" << runs << " runs each)\n";
    std::cout << "=============================================================\n\n";

    // -------------------------------------------------------------------
    // Test several problem sizes
    // -------------------------------------------------------------------
    for (auto [rows, cols] : std::initializer_list<std::pair<int, int>>{
             {1000, 10}, {5000, 50}, {10000, 100}}) {

        Eigen::MatrixXd X = Eigen::MatrixXd::Random(rows, cols);
        Eigen::VectorXd y = Eigen::VectorXd::Random(rows);

        std::cout << "--- " << rows << " x " << cols << " ---\n";

        // OLS
        {
            Skigen::LinearRegression<double> est;
            double fit_us = bench([&] { est.fit(X, y); }, runs);
            double pred_us = bench([&] {
                Eigen::VectorXd p = est.predict(X);
                static_cast<void>(p);
            }, runs);
            std::cout << "  LinearRegression  fit: " << std::setw(10) << fit_us
                      << " us   predict: " << std::setw(8) << pred_us << " us\n";
        }

        // Ridge
        {
            Skigen::Ridge<double> est(1.0);
            double fit_us = bench([&] { est.fit(X, y); }, runs);
            double pred_us = bench([&] {
                Eigen::VectorXd p = est.predict(X);
                static_cast<void>(p);
            }, runs);
            std::cout << "  Ridge(1.0)        fit: " << std::setw(10) << fit_us
                      << " us   predict: " << std::setw(8) << pred_us << " us\n";
        }

        // Lasso
        {
            Skigen::Lasso<double> est(0.01);
            double fit_us = bench([&] { est.fit(X, y); }, runs);
            double pred_us = bench([&] {
                Eigen::VectorXd p = est.predict(X);
                static_cast<void>(p);
            }, runs);
            std::cout << "  Lasso(0.01)       fit: " << std::setw(10) << fit_us
                      << " us   predict: " << std::setw(8) << pred_us << " us\n";
        }

        // ElasticNet
        {
            Skigen::ElasticNet<double> est(0.01, 0.5);
            double fit_us = bench([&] { est.fit(X, y); }, runs);
            double pred_us = bench([&] {
                Eigen::VectorXd p = est.predict(X);
                static_cast<void>(p);
            }, runs);
            std::cout << "  ElasticNet(0.01)  fit: " << std::setw(10) << fit_us
                      << " us   predict: " << std::setw(8) << pred_us << " us\n";
        }

        std::cout << "\n";
    }

    // -------------------------------------------------------------------
    // QuantileRegressor (LP interior-point) — benchmarked separately on a
    // small structured problem. The standard-form LP has one equality
    // constraint per sample, so the per-iteration dense Cholesky is
    // O(n_samples^3); a sparse / specialised LP backend for large n is
    // tracked as future work.
    // -------------------------------------------------------------------
    {
        constexpr int qn = 200, qp = 5;
        Eigen::MatrixXd Xq(qn, qp);
        Eigen::VectorXd yq(qn);
        for (int i = 0; i < qn; ++i) {
            for (int j = 0; j < qp; ++j)
                Xq(i, j) = static_cast<double>((i * (j + 1)) % 13) / 13.0;
            yq(i) = 2.0 * Xq(i, 0) - Xq(i, 1) + 0.5;
        }
        Skigen::QuantileRegressor<double> est(0.5, 0.0);
        double fit_us = bench([&] { est.fit(Xq, yq); }, 3);
        std::cout << "--- QuantileRegressor (" << qn << " x " << qp << ") ---\n";
        std::cout << "  QuantileRegressor fit: " << std::setw(10) << fit_us
                  << " us\n\n";
    }

    // -------------------------------------------------------------------
    // Throughput summary (elements/sec on largest problem)
    // -------------------------------------------------------------------
    constexpr int N = 10000;
    constexpr int P = 100;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, P);
    Eigen::VectorXd y = Eigen::VectorXd::Random(N);

    Skigen::LinearRegression<double> ols;
    double fit_us = bench([&] { ols.fit(X, y); }, runs);

    double elements = static_cast<double>(N) * P;
    double throughput = elements / (fit_us / 1e6);  // elements/sec

    std::cout << "=== OLS Throughput (" << N << "x" << P << ") ===\n";
    std::cout << "  " << throughput / 1e6 << " M elements/sec (fit)\n";
    std::cout << "  " << fit_us / elements * 1e3 << " ns/element\n";

    return 0;
}
