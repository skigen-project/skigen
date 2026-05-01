// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// bench_preprocessing.cpp — Comparative benchmark of all preprocessing scalers
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
    constexpr int rows = 10000;
    constexpr int cols = 100;
    constexpr int runs = 100;
    constexpr double elements = static_cast<double>(rows) * cols;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(rows, cols);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "================================================================\n";
    std::cout << "  Preprocessing Benchmark (" << rows << " x " << cols
              << ", " << runs << " runs)\n";
    std::cout << "================================================================\n";
    std::cout << std::setw(22) << "Scaler"
              << std::setw(12) << "fit (us)"
              << std::setw(14) << "transform (us)"
              << std::setw(12) << "ns/elem" << "\n";
    std::cout << "----------------------------------------------------------------\n";

    auto report = [&](const char* name, double fit_us, double trans_us) {
        std::cout << std::setw(22) << name
                  << std::setw(12) << fit_us
                  << std::setw(14) << trans_us
                  << std::setw(12) << trans_us * 1000.0 / elements << "\n";
    };

    // StandardScaler
    {
        Skigen::StandardScaler<double> s;
        double fit_us = bench([&] { s.fit(X); }, runs);
        double trans_us = bench([&] {
            Eigen::MatrixXd Z = s.transform(X);
            static_cast<void>(Z);
        }, runs);
        report("StandardScaler", fit_us, trans_us);
    }

    // MinMaxScaler
    {
        Skigen::MinMaxScaler<double> s;
        double fit_us = bench([&] { s.fit(X); }, runs);
        double trans_us = bench([&] {
            Eigen::MatrixXd Z = s.transform(X);
            static_cast<void>(Z);
        }, runs);
        report("MinMaxScaler", fit_us, trans_us);
    }

    // MaxAbsScaler
    {
        Skigen::MaxAbsScaler<double> s;
        double fit_us = bench([&] { s.fit(X); }, runs);
        double trans_us = bench([&] {
            Eigen::MatrixXd Z = s.transform(X);
            static_cast<void>(Z);
        }, runs);
        report("MaxAbsScaler", fit_us, trans_us);
    }

    // RobustScaler
    {
        Skigen::RobustScaler<double> s;
        double fit_us = bench([&] { s.fit(X); }, runs);
        double trans_us = bench([&] {
            Eigen::MatrixXd Z = s.transform(X);
            static_cast<void>(Z);
        }, runs);
        report("RobustScaler", fit_us, trans_us);
    }

    // Normalizer (row-wise, stateless — no fit needed)
    {
        Skigen::Normalizer<double> s;
        double trans_us = bench([&] {
            Eigen::MatrixXd Z = s.fit_transform(X);
            static_cast<void>(Z);
        }, runs);
        report("Normalizer(L2)", 0.0, trans_us);
    }

    // StandardScaler in-place
    {
        Skigen::StandardScaler<double> s;
        s.fit(X);
        Eigen::MatrixXd X_mut = X;
        double trans_us = bench([&] {
            X_mut = X;
            s.transform_inplace(X_mut);
        }, runs);
        report("StandardScaler(ip)", 0.0, trans_us);
    }

    std::cout << "\n";

    // -------------------------------------------------------------------
    // Throughput summary
    // -------------------------------------------------------------------
    Skigen::StandardScaler<double> ss;
    double trans_us = bench([&] {
        ss.fit(X);
        Eigen::MatrixXd Z = ss.transform(X);
        static_cast<void>(Z);
    }, runs);

    double throughput = elements / (trans_us / 1e6);
    std::cout << "=== StandardScaler Throughput ===\n";
    std::cout << "  " << throughput / 1e6 << " M elements/sec (transform)\n";
    std::cout << "  " << trans_us * 1000.0 / elements << " ns/element\n";

    return 0;
}
