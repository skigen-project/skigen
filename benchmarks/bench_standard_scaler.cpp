// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// bench_standard_scaler.cpp — Placeholder for energy/cycles-per-op profiling
#include <Skigen/Dense>
#include <Eigen/Core>
#include <chrono>
#include <iostream>

int main() {
    constexpr int rows = 10000;
    constexpr int cols = 100;
    constexpr int runs = 100;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(rows, cols);

    // Benchmark fit
    Skigen::StandardScaler scaler;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        scaler.fit(X);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double fit_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    // Benchmark transform
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        Eigen::MatrixXd Z = scaler.transform(X);
        static_cast<void>(Z);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double transform_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    // Benchmark transform_inplace
    Eigen::MatrixXd X_mut = X;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        X_mut = X;
        scaler.transform_inplace(X_mut);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double inplace_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    double elements = static_cast<double>(rows) * cols;
    std::cout << "StandardScaler (" << rows << "x" << cols << ")\n";
    std::cout << "  fit:                " << fit_us << " us  ("
              << fit_us * 1000.0 / elements << " ns/elem)\n";
    std::cout << "  transform:          " << transform_us << " us  ("
              << transform_us * 1000.0 / elements << " ns/elem)\n";
    std::cout << "  transform_inplace:  " << inplace_us << " us  ("
              << inplace_us * 1000.0 / elements << " ns/elem)\n";

    return 0;
}
