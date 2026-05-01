// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// bench_float_vs_double.cpp — Proves 2× SIMD throughput density for float
//
// Key insight: 256-bit SIMD registers fit 8 floats but only 4 doubles.
// Eigen auto-vectorizes, so float operations should approach 2× throughput
// for memory-bandwidth-bound operations.
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

template <typename Scalar>
struct BenchResult {
    double ss_trans;
    double lr_fit;
    double rr_fit;
};

template <typename Scalar>
BenchResult<Scalar> run_benchmark(const char* label, int rows, int cols, int runs) {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Matrix X = Matrix::Random(rows, cols);
    Vector y = Vector::Random(rows);

    double elements = static_cast<double>(rows) * cols;

    // StandardScaler
    Skigen::StandardScaler<Scalar> scaler;
    double ss_fit = bench([&] { scaler.fit(X); }, runs);
    double ss_trans = bench([&] {
        Matrix Z = scaler.transform(X);
        static_cast<void>(Z);
    }, runs);

    // LinearRegression
    Skigen::LinearRegression<Scalar> ols;
    double lr_fit = bench([&] { ols.fit(X, y); }, runs);
    double lr_pred = bench([&] {
        Vector p = ols.predict(X);
        static_cast<void>(p);
    }, runs);

    // Ridge
    Skigen::Ridge<Scalar> ridge(static_cast<Scalar>(1.0));
    double rr_fit = bench([&] { ridge.fit(X, y); }, runs);

    // PCA
    Skigen::PCA<Scalar> pca(std::min(10, static_cast<int>(cols)));
    double pca_fit = bench([&] { pca.fit(X); }, runs / 5);  // SVD is expensive

    std::cout << "  " << label << ":\n";
    std::cout << "    StandardScaler  fit: " << std::setw(8) << ss_fit
              << " us  transform: " << std::setw(8) << ss_trans << " us"
              << "  (" << ss_trans * 1000.0 / elements << " ns/elem)\n";
    std::cout << "    LinearRegression fit:" << std::setw(8) << lr_fit
              << " us  predict:   " << std::setw(8) << lr_pred << " us\n";
    std::cout << "    Ridge            fit:" << std::setw(8) << rr_fit << " us\n";
    std::cout << "    PCA              fit:" << std::setw(8) << pca_fit << " us\n";

    return {ss_trans, lr_fit, rr_fit};
}

int main() {
    constexpr int rows = 10000;
    constexpr int cols = 100;
    constexpr int runs = 50;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "================================================================\n";
    std::cout << "  float vs double Benchmark (" << rows << " x " << cols
              << ", " << runs << " runs)\n";
    std::cout << "  sizeof(float)=" << sizeof(float)
              << "  sizeof(double)=" << sizeof(double) << "\n";
    std::cout << "================================================================\n\n";

    auto dbl = run_benchmark<double>("double", rows, cols, runs);
    std::cout << "\n";
    auto flt = run_benchmark<float>("float", rows, cols, runs);

    std::cout << "\n  Speedup (double/float):\n";
    std::cout << "    StandardScaler transform: "
              << std::setprecision(2) << dbl.ss_trans / flt.ss_trans << "x\n";
    std::cout << "    LinearRegression fit:     "
              << dbl.lr_fit / flt.lr_fit << "x\n";
    std::cout << "    Ridge fit:                "
              << dbl.rr_fit / flt.rr_fit << "x\n";

    std::cout << "\n================================================================\n";
    std::cout << "  Expected: ~1.5-2x speedup for memory-bound operations\n";
    std::cout << "  (8 floats vs 4 doubles per 256-bit SIMD register)\n";
    std::cout << "================================================================\n";

    return 0;
}
