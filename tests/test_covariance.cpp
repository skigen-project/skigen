// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <random>

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;

struct TestFailure : std::exception {
    std::string msg;
    TestFailure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

#define ASSERT_TRUE(cond)                                                     \
    do {                                                                       \
        if (!(cond)) {                                                         \
            throw TestFailure(std::string(__FILE__) + ":" +                    \
                              std::to_string(__LINE__) + ": ASSERT_TRUE(" +    \
                              #cond + ") failed");                             \
        }                                                                      \
    } while (false)

template <typename Scalar>
void assert_near(Scalar a, Scalar b, Scalar tol, const char* file, int line) {
    if (std::abs(a - b) > tol) {
        std::ostringstream oss;
        oss << file << ":" << line << ": ASSERT_NEAR failed: " << a
            << " vs " << b << " (tol=" << tol << ")";
        throw TestFailure(oss.str());
    }
}
#define ASSERT_NEAR(a, b, tol) assert_near(a, b, tol, __FILE__, __LINE__)

template <typename MatA, typename MatB>
void assert_matrix_near(const MatA& A, const MatB& B, double tol,
                         const char* file, int line) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        std::ostringstream oss;
        oss << file << ":" << line << ": ASSERT_MATRIX_NEAR shape mismatch: ("
            << A.rows() << "," << A.cols() << ") vs ("
            << B.rows() << "," << B.cols() << ")";
        throw TestFailure(oss.str());
    }
    double err = (A - B).template cast<double>().norm();
    if (err > tol) {
        std::ostringstream oss;
        oss << file << ":" << line << ": ASSERT_MATRIX_NEAR failed: norm="
            << err << " > tol=" << tol;
        throw TestFailure(oss.str());
    }
}
#define ASSERT_MATRIX_NEAR(A, B, tol) assert_matrix_near(A, B, tol, __FILE__, __LINE__)

static void run_test(const std::string& name, std::function<void()> fn) {
    try {
        fn();
        ++g_passed;
        std::cout << "  PASS  " << name << "\n";
    } catch (const TestFailure& e) {
        ++g_failed;
        std::cout << "  FAIL  " << name << "\n        " << e.what() << "\n";
    } catch (const std::exception& e) {
        ++g_failed;
        std::cout << "  FAIL  " << name << "\n        exception: " << e.what() << "\n";
    }
}

// ===================================================================
// Helper: generate random normal data
// ===================================================================

Eigen::MatrixXd random_data(int n, int p, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd X(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            X(i, j) = dist(gen);
    return X;
}

// ===================================================================
// EmpiricalCovariance Tests
// ===================================================================

void test_empirical_basic() {
    Eigen::MatrixXd X = random_data(100, 3);
    Skigen::EmpiricalCovariance<double> ec;
    ec.fit(X);

    ASSERT_TRUE(ec.is_fitted());
    auto cov = ec.covariance();
    ASSERT_TRUE(cov.rows() == 3 && cov.cols() == 3);

    // Covariance should be symmetric
    ASSERT_MATRIX_NEAR(cov, cov.transpose(), 1e-12);

    // Diagonal should be positive
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(cov(i, i) > 0);
    }
}

void test_empirical_identity() {
    // For standard normal data, cov ≈ I as n → ∞
    Eigen::MatrixXd X = random_data(10000, 2);
    Skigen::EmpiricalCovariance<double> ec;
    ec.fit(X);

    auto cov = ec.covariance();
    ASSERT_MATRIX_NEAR(cov, Eigen::MatrixXd::Identity(2, 2), 0.1);
}

// ===================================================================
// LedoitWolf Tests
// ===================================================================

void test_lw_basic() {
    Eigen::MatrixXd X = random_data(50, 5);
    Skigen::LedoitWolf<double> lw;
    lw.fit(X);

    ASSERT_TRUE(lw.is_fitted());
    ASSERT_TRUE(lw.shrinkage() >= 0.0 && lw.shrinkage() <= 1.0);

    auto cov = lw.covariance();
    ASSERT_TRUE(cov.rows() == 5 && cov.cols() == 5);
    ASSERT_MATRIX_NEAR(cov, cov.transpose(), 1e-12);
}

void test_lw_shrinkage_bounds() {
    // With very few samples and many features, shrinkage should be high
    Eigen::MatrixXd X = random_data(10, 50);
    Skigen::LedoitWolf<double> lw;
    lw.fit(X);

    ASSERT_TRUE(lw.shrinkage() > 0.5);
}

void test_lw_large_n_low_shrinkage() {
    // With correlated data (true cov ≠ mu*I) and many samples,
    // shrinkage should be low because the sample covariance is accurate.
    std::mt19937 gen(123);
    std::normal_distribution<double> dist(0.0, 1.0);
    const int n = 5000, p = 3;
    Eigen::MatrixXd Z(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            Z(i, j) = dist(gen);

    // Introduce correlation: X = Z * L^T with non-trivial L
    Eigen::MatrixXd L(p, p);
    L << 2.0, 0.0, 0.0,
         0.5, 1.5, 0.0,
         0.3, 0.7, 1.0;
    Eigen::MatrixXd X = Z * L.transpose();

    Skigen::LedoitWolf<double> lw;
    lw.fit(X);

    ASSERT_TRUE(lw.shrinkage() < 0.05);
}

void test_lw_not_fitted() {
    Skigen::LedoitWolf<double> lw;
    bool threw = false;
    try { static_cast<void>(lw.covariance()); } catch (const std::runtime_error&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ===================================================================
// OAS Tests
// ===================================================================

void test_oas_basic() {
    Eigen::MatrixXd X = random_data(50, 5);
    Skigen::OAS<double> oas;
    oas.fit(X);

    ASSERT_TRUE(oas.is_fitted());
    ASSERT_TRUE(oas.shrinkage() >= 0.0 && oas.shrinkage() <= 1.0);

    auto cov = oas.covariance();
    ASSERT_TRUE(cov.rows() == 5 && cov.cols() == 5);
    ASSERT_MATRIX_NEAR(cov, cov.transpose(), 1e-12);
}

void test_oas_positive_definite() {
    Eigen::MatrixXd X = random_data(20, 10);
    Skigen::OAS<double> oas;
    oas.fit(X);

    auto cov = oas.covariance();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    // All eigenvalues should be positive (shrinkage ensures this)
    ASSERT_TRUE(solver.eigenvalues().minCoeff() > 0);
}

void test_oas_not_fitted() {
    Skigen::OAS<double> oas;
    bool threw = false;
    try { static_cast<void>(oas.shrinkage()); } catch (const std::runtime_error&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ===================================================================

int main() {
    std::cout << "=== EmpiricalCovariance Tests ===\n";
    run_test("empirical_basic", test_empirical_basic);
    run_test("empirical_identity", test_empirical_identity);

    std::cout << "\n=== LedoitWolf Tests ===\n";
    run_test("lw_basic", test_lw_basic);
    run_test("lw_shrinkage_bounds", test_lw_shrinkage_bounds);
    run_test("lw_large_n_low_shrinkage", test_lw_large_n_low_shrinkage);
    run_test("lw_not_fitted", test_lw_not_fitted);

    std::cout << "\n=== OAS Tests ===\n";
    run_test("oas_basic", test_oas_basic);
    run_test("oas_positive_definite", test_oas_positive_definite);
    run_test("oas_not_fitted", test_oas_not_fitted);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
