// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/Dense>

#include <Eigen/Core>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

static int g_passed = 0;
static int g_failed = 0;

struct TestFailure : std::exception {
    std::string msg;
    explicit TestFailure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

#define ASSERT_TRUE(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            throw TestFailure(std::string(__FILE__) + ":" +                  \
                              std::to_string(__LINE__) + ": ASSERT_TRUE(" +  \
                              #cond + ") failed");                           \
        }                                                                     \
    } while (false)

#define ASSERT_THROW(expr, ExType)                                             \
    do {                                                                       \
        bool caught = false;                                                   \
        try {                                                                  \
            static_cast<void>(expr);                                           \
        } catch (const ExType&) {                                              \
            caught = true;                                                     \
        }                                                                      \
        if (!caught) {                                                         \
            throw TestFailure(std::string(__FILE__) + ":" +                   \
                              std::to_string(__LINE__) +                       \
                              ": Expected exception " #ExType);                \
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

template <typename VecA, typename VecB, typename Scalar>
void assert_vector_near(const VecA& a, const VecB& b, Scalar tol,
                        const char* file, int line) {
    if (a.size() != b.size()) {
        throw TestFailure(std::string(file) + ":" + std::to_string(line) +
                          ": vector size mismatch");
    }
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        assert_near(a(i), b(i), tol, file, line);
    }
}
#define ASSERT_VECTOR_NEAR(a, b, tol) assert_vector_near(a, b, tol, __FILE__, __LINE__)

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

void test_linear_matches_ridge_no_intercept() {
    Eigen::MatrixXd X(5, 2);
    X << 1.0, 0.0,
         0.0, 1.0,
         2.0, 1.0,
         1.0, 2.0,
         3.0, 1.0;
    Eigen::VectorXd y(5);
    y << 1.0, -1.0, 2.0, 0.5, 3.0;

    Skigen::KernelRidge<double> krr(1.0);
    krr.fit(X, y);
    Skigen::Ridge<double> ridge(1.0, /*fit_intercept=*/false);
    ridge.fit(X, y);

    ASSERT_VECTOR_NEAR(krr.predict(X), ridge.predict(X), 1e-10);
    ASSERT_TRUE(krr.dual_coef().size() == X.rows());
    ASSERT_NEAR(krr.gamma_effective(), 0.5, 1e-12);
}

void test_rbf_kernel_interpolates_training_reasonably() {
    Eigen::MatrixXd X(4, 1);
    X << -1.0, 0.0, 1.0, 2.0;
    Eigen::VectorXd y(4);
    y << 1.0, 0.0, 1.0, 4.0;

    Skigen::KernelRidge<double> krr(
        1e-6, Skigen::KernelRidge<double>::Kernel::RBF, 0.8);
    krr.fit(X, y);
    ASSERT_VECTOR_NEAR(krr.predict(X), y, 1e-4);
}

void test_poly_and_sigmoid_paths_are_finite() {
    Eigen::MatrixXd X(3, 2);
    X << 1.0, 2.0,
         2.0, 0.5,
        -1.0, 1.0;
    Eigen::VectorXd y(3);
    y << 1.0, 2.0, -1.0;

    Skigen::KernelRidge<double> poly(
        0.5, Skigen::KernelRidge<double>::Kernel::Poly, 0.2, 3, 1.0);
    Skigen::KernelRidge<double> sigmoid(
        0.5, Skigen::KernelRidge<double>::Kernel::Sigmoid, 0.1, 3, 0.2);
    poly.fit(X, y);
    sigmoid.fit(X, y);

    ASSERT_TRUE(poly.predict(X).allFinite());
    ASSERT_TRUE(sigmoid.predict(X).allFinite());
}

void test_invalid_and_unfitted_errors() {
    Eigen::MatrixXd X(2, 2);
    X << 1.0, 2.0,
         3.0, 4.0;
    Eigen::VectorXd y(2);
    y << 1.0, 2.0;

    Skigen::KernelRidge<double> krr;
    ASSERT_THROW(krr.predict(X), std::runtime_error);
    Skigen::KernelRidge<double> bad(-1.0);
    ASSERT_THROW(bad.fit(X, y), std::invalid_argument);

    krr.fit(X, y);
    Eigen::MatrixXd wrong(1, 3);
    wrong << 1.0, 2.0, 3.0;
    ASSERT_THROW(krr.predict(wrong), std::invalid_argument);
}

int main() {
    std::cout << "Running KernelRidge tests...\n";
    run_test("linear matches Ridge no intercept", test_linear_matches_ridge_no_intercept);
    run_test("rbf training interpolation", test_rbf_kernel_interpolates_training_reasonably);
    run_test("poly and sigmoid finite", test_poly_and_sigmoid_paths_are_finite);
    run_test("invalid and unfitted errors", test_invalid_and_unfitted_errors);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
