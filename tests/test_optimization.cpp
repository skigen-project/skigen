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

void test_quadratic_converges() {
    Eigen::VectorXd x0(2);
    x0 << -3.0, 4.0;
    Skigen::NelderMead<double> nm(500, 1e-9, 1e-12, 0.2);
    auto result = nm.minimize(x0, [](const Eigen::VectorXd& x) {
        return std::pow(x(0) - 1.5, 2) + std::pow(x(1) + 2.0, 2);
    });
    ASSERT_TRUE(result.success);
    ASSERT_NEAR(result.x(0), 1.5, 1e-5);
    ASSERT_NEAR(result.x(1), -2.0, 1e-5);
    ASSERT_TRUE(result.fun < 1e-10);
}

void test_rosenbrock_converges() {
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;
    Skigen::NelderMead<double> nm(2000, 1e-8, 1e-10, 0.1);
    auto result = nm.minimize(x0, [](const Eigen::VectorXd& x) {
        const double a = 1.0 - x(0);
        const double b = x(1) - x(0) * x(0);
        return a * a + 100.0 * b * b;
    });
    ASSERT_TRUE(result.success);
    ASSERT_NEAR(result.x(0), 1.0, 1e-4);
    ASSERT_NEAR(result.x(1), 1.0, 1e-4);
}

void test_max_iter_reports_failure() {
    Eigen::VectorXd x0(2);
    x0 << 4.0, 4.0;
    Skigen::NelderMead<double> nm(1, 1e-12, 1e-12, 0.1);
    auto result = nm.minimize(x0, [](const Eigen::VectorXd& x) {
        return x.squaredNorm();
    });
    ASSERT_TRUE(!result.success);
    ASSERT_TRUE(result.nit == 1);
}

void test_invalid_inputs() {
    ASSERT_THROW(Skigen::NelderMead<double>(0), std::invalid_argument);
    Skigen::NelderMead<double> nm;
    Eigen::VectorXd empty;
    ASSERT_THROW(nm.minimize(empty, [](const Eigen::VectorXd&) { return 0.0; }), std::invalid_argument);
}

int main() {
    std::cout << "Running Optimization tests...\n";
    run_test("quadratic convergence", test_quadratic_converges);
    run_test("rosenbrock convergence", test_rosenbrock_converges);
    run_test("max_iter failure", test_max_iter_reports_failure);
    run_test("invalid inputs", test_invalid_inputs);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
