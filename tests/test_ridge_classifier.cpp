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

void test_binary_original_labels() {
    Eigen::MatrixXd X(6, 2);
    X << -2.0, -1.0,
         -1.5, -1.2,
         -1.2, -1.8,
          1.0,  1.2,
          1.5,  1.7,
          2.0,  1.3;
    Eigen::VectorXi y(6);
    y << 2, 2, 2, 9, 9, 9;

    Skigen::RidgeClassifier<double> clf;
    clf.fit(X, y);
    Eigen::VectorXi pred = clf.predict(X);
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        ASSERT_TRUE(pred(i) == y(i));
    }
    ASSERT_TRUE(clf.classes()(0) == 2);
    ASSERT_TRUE(clf.classes()(1) == 9);
    ASSERT_TRUE(clf.coef().rows() == 1);
    ASSERT_TRUE(clf.coef().cols() == X.cols());
    ASSERT_TRUE(clf.intercept().size() == 1);
}

void test_multiclass_argmax() {
    Eigen::MatrixXd X(9, 2);
    X << -2.0,  0.0,
         -1.5,  0.2,
         -1.8, -0.2,
          2.0,  0.0,
          1.6,  0.3,
          1.8, -0.4,
          0.0,  2.0,
          0.3,  1.6,
         -0.2,  1.8;
    Eigen::VectorXi y(9);
    y << 0, 0, 0, 1, 1, 1, 2, 2, 2;

    Skigen::RidgeClassifier<double> clf(0.1);
    clf.fit(X, y);
    Eigen::VectorXi pred = clf.predict(X);
    int correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (pred(i) == y(i)) ++correct;
    }
    ASSERT_TRUE(correct == y.size());
    ASSERT_TRUE(clf.coef().rows() == 3);
    ASSERT_TRUE(clf.intercept().size() == 3);
}

void test_fit_intercept_false() {
    Eigen::MatrixXd X(4, 1);
    X << -2.0, -1.0, 1.0, 2.0;
    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;

    Skigen::RidgeClassifier<double> clf(1.0, /*fit_intercept=*/false);
    clf.fit(X, y);
    ASSERT_NEAR(clf.intercept()(0), 0.0, 1e-12);
}

void test_errors() {
    Eigen::MatrixXd X(2, 2);
    X << 1.0, 2.0,
         3.0, 4.0;
    Eigen::VectorXi one_class(2);
    one_class << 1, 1;
    Skigen::RidgeClassifier<double> clf;
    ASSERT_THROW(clf.fit(X, one_class), std::invalid_argument);
    Skigen::RidgeClassifier<double> bad(-1.0);
    Eigen::VectorXi y(2);
    y << 0, 1;
    ASSERT_THROW(bad.fit(X, y), std::invalid_argument);
    ASSERT_THROW(clf.predict(X), std::runtime_error);
}

int main() {
    std::cout << "Running RidgeClassifier tests...\n";
    run_test("binary original labels", test_binary_original_labels);
    run_test("multiclass argmax", test_multiclass_argmax);
    run_test("fit_intercept false", test_fit_intercept_false);
    run_test("errors", test_errors);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
