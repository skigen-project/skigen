// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/Dense>

#include <Eigen/Core>
#include <functional>
#include <iostream>
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

void test_binary_separable() {
    Eigen::MatrixXd X(6, 2);
    X << -2.0, -1.0,
         -1.5, -1.2,
         -1.2, -1.8,
          1.0,  1.2,
          1.5,  1.7,
          2.0,  1.3;
    Eigen::VectorXi y(6);
    y << 2, 2, 2, 9, 9, 9;

    Skigen::Perceptron<double> clf(/*max_iter=*/50, /*tol=*/0.0, /*eta0=*/1.0, /*random_state=*/0);
    clf.fit(X, y);
    Eigen::VectorXi pred = clf.predict(X);
    for (Eigen::Index i = 0; i < y.size(); ++i) ASSERT_TRUE(pred(i) == y(i));
    ASSERT_TRUE(clf.coef().rows() == 1);
}

void test_multiclass_ovr() {
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

    Skigen::Perceptron<double> clf(/*max_iter=*/80, /*tol=*/0.0, /*eta0=*/1.0, /*random_state=*/0);
    clf.fit(X, y);
    Eigen::VectorXi pred = clf.predict(X);
    int correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i) if (pred(i) == y(i)) ++correct;
    ASSERT_TRUE(correct == y.size());
    ASSERT_TRUE(clf.coef().rows() == 3);
}

void test_partial_fit() {
    Eigen::MatrixXd X(4, 2);
    X << -2.0, -1.0,
         -1.5, -1.2,
          1.0,  1.2,
          1.5,  1.7;
    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;
    Eigen::VectorXi classes(2);
    classes << 0, 1;

    Skigen::Perceptron<double> clf(/*max_iter=*/1, /*tol=*/0.0, /*eta0=*/1.0, /*random_state=*/0);
    clf.partial_fit(X, y, classes);
    Eigen::VectorXi empty;
    clf.partial_fit(X, y, empty);
    ASSERT_TRUE(clf.predict(X).size() == y.size());
}

void test_errors() {
    Skigen::Perceptron<double> clf;
    Eigen::MatrixXd X(2, 2);
    X << 1.0, 2.0,
         3.0, 4.0;
    Eigen::VectorXi y(2);
    y << 0, 1;
    ASSERT_THROW(clf.predict(X), std::runtime_error);
    Eigen::VectorXi empty;
    ASSERT_THROW(clf.partial_fit(X, y, empty), std::invalid_argument);
}

int main() {
    std::cout << "Running Perceptron tests...\n";
    run_test("binary separable", test_binary_separable);
    run_test("multiclass OvR", test_multiclass_ovr);
    run_test("partial_fit", test_partial_fit);
    run_test("errors", test_errors);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
