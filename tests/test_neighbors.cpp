// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>

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

#define ASSERT_THROW(expr, ExType)                                            \
    do {                                                                       \
        bool caught = false;                                                   \
        try { static_cast<void>(expr); } catch (const ExType&) { caught = true; } \
        if (!caught) {                                                         \
            throw TestFailure(std::string(__FILE__) + ":" +                    \
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

// ===================================================================
// KNeighborsClassifier Tests
// ===================================================================

void test_knn_classifier_basic() {
    Eigen::MatrixXd X(6, 2);
    X << 0, 0,
         0, 1,
         1, 0,
         10, 10,
         10, 11,
         11, 10;
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    Skigen::KNeighborsClassifier<double> knn(3);
    knn.fit(X, y);

    Eigen::MatrixXd X_test(2, 2);
    X_test << 0.5, 0.5,
              10.5, 10.5;

    auto preds = knn.predict(X_test);
    ASSERT_TRUE(preds(0) == 0);
    ASSERT_TRUE(preds(1) == 1);
}

void test_knn_classifier_score() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0, 0, 1, 10, 10, 10, 11;
    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;

    Skigen::KNeighborsClassifier<double> knn(1);
    knn.fit(X, y);

    // Score on training data should be perfect with k=1
    double acc = knn.score(X, y);
    ASSERT_NEAR(acc, 1.0, 1e-10);
}

void test_knn_classifier_not_fitted() {
    Skigen::KNeighborsClassifier<double> knn;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(knn.predict(X), std::runtime_error);
}

// ===================================================================
// KNeighborsRegressor Tests
// ===================================================================

void test_knn_regressor_basic() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::KNeighborsRegressor<double> knn(2);
    knn.fit(X, y);

    Eigen::MatrixXd X_test(1, 1);
    X_test << 2.5;
    auto pred = knn.predict(X_test);
    // Average of two nearest: y(1)=4, y(2)=6 → 5
    ASSERT_NEAR(pred(0), 5.0, 1e-10);
}

void test_knn_regressor_score() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::KNeighborsRegressor<double> knn(1);
    knn.fit(X, y);

    double r2 = knn.score(X, y);
    ASSERT_NEAR(r2, 1.0, 1e-10);
}

void test_knn_regressor_not_fitted() {
    Skigen::KNeighborsRegressor<double> knn;
    Eigen::MatrixXd X(2, 1);
    X << 1, 2;
    ASSERT_THROW(knn.predict(X), std::runtime_error);
}

// ===================================================================

int main() {
    std::cout << "=== KNeighborsClassifier Tests ===\n";
    run_test("knn_classifier_basic", test_knn_classifier_basic);
    run_test("knn_classifier_score", test_knn_classifier_score);
    run_test("knn_classifier_not_fitted", test_knn_classifier_not_fitted);

    std::cout << "\n=== KNeighborsRegressor Tests ===\n";
    run_test("knn_regressor_basic", test_knn_regressor_basic);
    run_test("knn_regressor_score", test_knn_regressor_score);
    run_test("knn_regressor_not_fitted", test_knn_regressor_not_fitted);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
