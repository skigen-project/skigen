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
// DecisionTreeClassifier Tests
// ===================================================================

void test_dtc_basic() {
    // Simple XOR-like problem
    Eigen::MatrixXd X(4, 2);
    X << 0, 0,
         0, 1,
         1, 0,
         1, 1;
    Eigen::VectorXi y(4);
    y << 0, 1, 1, 0;

    Skigen::DecisionTreeClassifier<double> dt;
    dt.fit(X, y);

    // Should perfectly memorize training data
    auto preds = dt.predict(X);
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        ASSERT_TRUE(preds(i) == y(i));
    }
}

void test_dtc_score() {
    Eigen::MatrixXd X(6, 2);
    X << 0, 0, 0, 1, 1, 0, 10, 10, 10, 11, 11, 10;
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    Skigen::DecisionTreeClassifier<double> dt;
    dt.fit(X, y);

    double acc = dt.score(X, y);
    ASSERT_NEAR(acc, 1.0, 1e-10);
}

void test_dtc_max_depth() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0, 0, 1, 1, 0, 1, 1;
    Eigen::VectorXi y(4);
    y << 0, 1, 1, 0;

    // Depth 1 should not be able to perfectly learn XOR
    Skigen::DecisionTreeClassifier<double> dt(1);
    dt.fit(X, y);

    // Not checking accuracy exactly, just that it doesn't crash
    auto preds = dt.predict(X);
    ASSERT_TRUE(preds.size() == 4);
}

void test_dtc_not_fitted() {
    Skigen::DecisionTreeClassifier<double> dt;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(dt.predict(X), std::runtime_error);
}

// ===================================================================
// DecisionTreeRegressor Tests
// ===================================================================

void test_dtr_basic() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::DecisionTreeRegressor<double> dt;
    dt.fit(X, y);

    // Should memorize training data
    auto preds = dt.predict(X);
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        ASSERT_NEAR(preds(i), y(i), 1e-10);
    }
}

void test_dtr_score() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::DecisionTreeRegressor<double> dt;
    dt.fit(X, y);

    double r2 = dt.score(X, y);
    ASSERT_NEAR(r2, 1.0, 1e-10);
}

void test_dtr_not_fitted() {
    Skigen::DecisionTreeRegressor<double> dt;
    Eigen::MatrixXd X(2, 1);
    X << 1, 2;
    ASSERT_THROW(dt.predict(X), std::runtime_error);
}

// ===================================================================

int main() {
    std::cout << "=== DecisionTreeClassifier Tests ===\n";
    run_test("dtc_basic", test_dtc_basic);
    run_test("dtc_score", test_dtc_score);
    run_test("dtc_max_depth", test_dtc_max_depth);
    run_test("dtc_not_fitted", test_dtc_not_fitted);

    std::cout << "\n=== DecisionTreeRegressor Tests ===\n";
    run_test("dtr_basic", test_dtr_basic);
    run_test("dtr_score", test_dtr_score);
    run_test("dtr_not_fitted", test_dtr_not_fitted);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
