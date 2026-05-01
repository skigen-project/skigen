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

template <typename VecA, typename VecB, typename Scalar>
void assert_vector_near(const VecA& A, const VecB& B, Scalar tol,
                        const char* file, int line) {
    if (A.size() != B.size()) {
        throw TestFailure(std::string(file) + ":" + std::to_string(line) +
                          ": Size mismatch");
    }
    for (Eigen::Index i = 0; i < A.size(); ++i)
        assert_near(A(i), B(i), tol, file, line);
}
#define ASSERT_VECTOR_NEAR(A, B, tol) assert_vector_near(A, B, tol, __FILE__, __LINE__)

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
// LinearRegression Tests
// ===================================================================

void test_lr_perfect_fit() {
    // y = 2*x1 + 3*x2 + 1
    Eigen::MatrixXd X(4, 2);
    X << 1, 0,
         0, 1,
         1, 1,
         2, 1;
    Eigen::VectorXd y(4);
    y << 3, 4, 6, 8;

    Skigen::LinearRegression reg;
    reg.fit(X, y);

    ASSERT_NEAR(reg.coef()(0), 2.0, 1e-10);
    ASSERT_NEAR(reg.coef()(1), 3.0, 1e-10);
    ASSERT_NEAR(reg.intercept(), 1.0, 1e-10);
}

void test_lr_predict() {
    Eigen::MatrixXd X(3, 1);
    X << 1, 2, 3;
    Eigen::VectorXd y(3);
    y << 2, 4, 6;

    Skigen::LinearRegression reg;
    reg.fit(X, y);

    Eigen::MatrixXd X_test(2, 1);
    X_test << 4, 5;
    Eigen::VectorXd y_pred = reg.predict(X_test);

    ASSERT_NEAR(y_pred(0), 8.0, 1e-10);
    ASSERT_NEAR(y_pred(1), 10.0, 1e-10);
}

void test_lr_score() {
    // Perfect linear relationship → R² = 1
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::LinearRegression reg;
    reg.fit(X, y);
    double r2 = reg.score(X, y);

    ASSERT_NEAR(r2, 1.0, 1e-10);
}

void test_lr_no_intercept() {
    // y = 3*x (through origin)
    Eigen::MatrixXd X(3, 1);
    X << 1, 2, 3;
    Eigen::VectorXd y(3);
    y << 3, 6, 9;

    Skigen::LinearRegression<double> reg(false);
    reg.fit(X, y);

    ASSERT_NEAR(reg.coef()(0), 3.0, 1e-10);
    ASSERT_NEAR(reg.intercept(), 0.0, 1e-15);
}

void test_lr_not_fitted() {
    Skigen::LinearRegression reg;
    Eigen::MatrixXd X(2, 1);
    X << 1, 2;
    ASSERT_THROW(reg.predict(X), std::runtime_error);
}

void test_lr_feature_mismatch() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2, 3, 4, 5, 6;
    Eigen::VectorXd y(3);
    y << 1, 2, 3;

    Skigen::LinearRegression reg;
    reg.fit(X, y);

    Eigen::MatrixXd X_bad(1, 3);
    X_bad << 1, 2, 3;
    ASSERT_THROW(reg.predict(X_bad), std::invalid_argument);
}

void test_lr_concept() {
    static_assert(Skigen::PredictorLike<Skigen::LinearRegression<double>>);
    ASSERT_TRUE(true);
}

void test_lr_float() {
    Eigen::MatrixXf X(3, 1);
    X << 1, 2, 3;
    Eigen::VectorXf y(3);
    y << 2, 4, 6;

    Skigen::LinearRegression<float> reg;
    reg.fit(X, y);
    ASSERT_NEAR(reg.coef()(0), 2.0f, 1e-5f);
}

// ===================================================================
// Ridge Tests
// ===================================================================

void test_ridge_basic() {
    // With strong regularization, coefficients shrink toward zero
    Eigen::MatrixXd X(4, 2);
    X << 1, 0,
         0, 1,
         1, 1,
         2, 1;
    Eigen::VectorXd y(4);
    y << 3, 4, 6, 8;

    Skigen::Ridge<double> ridge_weak(0.001);
    ridge_weak.fit(X, y);

    Skigen::Ridge<double> ridge_strong(1000.0);
    ridge_strong.fit(X, y);

    // Strong regularization → smaller coefficients
    ASSERT_TRUE(ridge_strong.coef().norm() < ridge_weak.coef().norm());
}

void test_ridge_converges_to_ols() {
    // With alpha → 0, Ridge should match OLS
    Eigen::MatrixXd X(4, 2);
    X << 1, 0,
         0, 1,
         1, 1,
         2, 1;
    Eigen::VectorXd y(4);
    y << 3, 4, 6, 8;

    Skigen::LinearRegression ols;
    ols.fit(X, y);

    Skigen::Ridge<double> ridge(1e-10);
    ridge.fit(X, y);

    ASSERT_NEAR(ridge.coef()(0), ols.coef()(0), 1e-6);
    ASSERT_NEAR(ridge.coef()(1), ols.coef()(1), 1e-6);
    ASSERT_NEAR(ridge.intercept(), ols.intercept(), 1e-6);
}

void test_ridge_predict() {
    Eigen::MatrixXd X(3, 1);
    X << 1, 2, 3;
    Eigen::VectorXd y(3);
    y << 2, 4, 6;

    Skigen::Ridge<double> ridge(0.01);
    ridge.fit(X, y);

    Eigen::MatrixXd X_test(1, 1);
    X_test << 4;
    Eigen::VectorXd y_pred = ridge.predict(X_test);
    // With small alpha, should be close to OLS result of 8
    ASSERT_NEAR(y_pred(0), 8.0, 0.1);
}

void test_ridge_score() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::Ridge<double> ridge(0.01);
    ridge.fit(X, y);
    double r2 = ridge.score(X, y);
    // Near-perfect fit with small alpha
    ASSERT_TRUE(r2 > 0.99);
}

void test_ridge_no_intercept() {
    Eigen::MatrixXd X(3, 1);
    X << 1, 2, 3;
    Eigen::VectorXd y(3);
    y << 3, 6, 9;

    Skigen::Ridge<double> ridge(1.0, false);
    ridge.fit(X, y);

    ASSERT_NEAR(ridge.intercept(), 0.0, 1e-15);
}

void test_ridge_concept() {
    static_assert(Skigen::PredictorLike<Skigen::Ridge<double>>);
    ASSERT_TRUE(true);
}

// ===================================================================
// Main
// ===================================================================

int main() {
    std::cout << "=== LinearRegression Tests ===\n";
    run_test("lr_perfect_fit",       test_lr_perfect_fit);
    run_test("lr_predict",           test_lr_predict);
    run_test("lr_score",             test_lr_score);
    run_test("lr_no_intercept",      test_lr_no_intercept);
    run_test("lr_not_fitted",        test_lr_not_fitted);
    run_test("lr_feature_mismatch",  test_lr_feature_mismatch);
    run_test("lr_concept",           test_lr_concept);
    run_test("lr_float",             test_lr_float);

    std::cout << "\n=== Ridge Tests ===\n";
    run_test("ridge_basic",              test_ridge_basic);
    run_test("ridge_converges_to_ols",   test_ridge_converges_to_ols);
    run_test("ridge_predict",            test_ridge_predict);
    run_test("ridge_score",              test_ridge_score);
    run_test("ridge_no_intercept",       test_ridge_no_intercept);
    run_test("ridge_concept",            test_ridge_concept);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
