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

void test_lr_sparse_matches_dense_with_intercept() {
    Eigen::MatrixXd Xd(20, 4);
    Xd.setZero();
    Xd(0, 0) = 1; Xd(0, 1) = 2;
    Xd(1, 0) = 3; Xd(1, 2) = 1;
    Xd(2, 1) = 4; Xd(2, 3) = 2;
    Xd(3, 0) = 2; Xd(3, 3) = 3;
    Xd(4, 2) = 5;
    Xd(5, 1) = 1; Xd(5, 2) = 2;
    Xd(6, 0) = 4; Xd(6, 1) = 1;
    Xd(7, 3) = 6;
    Xd(8, 0) = 1; Xd(8, 2) = 1;
    Xd(9, 1) = 3; Xd(9, 3) = 1;
    Xd(10, 0) = 5;
    Xd(11, 2) = 4;
    Xd(12, 1) = 2;
    Xd(13, 0) = 1; Xd(13, 1) = 1; Xd(13, 3) = 2;
    Xd(14, 2) = 3;
    Xd(15, 0) = 2; Xd(15, 3) = 1;
    Xd(16, 1) = 5;
    Xd(17, 2) = 2; Xd(17, 3) = 4;
    Xd(18, 0) = 3;
    Xd(19, 1) = 1; Xd(19, 2) = 2; Xd(19, 3) = 1;

    Eigen::VectorXd w_true(4);
    w_true << 1.0, -1.5, 0.5, 0.8;
    const double b_true = 2.5;
    Eigen::VectorXd y = (Xd * w_true).array() + b_true;

    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::LinearRegression<double> ld(true);
    Skigen::LinearRegression<double> ls(true);
    ld.fit(Xd, y);
    ls.fit(Xs, y);

    for (int j = 0; j < 4; ++j) {
        ASSERT_NEAR(ld.coef()(j), ls.coef()(j), 1e-9);
    }
    ASSERT_NEAR(ld.intercept(), ls.intercept(), 1e-9);
    ASSERT_NEAR(ld.intercept(), b_true, 1e-9);
}

void test_lr_sparse_no_intercept() {
    Eigen::MatrixXd Xd(10, 3);
    Xd.setZero();
    Xd(0, 0) = 1; Xd(1, 1) = 2; Xd(2, 2) = 3;
    Xd(3, 0) = 4; Xd(4, 1) = 1; Xd(5, 2) = 5;
    Xd(6, 0) = 2; Xd(7, 1) = 4; Xd(8, 2) = 1;
    Xd(9, 0) = 3; Xd(9, 1) = 2;

    Eigen::VectorXd w_true(3);
    w_true << 0.5, 1.0, -0.7;
    Eigen::VectorXd y = Xd * w_true;

    Eigen::SparseMatrix<double> Xs = Xd.sparseView();
    Skigen::LinearRegression<double> ld(false);
    Skigen::LinearRegression<double> ls(false);
    ld.fit(Xd, y);
    ls.fit(Xs, y);

    for (int j = 0; j < 3; ++j) {
        ASSERT_NEAR(ld.coef()(j), ls.coef()(j), 1e-9);
    }
    ASSERT_NEAR(ls.intercept(), 0.0, 1e-12);
}

void test_lr_sparse_empty_throws() {
    Eigen::SparseMatrix<double> X(0, 0);
    Eigen::VectorXd y(0);
    Skigen::LinearRegression<double> r;
    bool threw = false;
    try { r.fit(X, y); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

// -- Multi-target regression --------------------------------------------

void test_lr_multi_target_recovers_two_outputs() {
    // y1 = 2*x + 1, y2 = -3*x + 5  (single feature, two targets)
    constexpr int n = 50;
    Eigen::MatrixXd X(n, 1);
    Eigen::MatrixXd Y(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i);
        Y(i, 0) =  2.0 * X(i, 0) + 1.0;
        Y(i, 1) = -3.0 * X(i, 0) + 5.0;
    }

    Skigen::LinearRegression<double> lr;
    lr.fit_multi(X, Y);

    ASSERT_TRUE(lr.n_targets() == 2);
    ASSERT_TRUE(lr.coef_matrix().rows() == 2);
    ASSERT_TRUE(lr.coef_matrix().cols() == 1);
    ASSERT_NEAR(lr.coef_matrix()(0, 0),  2.0, 1e-9);
    ASSERT_NEAR(lr.coef_matrix()(1, 0), -3.0, 1e-9);
    ASSERT_NEAR(lr.intercept_vector()(0),  1.0, 1e-9);
    ASSERT_NEAR(lr.intercept_vector()(1),  5.0, 1e-9);

    Eigen::MatrixXd Y_pred = lr.predict_multi(X);
    ASSERT_TRUE(Y_pred.rows() == n);
    ASSERT_TRUE(Y_pred.cols() == 2);
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(Y_pred(i, 0), Y(i, 0), 1e-9);
        ASSERT_NEAR(Y_pred(i, 1), Y(i, 1), 1e-9);
    }
}

void test_lr_multi_target_no_intercept() {
    // y1 = 2x + 3z, y2 = x − z   (two features, two targets, no intercept)
    constexpr int n = 30;
    Eigen::MatrixXd X(n, 2);
    Eigen::MatrixXd Y(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i);
        X(i, 1) = 0.5 * static_cast<double>(i) - 1.0;
        Y(i, 0) = 2.0 * X(i, 0) + 3.0 * X(i, 1);
        Y(i, 1) =       X(i, 0) -       X(i, 1);
    }

    Skigen::LinearRegression<double> lr(false);
    lr.fit_multi(X, Y);

    ASSERT_NEAR(lr.coef_matrix()(0, 0),  2.0, 1e-9);
    ASSERT_NEAR(lr.coef_matrix()(0, 1),  3.0, 1e-9);
    ASSERT_NEAR(lr.coef_matrix()(1, 0),  1.0, 1e-9);
    ASSERT_NEAR(lr.coef_matrix()(1, 1), -1.0, 1e-9);
    ASSERT_NEAR(lr.intercept_vector()(0), 0.0, 1e-12);
    ASSERT_NEAR(lr.intercept_vector()(1), 0.0, 1e-12);
}

void test_lr_multi_target_single_target_API_still_works() {
    // After fit_multi(...) the single-target accessors mirror the first
    // target column, so existing callers see consistent state.
    Eigen::MatrixXd X(5, 1); X << 0, 1, 2, 3, 4;
    Eigen::MatrixXd Y(5, 2);
    Y.col(0) << 0.0, 2.0, 4.0, 6.0, 8.0;     // target 0: y = 2x
    Y.col(1) << 1.0, 0.0, -1.0, -2.0, -3.0;  // target 1: y = 1 − x

    Skigen::LinearRegression<double> lr;
    lr.fit_multi(X, Y);

    // coef() / intercept() should reflect target 0.
    ASSERT_NEAR(lr.coef()(0),     2.0, 1e-9);
    ASSERT_NEAR(lr.intercept(),   0.0, 1e-9);
}

void test_lr_single_target_then_coef_matrix_synthesised() {
    // After single-target fit, coef_matrix() / intercept_vector() must
    // synthesise a 1-row view consistent with the single-target state.
    Eigen::MatrixXd X(4, 2); X << 1, 2, 3, 4, 5, 6, 7, 8;
    Eigen::VectorXd y(4); y << 1, 2, 3, 4;

    Skigen::LinearRegression<double> lr;
    lr.fit(X, y);

    ASSERT_TRUE(lr.coef_matrix().rows() == 1);
    ASSERT_TRUE(lr.coef_matrix().cols() == 2);
    ASSERT_NEAR(lr.coef_matrix()(0, 0), lr.coef()(0), 1e-12);
    ASSERT_NEAR(lr.coef_matrix()(0, 1), lr.coef()(1), 1e-12);
    ASSERT_NEAR(lr.intercept_vector()(0), lr.intercept(), 1e-12);
    ASSERT_TRUE(lr.n_targets() == 1);
}

void test_lr_multi_target_dim_mismatch_throws() {
    Eigen::MatrixXd X(4, 2); X.setOnes();
    Eigen::MatrixXd Y(5, 2); Y.setOnes();      // n rows mismatch
    Skigen::LinearRegression<double> lr;
    bool threw = false;
    try { lr.fit_multi(X, Y); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
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

void test_ridge_sparse_matches_dense() {
    Eigen::MatrixXd Xd(15, 4);
    Xd <<  1, 2, 0, 0,
           0, 3, 1, 0,
           4, 0, 0, 5,
           0, 6, 7, 0,
           1, 0, 0, 8,
           0, 9, 0, 0,
           0, 0, 1, 1,
           5, 0, 0, 4,
           0, 7, 6, 0,
           2, 0, 0, 9,
           0, 1, 0, 0,
           3, 0, 0, 0,
           0, 0, 4, 5,
           0, 8, 0, 0,
           0, 0, 2, 1;
    Eigen::VectorXd y(15);
    y <<  3.0, 7.0, 18.0, 16.0, 17.0, 18.0, 4.0, 18.0, 19.0,
         20.0, 2.0, 6.0, 18.0, 16.0, 5.0;

    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::Ridge<double> rd(0.5, /*fit_intercept=*/true);
    Skigen::Ridge<double> rs(0.5, /*fit_intercept=*/true);
    rd.fit(Xd, y);
    rs.fit(Xs, y);

    for (int j = 0; j < Xd.cols(); ++j) {
        ASSERT_NEAR(rd.coef()(j), rs.coef()(j), 1e-10);
    }
    ASSERT_NEAR(rd.intercept(), rs.intercept(), 1e-10);

    // Prediction should agree (using dense X for both).
    Eigen::VectorXd pd = rd.predict(Xd);
    Eigen::VectorXd ps = rs.predict(Xd);
    for (int i = 0; i < Xd.rows(); ++i) {
        ASSERT_NEAR(pd(i), ps(i), 1e-10);
    }
}

void test_ridge_sparse_no_intercept() {
    // No-intercept path should also agree with dense.
    Eigen::MatrixXd Xd(8, 3);
    Xd << 1, 0, 2,
          0, 3, 0,
          0, 0, 1,
          4, 0, 0,
          0, 5, 0,
          1, 1, 1,
          0, 2, 0,
          3, 0, 4;
    Eigen::VectorXd y(8);
    y << 5, 9, 1, 8, 15, 6, 6, 14;

    Eigen::SparseMatrix<double> Xs = Xd.sparseView();
    Skigen::Ridge<double> rd(1.0, false);
    Skigen::Ridge<double> rs(1.0, false);
    rd.fit(Xd, y);
    rs.fit(Xs, y);

    for (int j = 0; j < Xd.cols(); ++j) {
        ASSERT_NEAR(rd.coef()(j), rs.coef()(j), 1e-10);
    }
    ASSERT_NEAR(rs.intercept(), 0.0, 1e-12);
}

void test_ridge_sparse_empty_throws() {
    Eigen::SparseMatrix<double> X(0, 0);
    Eigen::VectorXd y(0);
    Skigen::Ridge<double> r;
    bool threw = false;
    try { r.fit(X, y); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_ridge_multi_target_recovers_two_outputs() {
    // y1 = 2x + 3z, y2 = x − z   (small alpha so coefs come close to OLS)
    constexpr int n = 60;
    Eigen::MatrixXd X(n, 2);
    Eigen::MatrixXd Y(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        X(i, 1) = static_cast<double>(n - i) / 10.0;
        Y(i, 0) = 2.0 * X(i, 0) + 3.0 * X(i, 1);
        Y(i, 1) =       X(i, 0) -       X(i, 1);
    }

    Skigen::Ridge<double> r(/*alpha=*/1e-6, /*fit_intercept=*/false);
    r.fit_multi(X, Y);

    ASSERT_TRUE(r.n_targets() == 2);
    ASSERT_TRUE(r.coef_matrix().rows() == 2);
    ASSERT_TRUE(r.coef_matrix().cols() == 2);
    ASSERT_NEAR(r.coef_matrix()(0, 0),  2.0, 1e-4);
    ASSERT_NEAR(r.coef_matrix()(0, 1),  3.0, 1e-4);
    ASSERT_NEAR(r.coef_matrix()(1, 0),  1.0, 1e-4);
    ASSERT_NEAR(r.coef_matrix()(1, 1), -1.0, 1e-4);

    Eigen::MatrixXd Y_pred = r.predict_multi(X);
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(Y_pred(i, 0), Y(i, 0), 1e-3);
        ASSERT_NEAR(Y_pred(i, 1), Y(i, 1), 1e-3);
    }
}

void test_ridge_multi_target_with_intercept() {
    // y1 = 2x + 1, y2 = -3x + 5
    constexpr int n = 40;
    Eigen::MatrixXd X(n, 1);
    Eigen::MatrixXd Y(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 5.0;
        Y(i, 0) =  2.0 * X(i, 0) + 1.0;
        Y(i, 1) = -3.0 * X(i, 0) + 5.0;
    }

    Skigen::Ridge<double> r(1e-6, true);
    r.fit_multi(X, Y);

    ASSERT_NEAR(r.coef_matrix()(0, 0),  2.0, 1e-3);
    ASSERT_NEAR(r.coef_matrix()(1, 0), -3.0, 1e-3);
    ASSERT_NEAR(r.intercept_vector()(0),  1.0, 1e-3);
    ASSERT_NEAR(r.intercept_vector()(1),  5.0, 1e-3);
}

void test_ridge_single_target_then_coef_matrix_synthesised() {
    Eigen::MatrixXd X(4, 2); X << 1, 2, 3, 4, 5, 6, 7, 8;
    Eigen::VectorXd y(4); y << 1, 2, 3, 4;
    Skigen::Ridge<double> r(1e-3, true);
    r.fit(X, y);
    ASSERT_TRUE(r.coef_matrix().rows() == 1);
    ASSERT_TRUE(r.coef_matrix().cols() == 2);
    ASSERT_NEAR(r.coef_matrix()(0, 0), r.coef()(0), 1e-12);
    ASSERT_NEAR(r.coef_matrix()(0, 1), r.coef()(1), 1e-12);
    ASSERT_NEAR(r.intercept_vector()(0), r.intercept(), 1e-12);
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
    run_test("lr_sparse_matches_dense_with_intercept",
             test_lr_sparse_matches_dense_with_intercept);
    run_test("lr_sparse_no_intercept",  test_lr_sparse_no_intercept);
    run_test("lr_sparse_empty_throws",  test_lr_sparse_empty_throws);
    run_test("lr_multi_target_recovers_two_outputs",
             test_lr_multi_target_recovers_two_outputs);
    run_test("lr_multi_target_no_intercept",
             test_lr_multi_target_no_intercept);
    run_test("lr_multi_target_single_target_API_still_works",
             test_lr_multi_target_single_target_API_still_works);
    run_test("lr_single_target_then_coef_matrix_synthesised",
             test_lr_single_target_then_coef_matrix_synthesised);
    run_test("lr_multi_target_dim_mismatch_throws",
             test_lr_multi_target_dim_mismatch_throws);

    std::cout << "\n=== Ridge Tests ===\n";
    run_test("ridge_basic",              test_ridge_basic);
    run_test("ridge_converges_to_ols",   test_ridge_converges_to_ols);
    run_test("ridge_predict",            test_ridge_predict);
    run_test("ridge_score",              test_ridge_score);
    run_test("ridge_no_intercept",       test_ridge_no_intercept);
    run_test("ridge_concept",            test_ridge_concept);
    run_test("ridge_sparse_matches_dense", test_ridge_sparse_matches_dense);
    run_test("ridge_sparse_no_intercept",  test_ridge_sparse_no_intercept);
    run_test("ridge_sparse_empty_throws",  test_ridge_sparse_empty_throws);
    run_test("ridge_multi_target_recovers_two_outputs",
             test_ridge_multi_target_recovers_two_outputs);
    run_test("ridge_multi_target_with_intercept",
             test_ridge_multi_target_with_intercept);
    run_test("ridge_single_target_then_coef_matrix_synthesised",
             test_ridge_single_target_then_coef_matrix_synthesised);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
