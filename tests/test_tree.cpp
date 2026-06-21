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

void test_dtr_multi_target_recovers_two_outputs() {
    // Two simple step-function targets; per-target trees should recover
    // each one exactly given enough samples per leaf.
    constexpr int n = 30;
    Eigen::MatrixXd X(n, 1);
    Eigen::MatrixXd Y(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i);
        // Target 0: piecewise constant in two halves.
        Y(i, 0) = (i < n / 2) ? 1.0 : 5.0;
        // Target 1: piecewise constant in three thirds.
        if      (i < n / 3)        Y(i, 1) = -2.0;
        else if (i < 2 * n / 3)    Y(i, 1) =  3.0;
        else                       Y(i, 1) =  7.0;
    }

    Skigen::DecisionTreeRegressor<double> dt(/*max_depth=*/-1, 2,
                                             0, 0.0,
                                             std::optional<uint64_t>(7));
    dt.fit_multi(X, Y);

    ASSERT_TRUE(dt.n_targets() == 2);
    Eigen::MatrixXd Yp = dt.predict_multi(X);
    ASSERT_TRUE(Yp.rows() == n);
    ASSERT_TRUE(Yp.cols() == 2);
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(Yp(i, 0), Y(i, 0), 1e-9);
        ASSERT_NEAR(Yp(i, 1), Y(i, 1), 1e-9);
    }
}

void test_dtr_single_target_API_after_fit_multi_consistent() {
    Eigen::MatrixXd X(8, 1);
    Eigen::MatrixXd Y(8, 2);
    for (int i = 0; i < 8; ++i) {
        X(i, 0) = i;
        Y(i, 0) = i * 2.0;
        Y(i, 1) = -3.0 * i;
    }
    Skigen::DecisionTreeRegressor<double> dt(-1, 2, 0, 0.0,
                                             std::optional<uint64_t>(0));
    dt.fit_multi(X, Y);

    // Single-target predict() should match the first target column of
    // predict_multi().
    Eigen::VectorXd v = dt.predict(X);
    Eigen::MatrixXd m = dt.predict_multi(X);
    ASSERT_TRUE(v.size() == m.rows());
    for (int i = 0; i < v.size(); ++i) {
        ASSERT_NEAR(v(i), m(i, 0), 1e-12);
    }
}

void test_dtr_multi_target_uses_shared_split() {
    Eigen::MatrixXd X(6, 1);
    X << 0, 1, 2, 3, 4, 5;
    Eigen::MatrixXd Y(6, 2);
    Y.col(0) << 0, 0, 0, 10, 10, 10;
    Y.col(1) << 0, 10, 10, 10, 10, 10;

    Skigen::DecisionTreeRegressor<double> dt(/*max_depth=*/1, 2,
                                             0, 0.0,
                                             std::optional<uint64_t>(3));
    dt.fit_multi(X, Y);
    Eigen::MatrixXd Yp = dt.predict_multi(X);

    ASSERT_TRUE(Yp.rows() == 6);
    ASSERT_TRUE(Yp.cols() == 2);
    ASSERT_NEAR(Yp(0, 0), 0.0, 1e-12);
    ASSERT_NEAR(Yp(3, 0), 10.0, 1e-12);
    ASSERT_NEAR(Yp(0, 1), Yp(1, 1), 1e-12);
    ASSERT_NEAR(Yp(1, 1), Yp(2, 1), 1e-12);
    ASSERT_NEAR(Yp(0, 1), 20.0 / 3.0, 1e-12);
    ASSERT_NEAR(Yp(3, 1), 10.0, 1e-12);
}

void test_dtr_predict_multi_after_single_target_fit() {
    Eigen::MatrixXd X(4, 1); X << 0, 1, 2, 3;
    Eigen::VectorXd y(4); y << 0.0, 2.0, 4.0, 6.0;
    Skigen::DecisionTreeRegressor<double> dt;
    dt.fit(X, y);
    ASSERT_TRUE(dt.n_targets() == 1);
    Eigen::MatrixXd m = dt.predict_multi(X);
    ASSERT_TRUE(m.rows() == 4);
    ASSERT_TRUE(m.cols() == 1);
    Eigen::VectorXd v = dt.predict(X);
    for (int i = 0; i < 4; ++i) ASSERT_NEAR(m(i, 0), v(i), 1e-12);
}

void test_dtr_multi_target_dim_mismatch_throws() {
    Eigen::MatrixXd X(4, 1); X << 0, 1, 2, 3;
    Eigen::MatrixXd Y(5, 2); Y.setOnes();
    Skigen::DecisionTreeRegressor<double> dt;
    ASSERT_THROW(dt.fit_multi(X, Y), std::invalid_argument);
}

// -- Sparse fit overloads (densify-and-fit) -----------------------------

void test_dtc_sparse_fit_predict_matches_dense() {
    Eigen::MatrixXd Xd(8, 2);
    Xd << 0, 0,  0, 1,  1, 0,  1, 1,
          0, 5,  0, 6,  1, 5,  1, 6;
    Eigen::VectorXi y(8); y << 0, 0, 0, 0, 1, 1, 1, 1;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::DecisionTreeClassifier<double> dt_d, dt_s;
    dt_d.fit(Xd, y);
    dt_s.fit(Xs, y);

    auto pd = dt_d.predict(Xd);
    auto ps = dt_s.predict(Xs);
    for (int i = 0; i < 8; ++i) ASSERT_TRUE(pd(i) == ps(i));
}

void test_dtr_sparse_fit_predict_matches_dense() {
    Eigen::MatrixXd Xd(10, 1);
    for (int i = 0; i < 10; ++i) Xd(i, 0) = i;
    Eigen::VectorXd y(10);
    y << 1, 1, 1, 1, 1, 5, 5, 5, 5, 5;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::DecisionTreeRegressor<double> dt_d, dt_s;
    dt_d.fit(Xd, y);
    dt_s.fit(Xs, y);

    auto pd = dt_d.predict(Xd);
    auto ps = dt_s.predict(Xs);
    for (int i = 0; i < 10; ++i) ASSERT_NEAR(pd(i), ps(i), 1e-12);
}

void test_dt_sparse_feature_count_check() {
    Eigen::MatrixXd Xd(4, 2); Xd << 0, 0, 1, 1, 0, 1, 1, 0;
    Eigen::VectorXi y(4); y << 0, 1, 0, 1;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::DecisionTreeClassifier<double> dt;
    dt.fit(Xs, y);

    Eigen::SparseMatrix<double> Xbad(2, 3);
    Xbad.insert(0, 0) = 1.0;
    Xbad.makeCompressed();
    ASSERT_THROW(dt.predict(Xbad), std::invalid_argument);
}

void test_dt_sparse_empty_throws() {
    Eigen::SparseMatrix<double> X(0, 0);
    Eigen::VectorXi y(0);
    Skigen::DecisionTreeClassifier<double> dt;
    ASSERT_THROW(dt.fit(X, y), std::invalid_argument);
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
    run_test("dtr_multi_target_recovers_two_outputs",
             test_dtr_multi_target_recovers_two_outputs);
    run_test("dtr_single_target_API_after_fit_multi_consistent",
             test_dtr_single_target_API_after_fit_multi_consistent);
    run_test("dtr_multi_target_uses_shared_split",
             test_dtr_multi_target_uses_shared_split);
    run_test("dtr_predict_multi_after_single_target_fit",
             test_dtr_predict_multi_after_single_target_fit);
    run_test("dtr_multi_target_dim_mismatch_throws",
             test_dtr_multi_target_dim_mismatch_throws);
    run_test("dtc_sparse_fit_predict_matches_dense",
             test_dtc_sparse_fit_predict_matches_dense);
    run_test("dtr_sparse_fit_predict_matches_dense",
             test_dtr_sparse_fit_predict_matches_dense);
    run_test("dt_sparse_feature_count_check",
             test_dt_sparse_feature_count_check);
    run_test("dt_sparse_empty_throws",
             test_dt_sparse_empty_throws);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
