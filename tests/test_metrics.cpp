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
// Regression Metrics Tests
// ===================================================================

void test_mse() {
    Eigen::VectorXd y_true(4);
    y_true << 3, -0.5, 2, 7;
    Eigen::VectorXd y_pred(4);
    y_pred << 2.5, 0.0, 2, 8;

    double mse = Skigen::Metrics::mean_squared_error(y_true, y_pred);
    ASSERT_NEAR(mse, 0.375, 1e-10);
}

void test_rmse() {
    Eigen::VectorXd y_true(4);
    y_true << 3, -0.5, 2, 7;
    Eigen::VectorXd y_pred(4);
    y_pred << 2.5, 0.0, 2, 8;

    double rmse = Skigen::Metrics::root_mean_squared_error(y_true, y_pred);
    ASSERT_NEAR(rmse, std::sqrt(0.375), 1e-10);
}

void test_mae() {
    Eigen::VectorXd y_true(4);
    y_true << 3, -0.5, 2, 7;
    Eigen::VectorXd y_pred(4);
    y_pred << 2.5, 0.0, 2, 8;

    double mae = Skigen::Metrics::mean_absolute_error(y_true, y_pred);
    ASSERT_NEAR(mae, 0.5, 1e-10);
}

void test_r2() {
    Eigen::VectorXd y_true(4);
    y_true << 3, -0.5, 2, 7;
    Eigen::VectorXd y_pred(4);
    y_pred << 2.5, 0.0, 2, 8;

    double r2 = Skigen::Metrics::r2_score(y_true, y_pred);
    // sklearn: r2_score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) ≈ 0.9486
    ASSERT_NEAR(r2, 0.9486, 0.001);
}

// ===================================================================
// Classification Metrics Tests
// ===================================================================

void test_accuracy() {
    Eigen::VectorXi y_true(4);
    y_true << 0, 1, 1, 0;
    Eigen::VectorXi y_pred(4);
    y_pred << 0, 1, 0, 0;

    double acc = Skigen::Metrics::accuracy_score(y_true, y_pred);
    ASSERT_NEAR(acc, 0.75, 1e-10);
}

void test_confusion_matrix() {
    Eigen::VectorXi y_true(4);
    y_true << 0, 0, 1, 1;
    Eigen::VectorXi y_pred(4);
    y_pred << 0, 1, 0, 1;

    auto cm = Skigen::Metrics::confusion_matrix(y_true, y_pred);
    ASSERT_TRUE(cm.rows() == 2);
    ASSERT_TRUE(cm.cols() == 2);
    // [[1,1],[1,1]]
    ASSERT_TRUE(cm(0, 0) == 1);
    ASSERT_TRUE(cm(0, 1) == 1);
    ASSERT_TRUE(cm(1, 0) == 1);
    ASSERT_TRUE(cm(1, 1) == 1);
}

void test_precision() {
    Eigen::VectorXi y_true(4);
    y_true << 0, 0, 1, 1;
    Eigen::VectorXi y_pred(4);
    y_pred << 0, 0, 1, 1;

    double p = Skigen::Metrics::precision_score(y_true, y_pred);
    ASSERT_NEAR(p, 1.0, 1e-10);
}

void test_recall() {
    Eigen::VectorXi y_true(4);
    y_true << 0, 0, 1, 1;
    Eigen::VectorXi y_pred(4);
    y_pred << 0, 0, 1, 1;

    double r = Skigen::Metrics::recall_score(y_true, y_pred);
    ASSERT_NEAR(r, 1.0, 1e-10);
}

void test_f1() {
    Eigen::VectorXi y_true(4);
    y_true << 0, 0, 1, 1;
    Eigen::VectorXi y_pred(4);
    y_pred << 0, 1, 0, 1;

    double f1 = Skigen::Metrics::f1_score(y_true, y_pred);
    // precision = 0.5, recall = 0.5, f1 = 0.5
    ASSERT_NEAR(f1, 0.5, 1e-10);
}

// ===================================================================
// Pairwise Metrics Tests
// ===================================================================

void test_euclidean_distances() {
    Eigen::MatrixXd X(2, 2);
    X << 0, 0,
         3, 4;
    Eigen::MatrixXd Y(1, 2);
    Y << 0, 0;

    auto D = Skigen::Metrics::euclidean_distances(X, Y);
    ASSERT_NEAR(D(0, 0), 0.0, 1e-10);
    ASSERT_NEAR(D(1, 0), 5.0, 1e-10);
}

void test_cosine_similarity() {
    Eigen::MatrixXd X(2, 2);
    X << 1, 0,
         0, 1;
    Eigen::MatrixXd Y(1, 2);
    Y << 1, 0;

    auto S = Skigen::Metrics::cosine_similarity(X, Y);
    ASSERT_NEAR(S(0, 0), 1.0, 1e-10);  // parallel
    ASSERT_NEAR(S(1, 0), 0.0, 1e-10);  // orthogonal
}

// ===================================================================

int main() {
    std::cout << "=== Regression Metrics Tests ===\n";
    run_test("mse", test_mse);
    run_test("rmse", test_rmse);
    run_test("mae", test_mae);
    run_test("r2", test_r2);

    std::cout << "\n=== Classification Metrics Tests ===\n";
    run_test("accuracy", test_accuracy);
    run_test("confusion_matrix", test_confusion_matrix);
    run_test("precision", test_precision);
    run_test("recall", test_recall);
    run_test("f1", test_f1);

    std::cout << "\n=== Pairwise Metrics Tests ===\n";
    run_test("euclidean_distances", test_euclidean_distances);
    run_test("cosine_similarity", test_cosine_similarity);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
