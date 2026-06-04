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

void test_knn_regressor_multi_target_basic() {
    // 6 training points with two correlated targets.
    Eigen::MatrixXd X(6, 1);
    X << 1, 2, 3, 7, 8, 9;
    Eigen::MatrixXd Y(6, 2);
    Y << 1, 10, 2, 20, 3, 30, 7, 70, 8, 80, 9, 90;

    Skigen::KNeighborsRegressor<double> knn(/*n_neighbors=*/3);
    knn.fit_multi(X, Y);

    ASSERT_TRUE(knn.n_targets() == 2);

    Eigen::MatrixXd Q(2, 1);
    Q << 2.0, 8.0;
    Eigen::MatrixXd Yp = knn.predict_multi(Q);
    ASSERT_TRUE(Yp.rows() == 2);
    ASSERT_TRUE(Yp.cols() == 2);
    // Mean of the 3 closest training rows for each query.
    // Q=2 → neighbours {1,2,3} → target0 mean=2, target1 mean=20.
    // Q=8 → neighbours {7,8,9} → target0 mean=8, target1 mean=80.
    ASSERT_NEAR(Yp(0, 0), 2.0, 1e-12);
    ASSERT_NEAR(Yp(0, 1), 20.0, 1e-12);
    ASSERT_NEAR(Yp(1, 0), 8.0, 1e-12);
    ASSERT_NEAR(Yp(1, 1), 80.0, 1e-12);
}

void test_knn_regressor_predict_multi_after_single_target_fit() {
    // After single-target fit, predict_multi() should still work and
    // return a 1-column matrix matching the 1-D predict() output.
    Eigen::MatrixXd X(4, 1); X << 1, 2, 3, 4;
    Eigen::VectorXd y(4); y << 2, 4, 6, 8;
    Skigen::KNeighborsRegressor<double> knn(2);
    knn.fit(X, y);
    ASSERT_TRUE(knn.n_targets() == 1);
    Eigen::MatrixXd Q(1, 1); Q << 2.5;
    Eigen::VectorXd v = knn.predict(Q);
    Eigen::MatrixXd m = knn.predict_multi(Q);
    ASSERT_TRUE(m.rows() == 1);
    ASSERT_TRUE(m.cols() == 1);
    ASSERT_NEAR(m(0, 0), v(0), 1e-12);
}

void test_knn_regressor_multi_target_dim_mismatch_throws() {
    Eigen::MatrixXd X(4, 1); X << 1, 2, 3, 4;
    Eigen::MatrixXd Y(5, 2); Y.setOnes();
    Skigen::KNeighborsRegressor<double> knn;
    bool threw = false;
    try { knn.fit_multi(X, Y); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ===================================================================
// LocalOutlierFactor Tests
// ===================================================================

void test_lof_basic_outlier() {
    // Tight cluster of 5 inliers + 1 obvious outlier
    Eigen::MatrixXd X(6, 2);
    X << 0, 0,
         0.1, 0,
         0, 0.1,
         0.1, 0.1,
         0.05, 0.05,
         10, 10;  // outlier

    Skigen::LocalOutlierFactor<double> lof(3);
    lof.fit(X);

    ASSERT_TRUE(lof.is_fitted());
    ASSERT_TRUE(lof.n_neighbors_used() == 3);

    auto scores = lof.lof_scores();
    ASSERT_TRUE(scores.size() == 6);

    // Outlier should have the highest LOF score
    Eigen::Index max_idx;
    scores.maxCoeff(&max_idx);
    ASSERT_TRUE(max_idx == 5);  // The outlier

    // Outlier score should be significantly > 1
    ASSERT_TRUE(scores(5) > 2.0);

    // Inlier scores should be close to 1
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(scores(i) < 2.0);
    }
}

void test_lof_negative_outlier_factor() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0, 1, 0, 0, 1, 1, 1;

    Skigen::LocalOutlierFactor<double> lof(2);
    lof.fit(X);

    auto nof = lof.negative_outlier_factor();
    // Negative outlier factor should be negative
    for (Eigen::Index i = 0; i < nof.size(); ++i) {
        ASSERT_TRUE(nof(i) <= 0);
    }

    // lof_scores should equal -nof
    auto scores = lof.lof_scores();
    for (Eigen::Index i = 0; i < nof.size(); ++i) {
        ASSERT_NEAR(scores(i), -nof(i), 1e-14);
    }
}

void test_lof_k_clamp() {
    // n_neighbors > n_samples-1 should be clamped
    Eigen::MatrixXd X(3, 1);
    X << 0, 1, 100;

    Skigen::LocalOutlierFactor<double> lof(50);
    lof.fit(X);

    ASSERT_TRUE(lof.n_neighbors_used() == 2);  // clamped to n-1
}

void test_lof_fit_predict_labels() {
    Eigen::MatrixXd X(7, 2);
    X << 0, 0,
         0.1, 0,
         0, 0.1,
         0.1, 0.1,
         0.05, 0.05,
         -0.05, 0.05,
         50, 50;  // outlier

    Skigen::LocalOutlierFactor<double> lof(3);
    lof.fit(X);

    auto labels = lof.fit_predict_labels();
    // The far-out point should be labeled -1
    ASSERT_TRUE(labels(6) == -1);
    // At least some inliers should be labeled +1
    int inlier_count = 0;
    for (Eigen::Index i = 0; i < 6; ++i) {
        if (labels(i) == 1) ++inlier_count;
    }
    ASSERT_TRUE(inlier_count >= 3);
}

void test_lof_not_fitted() {
    Skigen::LocalOutlierFactor<double> lof;
    ASSERT_THROW(lof.negative_outlier_factor(), std::runtime_error);
    ASSERT_THROW(lof.lof_scores(), std::runtime_error);
    ASSERT_THROW(lof.fit_predict_labels(), std::runtime_error);
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
    run_test("knn_regressor_multi_target_basic",
             test_knn_regressor_multi_target_basic);
    run_test("knn_regressor_predict_multi_after_single_target_fit",
             test_knn_regressor_predict_multi_after_single_target_fit);
    run_test("knn_regressor_multi_target_dim_mismatch_throws",
             test_knn_regressor_multi_target_dim_mismatch_throws);

    std::cout << "\n=== LocalOutlierFactor Tests ===\n";
    run_test("lof_basic_outlier", test_lof_basic_outlier);
    run_test("lof_negative_outlier_factor", test_lof_negative_outlier_factor);
    run_test("lof_k_clamp", test_lof_k_clamp);
    run_test("lof_fit_predict_labels", test_lof_fit_predict_labels);
    run_test("lof_not_fitted", test_lof_not_fitted);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
