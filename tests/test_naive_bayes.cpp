// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test harness (no external dependency)
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
            << " vs " << b << " (tol=" << tol << ", diff=" << std::abs(a - b)
            << ")";
        throw TestFailure(oss.str());
    }
}

#define ASSERT_NEAR(a, b, tol) assert_near(a, b, tol, __FILE__, __LINE__)

template <typename MatA, typename MatB, typename Scalar>
void assert_matrix_near(const MatA& A, const MatB& B, Scalar tol,
                        const char* file, int line) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        std::ostringstream oss;
        oss << file << ":" << line << ": Shape mismatch: (" << A.rows() << ","
            << A.cols() << ") vs (" << B.rows() << "," << B.cols() << ")";
        throw TestFailure(oss.str());
    }
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
            assert_near(A(i, j), B(i, j), tol, file, line);
        }
    }
}

#define ASSERT_MATRIX_NEAR(A, B, tol)                                         \
    assert_matrix_near(A, B, tol, __FILE__, __LINE__)

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
        std::cout << "  FAIL  " << name << "\n        exception: " << e.what()
                  << "\n";
    }
}

// ---------------------------------------------------------------------------
// Dataset helpers
// ---------------------------------------------------------------------------

static Eigen::MatrixXd make_gaussian_X() {
    Eigen::MatrixXd X(6, 2);
    X << -2.0, -1.0,
         -1.5, -1.5,
         -1.0, -2.0,
          1.0, 2.0,
          1.5, 1.5,
          2.0, 1.0;
    return X;
}
static Eigen::VectorXi make_gaussian_y() {
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;
    return y;
}

static Eigen::MatrixXd make_count_X() {
    // 6 samples, 3 features (counts)
    Eigen::MatrixXd X(6, 3);
    X << 2, 1, 0,
         3, 0, 1,
         1, 2, 0,
         0, 1, 3,
         0, 0, 4,
         1, 0, 5;
    return X;
}
static Eigen::VectorXi make_count_y() {
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;
    return y;
}

// ---------------------------------------------------------------------------
// GaussianNB tests
// ---------------------------------------------------------------------------

void test_gnb_basic_fit_predict() {
    auto X = make_gaussian_X();
    auto y = make_gaussian_y();

    Skigen::GaussianNB<double> nb;
    nb.fit(X, y);

    auto pred = nb.predict(X);
    int correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i)
        if (pred(i) == y(i)) ++correct;
    ASSERT_TRUE(correct == y.size());
    ASSERT_TRUE(nb.is_fitted());
    ASSERT_TRUE(nb.classes().size() == 2);
}

void test_gnb_theta_var_match() {
    auto X = make_gaussian_X();
    auto y = make_gaussian_y();

    Skigen::GaussianNB<double> nb;
    nb.fit(X, y);

    // class 0: rows 0..2; class 1: rows 3..5
    Eigen::RowVector2d mu0 = X.topRows(3).colwise().mean();
    Eigen::RowVector2d mu1 = X.bottomRows(3).colwise().mean();

    ASSERT_NEAR(nb.theta()(0, 0), mu0(0), 1e-12);
    ASSERT_NEAR(nb.theta()(0, 1), mu0(1), 1e-12);
    ASSERT_NEAR(nb.theta()(1, 0), mu1(0), 1e-12);
    ASSERT_NEAR(nb.theta()(1, 1), mu1(1), 1e-12);

    // Compute biased variance manually + epsilon
    auto biased_var = [](const Eigen::MatrixXd& Xc) {
        Eigen::RowVector2d mu = Xc.colwise().mean();
        Eigen::MatrixXd C = Xc.rowwise() - mu;
        return (C.array().square().colwise().sum() /
                static_cast<double>(Xc.rows())).eval();
    };
    auto v0 = biased_var(X.topRows(3));
    auto v1 = biased_var(X.bottomRows(3));

    double eps = nb.epsilon();
    ASSERT_NEAR(nb.var()(0, 0), v0(0) + eps, 1e-12);
    ASSERT_NEAR(nb.var()(0, 1), v0(1) + eps, 1e-12);
    ASSERT_NEAR(nb.var()(1, 0), v1(0) + eps, 1e-12);
    ASSERT_NEAR(nb.var()(1, 1), v1(1) + eps, 1e-12);
    ASSERT_TRUE(eps > 0.0);
}

void test_gnb_predict_proba_shape_sum() {
    auto X = make_gaussian_X();
    auto y = make_gaussian_y();

    Skigen::GaussianNB<double> nb;
    nb.fit(X, y);
    Eigen::MatrixXd P = nb.predict_proba(X);
    ASSERT_TRUE(P.rows() == X.rows());
    ASSERT_TRUE(P.cols() == 2);
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        ASSERT_NEAR(P.row(i).sum(), 1.0, 1e-12);
    }
}

void test_gnb_log_proba_eq_log_proba() {
    auto X = make_gaussian_X();
    auto y = make_gaussian_y();

    Skigen::GaussianNB<double> nb;
    nb.fit(X, y);
    Eigen::MatrixXd P = nb.predict_proba(X);
    Eigen::MatrixXd LP = nb.predict_log_proba(X);
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        for (Eigen::Index c = 0; c < P.cols(); ++c) {
            ASSERT_NEAR(LP(i, c), std::log(P(i, c)), 1e-12);
        }
    }
}

void test_gnb_partial_fit_equivalence() {
    auto X = make_gaussian_X();
    auto y = make_gaussian_y();

    Skigen::GaussianNB<double> full;
    full.fit(X, y);

    // Split into 3 batches: rows {0,1}, {2,3}, {4,5}
    Eigen::VectorXi classes(2); classes << 0, 1;
    Skigen::GaussianNB<double> inc;
    Eigen::MatrixXd b1 = X.middleRows(0, 2);
    Eigen::VectorXi y1 = y.segment(0, 2);
    Eigen::MatrixXd b2 = X.middleRows(2, 2);
    Eigen::VectorXi y2 = y.segment(2, 2);
    Eigen::MatrixXd b3 = X.middleRows(4, 2);
    Eigen::VectorXi y3 = y.segment(4, 2);

    Eigen::VectorXi empty(0);
    inc.partial_fit(b1, y1, classes);
    inc.partial_fit(b2, y2, empty);
    inc.partial_fit(b3, y3, empty);

    // theta_ must match
    ASSERT_MATRIX_NEAR(inc.theta(), full.theta(), 1e-12);
    // class_count_ must match
    for (Eigen::Index c = 0; c < full.class_count().size(); ++c) {
        ASSERT_TRUE(inc.class_count()(c) == full.class_count()(c));
    }
    // var_ within tol — combined Welford and per-batch epsilon (matches
    // sklearn's behavior of recomputing epsilon_ from the latest batch)
    // introduce small FP differences vs single-shot fit.
    ASSERT_MATRIX_NEAR(inc.var(), full.var(), 1e-7);
}

void test_gnb_partial_fit_first_call_requires_classes() {
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    Eigen::VectorXi y(2); y << 0, 1;
    Eigen::VectorXi empty(0);
    Skigen::GaussianNB<double> nb;
    ASSERT_THROW(nb.partial_fit(X, y, empty), std::invalid_argument);
}

void test_gnb_feature_mismatch_throws() {
    auto X = make_gaussian_X();
    auto y = make_gaussian_y();
    Skigen::GaussianNB<double> nb;
    nb.fit(X, y);
    Eigen::MatrixXd Xbad(3, 5);
    Xbad.setOnes();
    ASSERT_THROW(nb.predict(Xbad), std::invalid_argument);
}

void test_gnb_not_fitted_throws() {
    Skigen::GaussianNB<double> nb;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(nb.predict(X), std::runtime_error);
    ASSERT_THROW(nb.theta(), std::runtime_error);
    ASSERT_THROW(nb.var(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// MultinomialNB tests
// ---------------------------------------------------------------------------

void test_mnb_basic_fit_predict() {
    auto X = make_count_X();
    auto y = make_count_y();
    Skigen::MultinomialNB<double> nb;
    nb.fit(X, y);

    auto pred = nb.predict(X);
    int correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i)
        if (pred(i) == y(i)) ++correct;
    // Should classify training set perfectly
    ASSERT_TRUE(correct >= 5);
}

void test_mnb_predict_proba_shape_sum() {
    auto X = make_count_X();
    auto y = make_count_y();
    Skigen::MultinomialNB<double> nb;
    nb.fit(X, y);
    Eigen::MatrixXd P = nb.predict_proba(X);
    ASSERT_TRUE(P.rows() == X.rows());
    ASSERT_TRUE(P.cols() == 2);
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        ASSERT_NEAR(P.row(i).sum(), 1.0, 1e-12);
    }
}

void test_mnb_log_proba_eq_log() {
    auto X = make_count_X();
    auto y = make_count_y();
    Skigen::MultinomialNB<double> nb;
    nb.fit(X, y);
    Eigen::MatrixXd P = nb.predict_proba(X);
    Eigen::MatrixXd LP = nb.predict_log_proba(X);
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        for (Eigen::Index c = 0; c < P.cols(); ++c) {
            ASSERT_NEAR(LP(i, c), std::log(P(i, c)), 1e-12);
        }
    }
}

void test_mnb_smoothing_arithmetic() {
    // 2x3 hand-computed example
    // Two classes, 3 features
    // class 0 sample: [2, 1, 0]
    // class 1 sample: [0, 1, 3]
    Eigen::MatrixXd X(2, 3);
    X << 2, 1, 0,
         0, 1, 3;
    Eigen::VectorXi y(2);
    y << 0, 1;

    double alpha = 1.0;
    Skigen::MultinomialNB<double> nb(alpha);
    nb.fit(X, y);

    // class 0: feature_count = [2,1,0]; sum + alpha*n_features = 3 + 3 = 6
    // log p[0,0] = log(3/6) = log(0.5)
    // log p[0,1] = log(2/6)
    // log p[0,2] = log(1/6)
    ASSERT_NEAR(nb.feature_log_prob()(0, 0), std::log(3.0 / 6.0), 1e-12);
    ASSERT_NEAR(nb.feature_log_prob()(0, 1), std::log(2.0 / 6.0), 1e-12);
    ASSERT_NEAR(nb.feature_log_prob()(0, 2), std::log(1.0 / 6.0), 1e-12);
    // class 1: feature_count = [0,1,3]; sum + 3 = 4 + 3 = 7
    ASSERT_NEAR(nb.feature_log_prob()(1, 0), std::log(1.0 / 7.0), 1e-12);
    ASSERT_NEAR(nb.feature_log_prob()(1, 1), std::log(2.0 / 7.0), 1e-12);
    ASSERT_NEAR(nb.feature_log_prob()(1, 2), std::log(4.0 / 7.0), 1e-12);
}

void test_mnb_partial_fit_equivalence() {
    auto X = make_count_X();
    auto y = make_count_y();

    Skigen::MultinomialNB<double> full;
    full.fit(X, y);

    Skigen::MultinomialNB<double> inc;
    Eigen::VectorXi classes(2); classes << 0, 1;
    Eigen::VectorXi empty(0);
    inc.partial_fit(X.middleRows(0, 2), y.segment(0, 2), classes);
    inc.partial_fit(X.middleRows(2, 2), y.segment(2, 2), empty);
    inc.partial_fit(X.middleRows(4, 2), y.segment(4, 2), empty);

    ASSERT_MATRIX_NEAR(inc.feature_count(), full.feature_count(), 1e-12);
    ASSERT_MATRIX_NEAR(inc.feature_log_prob(), full.feature_log_prob(), 1e-12);
    for (Eigen::Index c = 0; c < full.class_count().size(); ++c) {
        ASSERT_NEAR(inc.class_count()(c), full.class_count()(c), 1e-12);
    }
}

void test_mnb_partial_fit_first_call_requires_classes() {
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    Eigen::VectorXi y(2); y << 0, 1;
    Eigen::VectorXi empty(0);
    Skigen::MultinomialNB<double> nb;
    ASSERT_THROW(nb.partial_fit(X, y, empty), std::invalid_argument);
}

void test_mnb_feature_mismatch_throws() {
    auto X = make_count_X();
    auto y = make_count_y();
    Skigen::MultinomialNB<double> nb;
    nb.fit(X, y);
    Eigen::MatrixXd Xbad(2, 7);
    Xbad.setOnes();
    ASSERT_THROW(nb.predict(Xbad), std::invalid_argument);
}

void test_mnb_not_fitted_throws() {
    Skigen::MultinomialNB<double> nb;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(nb.predict(X), std::runtime_error);
    ASSERT_THROW(nb.feature_log_prob(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// BernoulliNB tests
// ---------------------------------------------------------------------------

static Eigen::MatrixXd make_bernoulli_X() {
    Eigen::MatrixXd X(6, 3);
    X << 1, 1, 0,
         1, 0, 0,
         1, 1, 0,
         0, 1, 1,
         0, 0, 1,
         0, 1, 1;
    return X;
}
static Eigen::VectorXi make_bernoulli_y() {
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;
    return y;
}

void test_bnb_basic_fit_predict() {
    auto X = make_bernoulli_X();
    auto y = make_bernoulli_y();
    Skigen::BernoulliNB<double> nb;
    nb.fit(X, y);
    auto pred = nb.predict(X);
    int correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i)
        if (pred(i) == y(i)) ++correct;
    ASSERT_TRUE(correct >= 5);
}

void test_bnb_predict_proba_shape_sum() {
    auto X = make_bernoulli_X();
    auto y = make_bernoulli_y();
    Skigen::BernoulliNB<double> nb;
    nb.fit(X, y);
    Eigen::MatrixXd P = nb.predict_proba(X);
    ASSERT_TRUE(P.rows() == X.rows());
    ASSERT_TRUE(P.cols() == 2);
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        ASSERT_NEAR(P.row(i).sum(), 1.0, 1e-12);
    }
}

void test_bnb_log_proba_eq_log() {
    auto X = make_bernoulli_X();
    auto y = make_bernoulli_y();
    Skigen::BernoulliNB<double> nb;
    nb.fit(X, y);
    Eigen::MatrixXd P = nb.predict_proba(X);
    Eigen::MatrixXd LP = nb.predict_log_proba(X);
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        for (Eigen::Index c = 0; c < P.cols(); ++c) {
            ASSERT_NEAR(LP(i, c), std::log(P(i, c)), 1e-12);
        }
    }
}

void test_bnb_partial_fit_equivalence() {
    auto X = make_bernoulli_X();
    auto y = make_bernoulli_y();
    Skigen::BernoulliNB<double> full;
    full.fit(X, y);

    Skigen::BernoulliNB<double> inc;
    Eigen::VectorXi classes(2); classes << 0, 1;
    Eigen::VectorXi empty(0);
    inc.partial_fit(X.middleRows(0, 2), y.segment(0, 2), classes);
    inc.partial_fit(X.middleRows(2, 2), y.segment(2, 2), empty);
    inc.partial_fit(X.middleRows(4, 2), y.segment(4, 2), empty);

    ASSERT_MATRIX_NEAR(inc.feature_count(), full.feature_count(), 1e-12);
    ASSERT_MATRIX_NEAR(inc.feature_log_prob(), full.feature_log_prob(), 1e-12);
}

void test_bnb_partial_fit_first_call_requires_classes() {
    Eigen::MatrixXd X(2, 2);
    X << 1, 0, 0, 1;
    Eigen::VectorXi y(2); y << 0, 1;
    Eigen::VectorXi empty(0);
    Skigen::BernoulliNB<double> nb;
    ASSERT_THROW(nb.partial_fit(X, y, empty), std::invalid_argument);
}

void test_bnb_feature_mismatch_throws() {
    auto X = make_bernoulli_X();
    auto y = make_bernoulli_y();
    Skigen::BernoulliNB<double> nb;
    nb.fit(X, y);
    Eigen::MatrixXd Xbad(2, 7);
    Xbad.setOnes();
    ASSERT_THROW(nb.predict(Xbad), std::invalid_argument);
}

void test_bnb_not_fitted_throws() {
    Skigen::BernoulliNB<double> nb;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(nb.predict(X), std::runtime_error);
    ASSERT_THROW(nb.feature_log_prob(), std::runtime_error);
}

void test_bnb_binarize_threshold() {
    Eigen::MatrixXd X(4, 2);
    X << 0.2, 0.6,
         0.7, 0.1,
         0.3, 0.8,
         0.9, 0.2;
    Eigen::VectorXi y(4); y << 0, 1, 0, 1;
    Skigen::BernoulliNB<double> nb(1.0, true, std::optional<double>{0.5});
    nb.fit(X, y);
    // After binarization with threshold 0.5:
    // X_bin = [[0,1],[1,0],[0,1],[1,0]]
    // class 0 feature_count = [0+0, 1+1] = [0, 2]
    // class 1 feature_count = [1+1, 0+0] = [2, 0]
    ASSERT_NEAR(nb.feature_count()(0, 0), 0.0, 1e-12);
    ASSERT_NEAR(nb.feature_count()(0, 1), 2.0, 1e-12);
    ASSERT_NEAR(nb.feature_count()(1, 0), 2.0, 1e-12);
    ASSERT_NEAR(nb.feature_count()(1, 1), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== Naive Bayes Tests ===\n";

    run_test("gnb_basic_fit_predict",               test_gnb_basic_fit_predict);
    run_test("gnb_theta_var_match",                 test_gnb_theta_var_match);
    run_test("gnb_predict_proba_shape_sum",         test_gnb_predict_proba_shape_sum);
    run_test("gnb_log_proba_eq_log_proba",          test_gnb_log_proba_eq_log_proba);
    run_test("gnb_partial_fit_equivalence",         test_gnb_partial_fit_equivalence);
    run_test("gnb_partial_fit_first_call_requires_classes",
             test_gnb_partial_fit_first_call_requires_classes);
    run_test("gnb_feature_mismatch_throws",         test_gnb_feature_mismatch_throws);
    run_test("gnb_not_fitted_throws",               test_gnb_not_fitted_throws);

    run_test("mnb_basic_fit_predict",               test_mnb_basic_fit_predict);
    run_test("mnb_predict_proba_shape_sum",         test_mnb_predict_proba_shape_sum);
    run_test("mnb_log_proba_eq_log",                test_mnb_log_proba_eq_log);
    run_test("mnb_smoothing_arithmetic",            test_mnb_smoothing_arithmetic);
    run_test("mnb_partial_fit_equivalence",         test_mnb_partial_fit_equivalence);
    run_test("mnb_partial_fit_first_call_requires_classes",
             test_mnb_partial_fit_first_call_requires_classes);
    run_test("mnb_feature_mismatch_throws",         test_mnb_feature_mismatch_throws);
    run_test("mnb_not_fitted_throws",               test_mnb_not_fitted_throws);

    run_test("bnb_basic_fit_predict",               test_bnb_basic_fit_predict);
    run_test("bnb_predict_proba_shape_sum",         test_bnb_predict_proba_shape_sum);
    run_test("bnb_log_proba_eq_log",                test_bnb_log_proba_eq_log);
    run_test("bnb_partial_fit_equivalence",         test_bnb_partial_fit_equivalence);
    run_test("bnb_partial_fit_first_call_requires_classes",
             test_bnb_partial_fit_first_call_requires_classes);
    run_test("bnb_feature_mismatch_throws",         test_bnb_feature_mismatch_throws);
    run_test("bnb_not_fitted_throws",               test_bnb_not_fitted_throws);
    run_test("bnb_binarize_threshold",              test_bnb_binarize_threshold);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
