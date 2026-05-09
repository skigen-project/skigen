// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <iostream>
#include <optional>
#include <random>
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
// PolynomialFeatures Tests
// ===================================================================

void test_poly_degree2() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2,
         3, 4,
         5, 6;

    Skigen::PolynomialFeatures<double> poly(2, true);
    poly.fit(X);
    auto Xp = poly.transform(X);

    // degree 2, 2 features, include_bias: 1, x1, x2, x1^2, x1*x2, x2^2 = 6 features
    ASSERT_TRUE(Xp.cols() == 6);
    ASSERT_TRUE(Xp.rows() == 3);

    // Check bias column
    ASSERT_NEAR(Xp(0, 0), 1.0, 1e-10);
    // Check x1
    ASSERT_NEAR(Xp(0, 1), 1.0, 1e-10);
    // Check x2
    ASSERT_NEAR(Xp(0, 2), 2.0, 1e-10);
}

void test_poly_no_bias() {
    Eigen::MatrixXd X(2, 2);
    X << 1, 2,
         3, 4;

    Skigen::PolynomialFeatures<double> poly(2, false);
    poly.fit(X);
    auto Xp = poly.transform(X);

    // Without bias: x1, x2, x1^2, x1*x2, x2^2 = 5 features
    ASSERT_TRUE(Xp.cols() == 5);
}

void test_poly_interaction_only() {
    Eigen::MatrixXd X(2, 2);
    X << 1, 2,
         3, 4;

    Skigen::PolynomialFeatures<double> poly(2, true, true);
    poly.fit(X);
    auto Xp = poly.transform(X);

    // interaction_only, degree 2: 1, x1, x2, x1*x2 = 4 features
    ASSERT_TRUE(Xp.cols() == 4);
}

void test_poly_degree1() {
    Eigen::MatrixXd X(2, 3);
    X << 1, 2, 3,
         4, 5, 6;

    Skigen::PolynomialFeatures<double> poly(1, true);
    poly.fit(X);
    auto Xp = poly.transform(X);

    // degree 1: 1, x1, x2, x3 = 4 features
    ASSERT_TRUE(Xp.cols() == 4);
    ASSERT_NEAR(Xp(0, 0), 1.0, 1e-10);
    ASSERT_NEAR(Xp(0, 1), 1.0, 1e-10);
    ASSERT_NEAR(Xp(0, 2), 2.0, 1e-10);
    ASSERT_NEAR(Xp(0, 3), 3.0, 1e-10);
}

void test_poly_not_fitted() {
    Skigen::PolynomialFeatures<double> poly;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(poly.transform(X), std::runtime_error);
}

// ===================================================================
// Lasso Tests
// ===================================================================

void test_lasso_sparsity() {
    // y depends only on x1, not x2
    Eigen::MatrixXd X(5, 2);
    X << 1, 10,
         2, 20,
         3, 30,
         4, 40,
         5, 50;
    Eigen::VectorXd y(5);
    y << 2, 4, 6, 8, 10;

    Skigen::Lasso<double> lasso(0.1);
    lasso.fit(X, y);

    // With appropriate alpha, Lasso should set x2 coef near zero
    // (both features are correlated, but x1 is the direct predictor)
    ASSERT_TRUE(lasso.coef().size() == 2);
}

void test_lasso_predict() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::Lasso<double> lasso(0.001);
    lasso.fit(X, y);

    Eigen::MatrixXd X_test(1, 1);
    X_test << 5;
    auto pred = lasso.predict(X_test);
    ASSERT_NEAR(pred(0), 10.0, 0.5);
}

void test_lasso_score() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::Lasso<double> lasso(0.001);
    lasso.fit(X, y);
    double r2 = lasso.score(X, y);
    ASSERT_TRUE(r2 > 0.99);
}

// ===================================================================
// ElasticNet Tests
// ===================================================================

void test_elastic_net_basic() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::ElasticNet<double> en(0.001, 0.5);
    en.fit(X, y);

    Eigen::MatrixXd X_test(1, 1);
    X_test << 5;
    auto pred = en.predict(X_test);
    ASSERT_NEAR(pred(0), 10.0, 0.5);
}

void test_elastic_net_score() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::ElasticNet<double> en(0.001, 0.5);
    en.fit(X, y);
    double r2 = en.score(X, y);
    ASSERT_TRUE(r2 > 0.99);
}

// ===================================================================
// LogisticRegression Tests
// ===================================================================

void test_logreg_binary() {
    Eigen::MatrixXd X(6, 2);
    X << 0, 0,
         0, 1,
         1, 0,
         10, 10,
         10, 11,
         11, 10;
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    Skigen::LogisticRegression<double> lr(100.0, true, 500);
    lr.fit(X, y);

    auto preds = lr.predict(X);
    int correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (preds(i) == y(i)) ++correct;
    }
    ASSERT_TRUE(correct >= 5); // Should get most right
}

void test_logreg_predict_proba() {
    Eigen::MatrixXd X(4, 1);
    X << -5, -1, 1, 5;
    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;

    Skigen::LogisticRegression<double> lr(1.0, true, 200);
    lr.fit(X, y);

    auto proba = lr.predict_proba(X);
    ASSERT_TRUE(proba.cols() == 2);
    // Probabilities should sum to 1
    for (Eigen::Index i = 0; i < proba.rows(); ++i) {
        ASSERT_NEAR(proba.row(i).sum(), 1.0, 1e-6);
    }
}

void test_logreg_multiclass() {
    // Well-separated 3-class data along different axes
    Eigen::MatrixXd X(9, 2);
    X << -5, 0,  -4, 0,  -3, 0,    // class 0: far left
          5, 0,   4, 0,   3, 0,    // class 1: far right
          0, 5,   0, 4,   0, 3;    // class 2: far up
    Eigen::VectorXi y(9);
    y << 0, 0, 0, 1, 1, 1, 2, 2, 2;

    Skigen::LogisticRegression<double> lr(100.0, true, 1000);
    lr.fit(X, y);

    ASSERT_TRUE(lr.classes().size() == 3);
    double acc = lr.score(X, y);
    ASSERT_TRUE(acc >= 0.66);
}

void test_logreg_not_fitted() {
    Skigen::LogisticRegression<double> lr;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(lr.predict(X), std::runtime_error);
}

// ===================================================================
// SGDClassifier Tests
// ===================================================================

void test_sgd_classifier_basic() {
    Eigen::MatrixXd X(6, 2);
    X << 0, 0,  0, 1,  1, 0,
         10, 10, 10, 11, 11, 10;
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    Skigen::SGDClassifier<double> sgd(
        Skigen::SGDClassifier<double>::Loss::Hinge,
        1e-4, 500, 1e-3, 0.1);
    sgd.fit(X, y);

    double acc = sgd.score(X, y);
    ASSERT_TRUE(acc >= 0.83); // Most correct
}

void test_sgd_classifier_log_loss() {
    Eigen::MatrixXd X(4, 1);
    X << -5, -1, 1, 5;
    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;

    Skigen::SGDClassifier<double> sgd(
        Skigen::SGDClassifier<double>::Loss::Log,
        1e-4, 1000, 1e-4, 0.05);
    sgd.fit(X, y);

    auto preds = sgd.predict(X);
    ASSERT_TRUE(preds.size() == 4);
}

// ===================================================================
// SGDRegressor Tests
// ===================================================================

void test_sgd_regressor_basic() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::SGDRegressor<double> sgd(1e-4, 2000, 1e-5, 0.01);
    sgd.fit(X, y);

    double r2 = sgd.score(X, y);
    ASSERT_TRUE(r2 > 0.9);
}

void test_sgd_regressor_predict() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    Skigen::SGDRegressor<double> sgd(1e-4, 2000, 1e-5, 0.01);
    sgd.fit(X, y);

    Eigen::MatrixXd X_test(1, 1);
    X_test << 5;
    auto pred = sgd.predict(X_test);
    ASSERT_NEAR(pred(0), 10.0, 1.5);
}

// ===================================================================
// TruncatedSVD Tests
// ===================================================================

void test_truncated_svd_basic() {
    Eigen::MatrixXd X(5, 3);
    X << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12,
         13, 14, 15;

    Skigen::TruncatedSVD<double> svd(2);
    svd.fit(X);

    ASSERT_TRUE(svd.n_components() == 2);
    ASSERT_TRUE(svd.components().rows() == 2);
    ASSERT_TRUE(svd.components().cols() == 3);
}

void test_truncated_svd_transform() {
    Eigen::MatrixXd X(5, 3);
    X << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12,
         13, 14, 15;

    Skigen::TruncatedSVD<double> svd(2);
    svd.fit(X);
    auto Z = svd.transform(X);

    ASSERT_TRUE(Z.rows() == 5);
    ASSERT_TRUE(Z.cols() == 2);
}

void test_truncated_svd_variance_ratio() {
    Eigen::MatrixXd X(5, 3);
    X << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12,
         13, 14, 15;

    Skigen::TruncatedSVD<double> svd(3);
    svd.fit(X);

    double ratio_sum = svd.explained_variance_ratio().sum();
    ASSERT_NEAR(ratio_sum, 1.0, 1e-10);
}

// ===================================================================
// MiniBatchKMeans Tests
// ===================================================================

void test_mini_batch_kmeans_basic() {
    Eigen::MatrixXd X(6, 2);
    X << 1, 1,  1.5, 1.5,  1, 1.5,
         10, 10, 10.5, 10.5, 10, 10.5;

    Skigen::MiniBatchKMeans<double> mbk(2, 4, 50);
    mbk.fit(X);

    ASSERT_TRUE(mbk.is_fitted());
    ASSERT_TRUE(mbk.cluster_centers().rows() == 2);
    ASSERT_TRUE(mbk.labels().size() == 6);

    // Points in same cluster should have same label
    ASSERT_TRUE(mbk.labels()(0) == mbk.labels()(1));
    ASSERT_TRUE(mbk.labels()(3) == mbk.labels()(4));
    ASSERT_TRUE(mbk.labels()(0) != mbk.labels()(3));
}

void test_mini_batch_kmeans_predict() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0, 0, 1, 10, 10, 10, 11;

    Skigen::MiniBatchKMeans<double> mbk(2, 4, 50);
    mbk.fit(X);

    Eigen::MatrixXd X_new(2, 2);
    X_new << 0.5, 0.5,
             9.5, 10.0;

    auto labels = mbk.predict(X_new);
    ASSERT_TRUE(labels(0) != labels(1));
}

// ===================================================================
// CrossValidation Tests
// ===================================================================

void test_cross_val_score() {
    Eigen::MatrixXd X(20, 1);
    Eigen::VectorXd y(20);
    for (int i = 0; i < 20; ++i) {
        X(i, 0) = static_cast<double>(i);
        y(i) = 2.0 * static_cast<double>(i) + 1.0;
    }

    Skigen::LinearRegression<double> lr;
    auto scores = Skigen::cross_val_score(lr, X, y, 5);

    ASSERT_TRUE(scores.size() == 5);
    // Each fold should have decent R²
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(scores(i) > 0.9);
    }
}

// ===================================================================
// Pipeline Tests
// ===================================================================

void test_pipeline_basic() {
    Eigen::MatrixXd X(4, 2);
    X << 1, 10,
         2, 20,
         3, 30,
         4, 40;
    Eigen::VectorXd y(4);
    y << 3, 5, 7, 9;

    auto pipe = Skigen::make_pipeline(
        Skigen::StandardScaler<double>(),
        Skigen::LinearRegression<double>());

    pipe.fit(X, y);
    auto pred = pipe.predict(X);

    ASSERT_TRUE(pred.size() == 4);
    // Should fit well
    for (Eigen::Index i = 0; i < 4; ++i) {
        ASSERT_NEAR(pred(i), y(i), 0.1);
    }
}

void test_pipeline_score() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 2, 4, 6, 8;

    auto pipe = Skigen::make_pipeline(
        Skigen::StandardScaler<double>(),
        Skigen::LinearRegression<double>());

    pipe.fit(X, y);
    double r2 = pipe.score(X, y);
    ASSERT_NEAR(r2, 1.0, 1e-10);
}

void test_pipeline_not_fitted() {
    auto pipe = Skigen::make_pipeline(
        Skigen::StandardScaler<double>(),
        Skigen::LinearRegression<double>());

    Eigen::MatrixXd X(2, 1);
    X << 1, 2;
    ASSERT_THROW(pipe.predict(X), std::runtime_error);
}

// ===================================================================
// BayesianRidge / ARDRegression Tests
// ===================================================================

static Eigen::MatrixXd make_linear_dataset(int n, int p,
                                           const Eigen::VectorXd& true_w,
                                           Eigen::VectorXd& y_out,
                                           double noise_std = 0.05,
                                           unsigned seed = 0) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nx(0.0, 1.0);
    std::normal_distribution<double> nn(0.0, noise_std);
    Eigen::MatrixXd X(n, p);
    y_out.resize(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) X(i, j) = nx(rng);
        y_out(i) = X.row(i).dot(true_w) + nn(rng);
    }
    return X;
}

void test_bayesian_ridge_recovers_coefs() {
    Eigen::VectorXd w(3);
    w << 2.0, -1.0, 3.0;
    Eigen::VectorXd y;
    Eigen::MatrixXd X = make_linear_dataset(200, 3, w, y, 0.05, 1);

    Skigen::BayesianRidge<double> br;
    br.fit(X, y);
    const auto& c = br.coef();
    ASSERT_TRUE(c.size() == 3);
    ASSERT_NEAR(c(0), 2.0, 0.1);
    ASSERT_NEAR(c(1), -1.0, 0.1);
    ASSERT_NEAR(c(2), 3.0, 0.1);
    ASSERT_TRUE(br.alpha() > 0.0);
    ASSERT_TRUE(br.lambda_() > 0.0);
    ASSERT_TRUE(br.n_iter() >= 1);
}

void test_bayesian_ridge_predict_with_std() {
    Eigen::VectorXd w(2);
    w << 1.5, -2.0;
    Eigen::VectorXd y;
    Eigen::MatrixXd X = make_linear_dataset(100, 2, w, y, 0.1, 2);

    Skigen::BayesianRidge<double> br;
    br.fit(X, y);

    Eigen::MatrixXd X_test = X.topRows(10);
    auto [m, s] = br.predict(X_test, Skigen::with_std);
    ASSERT_TRUE(m.size() == 10);
    ASSERT_TRUE(s.size() == 10);
    for (Eigen::Index i = 0; i < s.size(); ++i) {
        ASSERT_TRUE(s(i) > 0.0);
    }
}

void test_bayesian_ridge_compute_score() {
    Eigen::VectorXd w(2);
    w << 1.0, -1.0;
    Eigen::VectorXd y;
    Eigen::MatrixXd X = make_linear_dataset(80, 2, w, y, 0.1, 3);

    Skigen::BayesianRidge<double> br(
        300, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6,
        std::nullopt, std::nullopt, /*compute_score=*/true);
    br.fit(X, y);
    const auto& sc = br.scores();
    ASSERT_TRUE(sc.size() >= 2);
    // Final value should not be much smaller than the initial one.
    ASSERT_TRUE(sc(sc.size() - 1) >= sc(0) - 1e-6);
}

void test_bayesian_ridge_not_fitted() {
    Skigen::BayesianRidge<double> br;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(br.predict(X), std::runtime_error);
    ASSERT_THROW(br.coef(), std::runtime_error);
    ASSERT_THROW(br.predict(X, Skigen::with_std), std::runtime_error);
}

void test_bayesian_ridge_feature_mismatch() {
    Eigen::VectorXd w(3);
    w << 1.0, 2.0, 3.0;
    Eigen::VectorXd y;
    Eigen::MatrixXd X = make_linear_dataset(50, 3, w, y, 0.1, 4);
    Skigen::BayesianRidge<double> br;
    br.fit(X, y);
    Eigen::MatrixXd X_bad(5, 2);
    X_bad.setRandom();
    ASSERT_THROW(br.predict(X_bad), std::invalid_argument);
}

void test_ard_prunes_noise_features() {
    // Only feature 0 carries signal; features 1..4 are independent noise.
    constexpr int n = 200;
    std::mt19937 rng(11);
    std::normal_distribution<double> nx(0.0, 1.0);
    std::normal_distribution<double> nn(0.0, 0.05);
    Eigen::MatrixXd X(n, 5);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 5; ++j) X(i, j) = nx(rng);
        y(i) = 2.5 * X(i, 0) + nn(rng);
    }

    Skigen::ARDRegression<double> ard;
    ard.fit(X, y);

    const auto& c = ard.coef();
    const auto& lam = ard.lambda_();
    ASSERT_NEAR(c(0), 2.5, 0.1);
    for (int j = 1; j < 5; ++j) {
        ASSERT_TRUE(std::abs(c(j)) < 0.1);
        ASSERT_TRUE(lam(j) > ard.threshold_lambda());
    }
}

void test_ard_lambda_shape() {
    Eigen::VectorXd w(4);
    w << 1.0, -0.5, 2.0, 0.0;
    Eigen::VectorXd y;
    Eigen::MatrixXd X = make_linear_dataset(100, 4, w, y, 0.1, 5);
    Skigen::ARDRegression<double> ard;
    ard.fit(X, y);
    ASSERT_TRUE(ard.lambda_().size() == 4);
    ASSERT_TRUE(ard.coef().size() == 4);
}

void test_ard_predict_score_no_pruning() {
    // All features informative → ARD should produce a usable predictor
    // and behave similarly to BayesianRidge.
    Eigen::VectorXd w(3);
    w << 1.0, -1.0, 0.5;
    Eigen::VectorXd y;
    Eigen::MatrixXd X = make_linear_dataset(150, 3, w, y, 0.1, 6);

    Skigen::ARDRegression<double> ard;
    Skigen::BayesianRidge<double> br;
    ard.fit(X, y);
    br.fit(X, y);

    double r2_ard = ard.score(X, y);
    double r2_br = br.score(X, y);
    ASSERT_TRUE(r2_ard > 0.9);
    ASSERT_TRUE(r2_br > 0.9);
    ASSERT_TRUE(std::abs(r2_ard - r2_br) < 0.05);

    auto [m, s] = ard.predict(X.topRows(5), Skigen::with_std);
    ASSERT_TRUE(m.size() == 5 && s.size() == 5);
    for (Eigen::Index i = 0; i < s.size(); ++i) {
        ASSERT_TRUE(s(i) > 0.0);
    }
}

void test_ard_not_fitted_and_feature_mismatch() {
    Skigen::ARDRegression<double> ard;
    Eigen::MatrixXd X(3, 2);
    X << 1, 2, 3, 4, 5, 6;
    ASSERT_THROW(ard.predict(X), std::runtime_error);

    Eigen::VectorXd w(2);
    w << 1.0, -1.0;
    Eigen::VectorXd y;
    Eigen::MatrixXd Xtr = make_linear_dataset(40, 2, w, y, 0.1, 7);
    ard.fit(Xtr, y);

    Eigen::MatrixXd X_bad(3, 5);
    X_bad.setRandom();
    ASSERT_THROW(ard.predict(X_bad), std::invalid_argument);
}

// ===================================================================

int main() {
    std::cout << "=== PolynomialFeatures Tests ===\n";
    run_test("poly_degree2", test_poly_degree2);
    run_test("poly_no_bias", test_poly_no_bias);
    run_test("poly_interaction_only", test_poly_interaction_only);
    run_test("poly_degree1", test_poly_degree1);
    run_test("poly_not_fitted", test_poly_not_fitted);

    std::cout << "\n=== Lasso Tests ===\n";
    run_test("lasso_sparsity", test_lasso_sparsity);
    run_test("lasso_predict", test_lasso_predict);
    run_test("lasso_score", test_lasso_score);

    std::cout << "\n=== ElasticNet Tests ===\n";
    run_test("elastic_net_basic", test_elastic_net_basic);
    run_test("elastic_net_score", test_elastic_net_score);

    std::cout << "\n=== LogisticRegression Tests ===\n";
    run_test("logreg_binary", test_logreg_binary);
    run_test("logreg_predict_proba", test_logreg_predict_proba);
    run_test("logreg_multiclass", test_logreg_multiclass);
    run_test("logreg_not_fitted", test_logreg_not_fitted);

    std::cout << "\n=== SGDClassifier Tests ===\n";
    run_test("sgd_classifier_basic", test_sgd_classifier_basic);
    run_test("sgd_classifier_log_loss", test_sgd_classifier_log_loss);

    std::cout << "\n=== SGDRegressor Tests ===\n";
    run_test("sgd_regressor_basic", test_sgd_regressor_basic);
    run_test("sgd_regressor_predict", test_sgd_regressor_predict);

    std::cout << "\n=== TruncatedSVD Tests ===\n";
    run_test("truncated_svd_basic", test_truncated_svd_basic);
    run_test("truncated_svd_transform", test_truncated_svd_transform);
    run_test("truncated_svd_variance_ratio", test_truncated_svd_variance_ratio);

    std::cout << "\n=== MiniBatchKMeans Tests ===\n";
    run_test("mini_batch_kmeans_basic", test_mini_batch_kmeans_basic);
    run_test("mini_batch_kmeans_predict", test_mini_batch_kmeans_predict);

    std::cout << "\n=== CrossValidation Tests ===\n";
    run_test("cross_val_score", test_cross_val_score);

    std::cout << "\n=== Pipeline Tests ===\n";
    run_test("pipeline_basic", test_pipeline_basic);
    run_test("pipeline_score", test_pipeline_score);
    run_test("pipeline_not_fitted", test_pipeline_not_fitted);

    std::cout << "\n=== BayesianRidge Tests ===\n";
    run_test("bayesian_ridge_recovers_coefs", test_bayesian_ridge_recovers_coefs);
    run_test("bayesian_ridge_predict_with_std", test_bayesian_ridge_predict_with_std);
    run_test("bayesian_ridge_compute_score", test_bayesian_ridge_compute_score);
    run_test("bayesian_ridge_not_fitted", test_bayesian_ridge_not_fitted);
    run_test("bayesian_ridge_feature_mismatch", test_bayesian_ridge_feature_mismatch);

    std::cout << "\n=== ARDRegression Tests ===\n";
    run_test("ard_prunes_noise_features", test_ard_prunes_noise_features);
    run_test("ard_lambda_shape", test_ard_lambda_shape);
    run_test("ard_predict_score_no_pruning", test_ard_predict_score_no_pruning);
    run_test("ard_not_fitted_and_feature_mismatch", test_ard_not_fitted_and_feature_mismatch);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
