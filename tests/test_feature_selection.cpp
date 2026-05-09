// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Minimal test harness (mirrors test_standard_scaler.cpp)
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
        try { static_cast<void>(expr); }                                       \
        catch (const ExType&) { caught = true; }                               \
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

#define ASSERT_NEAR(a, b, tol) assert_near<double>(a, b, tol, __FILE__, __LINE__)

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
// VarianceThreshold tests
// ---------------------------------------------------------------------------

void test_variance_threshold_removes_constants() {
    Eigen::MatrixXd X(4, 3);
    X << 1, 5, 2,
         2, 5, 3,
         3, 5, 4,
         4, 5, 5;

    Skigen::VarianceThreshold<double> vt(0.0);
    vt.fit(X);

    auto support = vt.get_support_mask();
    ASSERT_TRUE(support(0));
    ASSERT_TRUE(!support(1));
    ASSERT_TRUE(support(2));

    Eigen::MatrixXd Xs = vt.transform(X);
    ASSERT_TRUE(Xs.cols() == 2);
    ASSERT_TRUE(Xs.rows() == 4);
}

void test_variance_threshold_positive() {
    Eigen::MatrixXd X(4, 3);
    // var per column: [1.25, 0.1875, 1.25]
    X << 1.0, 0.5, 1.0,
         2.0, 1.0, 2.0,
         3.0, 0.5, 3.0,
         4.0, 1.0, 4.0;

    Skigen::VarianceThreshold<double> vt(0.5);
    vt.fit(X);
    auto sup = vt.get_support_mask();
    ASSERT_TRUE(sup(0));
    ASSERT_TRUE(!sup(1));
    ASSERT_TRUE(sup(2));
}

void test_variance_threshold_inverse_transform_shape() {
    Eigen::MatrixXd X(4, 3);
    X << 1, 5, 2,
         2, 5, 3,
         3, 5, 4,
         4, 5, 5;

    Skigen::VarianceThreshold<double> vt(0.0);
    Eigen::MatrixXd Xs = vt.fit(X).transform(X);
    Eigen::MatrixXd Xb = vt.inverse_transform(Xs);
    ASSERT_TRUE(Xb.rows() == X.rows());
    ASSERT_TRUE(Xb.cols() == X.cols());
    // Constant column zeroed, others recovered
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(Xb(i, 0), X(i, 0), 1e-12);
        ASSERT_NEAR(Xb(i, 1), 0.0, 1e-12);
        ASSERT_NEAR(Xb(i, 2), X(i, 2), 1e-12);
    }
}

void test_variance_threshold_not_fitted_throws() {
    Skigen::VarianceThreshold<double> vt;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(vt.transform(X), std::runtime_error);
    ASSERT_THROW(vt.variances(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// SelectKBest tests
// ---------------------------------------------------------------------------

void test_select_k_best_classification() {
    // Feature 0: strong signal (cleanly separates classes)
    // Feature 1: noisy/random
    // Feature 2: stronger separation
    Eigen::MatrixXd X(6, 3);
    X << 1.0, 0.3, 5.0,
         2.0, 0.5, 5.5,
         1.5, 0.1, 5.2,
         10.0, 0.4, 50.0,
         11.0, 0.2, 51.0,
         10.5, 0.6, 50.5;
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    Skigen::SelectKBest<double> sel(Skigen::feature_selection::FClassif<double>{}, 2);
    sel.fit(X, y);
    auto sup = sel.get_support_mask();
    ASSERT_TRUE(sup(0));
    ASSERT_TRUE(!sup(1));
    ASSERT_TRUE(sup(2));

    Eigen::MatrixXd Xs = sel.transform(X);
    ASSERT_TRUE(Xs.cols() == 2);
}

void test_select_k_best_regression() {
    Eigen::MatrixXd X(5, 3);
    // Feature 0 perfectly correlated with y
    // Feature 1 anti-correlated
    // Feature 2 noisy
    X << 1.0, 5.0, 0.5,
         2.0, 4.0, 1.2,
         3.0, 3.0, 0.4,
         4.0, 2.0, 1.7,
         5.0, 1.0, 0.9;
    Eigen::VectorXd y(5);
    y << 1.0, 2.1, 2.9, 4.1, 5.0;

    Skigen::SelectKBestFRegression<double> sel(
        Skigen::feature_selection::FRegression<double>{}, 2);
    sel.fit(X, y);
    auto sup = sel.get_support_mask();
    // Top 2 should be the highly-correlated columns (0 and 1).
    ASSERT_TRUE(sup(0));
    ASSERT_TRUE(sup(1));
    ASSERT_TRUE(!sup(2));
}

void test_select_k_best_chi2_works() {
    Eigen::MatrixXd X(6, 2);
    X << 1.0, 0.0,
         0.0, 1.0,
         1.0, 0.0,
         0.0, 1.0,
         1.0, 0.0,
         0.0, 1.0;
    Eigen::VectorXi y(6);
    y << 0, 1, 0, 1, 0, 1;

    Skigen::SelectKBestChi2<double> sel(
        Skigen::feature_selection::Chi2<double>{}, 1);
    sel.fit(X, y);
    auto& s = sel.scores();
    // Both columns should yield chi2 = 3 (perfectly separating)
    ASSERT_NEAR(s(0), 3.0, 1e-6);
    ASSERT_NEAR(s(1), 3.0, 1e-6);
}

void test_chi2_negative_input_throws() {
    Eigen::MatrixXd X(3, 2);
    X << 1.0, 0.0,
         0.0, -1.0,
         1.0, 0.0;
    Eigen::VectorXi y(3);
    y << 0, 1, 0;

    ASSERT_THROW(Skigen::feature_selection::chi2<double>(X, y),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// f_classif p-values vs scipy
// ---------------------------------------------------------------------------

void test_f_classif_pvalues_scipy() {
    // From scipy: f_classif on 6x2 with second column constant
    // Expected: F = [72, NaN], p = [0.00105756, NaN]
    Eigen::MatrixXd X(6, 2);
    X << 1.0, 1.0,
         2.0, 1.0,
         1.0, 1.0,
         5.0, 1.0,
         6.0, 1.0,
         5.0, 1.0;
    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    auto [F, p] = Skigen::feature_selection::f_classif<double>(X, y);

    ASSERT_NEAR(F(0), 72.0, 1e-9);
    ASSERT_NEAR(p(0), 0.001057564615830686, 1e-9);
    // Constant column → NaN
    ASSERT_TRUE(std::isnan(F(1)));
    ASSERT_TRUE(std::isnan(p(1)));
}

void test_f_regression_pvalues_scipy() {
    // From scipy: f_regression on 5x2.
    // X = [[1,0.5],[2,1],[3,0.2],[4,1.5],[5,0.7]], y = [1,2.1,2.9,4.1,5]
    // Expected (F, p) approx: (1071.43, 6.267e-05) and (0.377244, 0.582521)
    Eigen::MatrixXd X(5, 2);
    X << 1.0, 0.5,
         2.0, 1.0,
         3.0, 0.2,
         4.0, 1.5,
         5.0, 0.7;
    Eigen::VectorXd y(5);
    y << 1.0, 2.1, 2.9, 4.1, 5.0;

    auto [F, p] = Skigen::feature_selection::f_regression<double>(X, y);

    // Allow modest relative tolerance (continued-fraction precision)
    ASSERT_NEAR(F(0), 1071.42857142857, 1e-6);
    ASSERT_NEAR(F(1), 0.37724430131923, 1e-9);
    ASSERT_NEAR(p(0), 6.26712857e-05, 1e-9);
    ASSERT_NEAR(p(1), 0.58252076716175, 1e-9);
}

void test_chi2_pvalues_scipy() {
    // scipy chi2 on 6x2 with perfectly-separating binary features
    // Expected stat = [3, 3], p = [0.08326452, 0.08326452]
    Eigen::MatrixXd X(6, 2);
    X << 1.0, 0.0,
         0.0, 1.0,
         1.0, 0.0,
         0.0, 1.0,
         1.0, 0.0,
         0.0, 1.0;
    Eigen::VectorXi y(6);
    y << 0, 1, 0, 1, 0, 1;

    auto [stat, p] = Skigen::feature_selection::chi2<double>(X, y);

    ASSERT_NEAR(stat(0), 3.0, 1e-12);
    ASSERT_NEAR(stat(1), 3.0, 1e-12);
    ASSERT_NEAR(p(0), 0.08326451666355, 1e-9);
    ASSERT_NEAR(p(1), 0.08326451666355, 1e-9);
}

// ---------------------------------------------------------------------------
// SelectFromModel tests (Ridge wrapper)
// ---------------------------------------------------------------------------

void test_select_from_model_with_ridge_mean() {
    // Build a target where feature 0 dominates and feature 1 is noise.
    Eigen::MatrixXd X(8, 3);
    X << 1, 0.1,  0.0,
         2, 0.2,  0.1,
         3, 0.3, -0.1,
         4, 0.1,  0.2,
         5, 0.2, -0.2,
         6, 0.3,  0.0,
         7, 0.1,  0.1,
         8, 0.2, -0.1;
    Eigen::VectorXd y = X.col(0) * 5.0;  // strong signal in col 0

    Skigen::Ridge<double> ridge(0.01);
    Skigen::SelectFromModel<Skigen::Ridge<double>> sfm(ridge, std::string("mean"));
    sfm.fit(X, y);

    auto sup = sfm.get_support_mask();
    ASSERT_TRUE(sup(0));
    // Features 1, 2 should fall below mean(|coef|).
    ASSERT_TRUE(!sup(1));
    ASSERT_TRUE(!sup(2));
}

void test_select_from_model_prefit() {
    Eigen::MatrixXd X(8, 3);
    X << 1, 0.1,  0.0,
         2, 0.2,  0.1,
         3, 0.3, -0.1,
         4, 0.1,  0.2,
         5, 0.2, -0.2,
         6, 0.3,  0.0,
         7, 0.1,  0.1,
         8, 0.2, -0.1;
    Eigen::VectorXd y = X.col(0) * 5.0;

    Skigen::Ridge<double> ridge(0.01);
    ridge.fit(X, y);
    Eigen::RowVectorXd c_before = ridge.coef();

    Skigen::SelectFromModel<Skigen::Ridge<double>> sfm(
        ridge, std::string("mean"), /*prefit=*/true);
    sfm.fit(X, y);

    // Ridge inside should not be re-fit; coefs unchanged.
    Eigen::RowVectorXd c_after = sfm.estimator().coef();
    for (Eigen::Index j = 0; j < c_before.size(); ++j) {
        ASSERT_NEAR(c_before(j), c_after(j), 1e-12);
    }
    auto sup = sfm.get_support_mask();
    ASSERT_TRUE(sup(0));
}

// ---------------------------------------------------------------------------
// RFE tests (Ridge wrapper)
// ---------------------------------------------------------------------------

void test_rfe_with_ridge() {
    Eigen::MatrixXd X(10, 4);
    // Cols 0 and 2 carry the signal, 1 and 3 are noise.
    X << 1.0, 0.1, -2.0,  0.05,
         2.0, 0.2, -4.0, -0.05,
         3.0, 0.3, -6.0,  0.10,
         4.0, 0.1, -8.0, -0.10,
         5.0, 0.2,-10.0,  0.05,
         6.0, 0.3,-12.0, -0.05,
         7.0, 0.1,-14.0,  0.10,
         8.0, 0.2,-16.0, -0.10,
         9.0, 0.3,-18.0,  0.05,
        10.0, 0.1,-20.0, -0.05;
    Eigen::VectorXd y = X.col(0) * 3.0 + X.col(2) * (-1.5);

    Skigen::Ridge<double> ridge(0.01);
    using NF = std::variant<int, double>;
    Skigen::RFE<Skigen::Ridge<double>> rfe(
        ridge, std::optional<NF>{NF{2}}, NF{1});
    rfe.fit(X, y);

    auto sup = rfe.support();
    int count = 0;
    for (Eigen::Index j = 0; j < sup.size(); ++j) {
        if (sup(j)) ++count;
    }
    ASSERT_TRUE(count == 2);
    ASSERT_TRUE(sup(0));
    ASSERT_TRUE(sup(2));
    ASSERT_TRUE(rfe.n_features() == 2);

    // Ranking: kept features rank=1, discarded > 1.
    const auto& rk = rfe.ranking();
    ASSERT_TRUE(rk(0) == 1);
    ASSERT_TRUE(rk(2) == 1);
    ASSERT_TRUE(rk(1) >= 2);
    ASSERT_TRUE(rk(3) >= 2);
}

void test_rfe_step_int_vs_fraction() {
    Eigen::MatrixXd X(8, 4);
    X << 1.0, 0.1, -2.0, 0.05,
         2.0, 0.2, -4.0, -0.05,
         3.0, 0.3, -6.0, 0.10,
         4.0, 0.1, -8.0, -0.10,
         5.0, 0.2,-10.0, 0.05,
         6.0, 0.3,-12.0, -0.05,
         7.0, 0.1,-14.0, 0.10,
         8.0, 0.2,-16.0, -0.10;
    Eigen::VectorXd y = X.col(0) * 2.0 + X.col(2) * (-1.0);

    using NF = std::variant<int, double>;
    Skigen::Ridge<double> ridge(0.01);

    // Same target with int step
    Skigen::RFE<Skigen::Ridge<double>> rfe_int(
        ridge, std::optional<NF>{NF{2}}, NF{1});
    rfe_int.fit(X, y);

    // Same target with fractional step (0.25 of features)
    Skigen::RFE<Skigen::Ridge<double>> rfe_frac(
        ridge, std::optional<NF>{NF{2}}, NF{0.25});
    rfe_frac.fit(X, y);

    auto sup_a = rfe_int.support();
    auto sup_b = rfe_frac.support();
    ASSERT_TRUE(sup_a.size() == sup_b.size());
    int ca = 0, cb = 0;
    for (Eigen::Index j = 0; j < sup_a.size(); ++j) {
        if (sup_a(j)) ++ca;
        if (sup_b(j)) ++cb;
    }
    ASSERT_TRUE(ca == 2);
    ASSERT_TRUE(cb == 2);
}

void test_select_from_model_max_features() {
    // 4 features; ask SelectFromModel to keep at most 1, even if mean
    // threshold would otherwise admit several.
    Eigen::MatrixXd X(8, 4);
    X << 1, 0.1, 1.0, 0.0,
         2, 0.2, 2.0, 0.1,
         3, 0.3, 3.0, -0.1,
         4, 0.1, 4.0, 0.2,
         5, 0.2, 5.0, -0.2,
         6, 0.3, 6.0, 0.0,
         7, 0.1, 7.0, 0.1,
         8, 0.2, 8.0, -0.1;
    Eigen::VectorXd y = X.col(0) * 2.0 + X.col(2) * 3.0;

    Skigen::Ridge<double> ridge(0.01);
    Skigen::SelectFromModel<Skigen::Ridge<double>> sfm(
        ridge, std::string("median"), /*prefit=*/false,
        /*max_features=*/std::optional<int>{1});
    sfm.fit(X, y);
    auto sup = sfm.get_support_mask();
    int count = 0;
    for (Eigen::Index j = 0; j < sup.size(); ++j) {
        if (sup(j)) ++count;
    }
    ASSERT_TRUE(count == 1);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== FeatureSelection Tests ===\n";

    run_test("variance_threshold_removes_constants", test_variance_threshold_removes_constants);
    run_test("variance_threshold_positive",         test_variance_threshold_positive);
    run_test("variance_threshold_inverse_transform_shape",
             test_variance_threshold_inverse_transform_shape);
    run_test("variance_threshold_not_fitted_throws", test_variance_threshold_not_fitted_throws);
    run_test("select_k_best_classification",        test_select_k_best_classification);
    run_test("select_k_best_regression",            test_select_k_best_regression);
    run_test("select_k_best_chi2_works",            test_select_k_best_chi2_works);
    run_test("chi2_negative_input_throws",          test_chi2_negative_input_throws);
    run_test("f_classif_pvalues_scipy",             test_f_classif_pvalues_scipy);
    run_test("f_regression_pvalues_scipy",          test_f_regression_pvalues_scipy);
    run_test("chi2_pvalues_scipy",                  test_chi2_pvalues_scipy);
    run_test("select_from_model_with_ridge_mean",   test_select_from_model_with_ridge_mean);
    run_test("select_from_model_prefit",            test_select_from_model_prefit);
    run_test("rfe_with_ridge",                      test_rfe_with_ridge);
    run_test("rfe_step_int_vs_fraction",            test_rfe_step_int_vs_fraction);
    run_test("select_from_model_max_features",      test_select_from_model_max_features);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
