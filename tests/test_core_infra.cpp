// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// Phase A core infrastructure: SparseSupport.h helpers + new concepts.

#include <Skigen/Core>
#include <Skigen/Preprocessing>
#include <Skigen/LinearModel>
#include <Skigen/Cluster>
#include <Skigen/NaiveBayes>

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

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
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        const double aa = static_cast<double>(a);                              \
        const double bb = static_cast<double>(b);                              \
        if (std::fabs(aa - bb) > (tol)) {                                      \
            std::ostringstream oss;                                            \
            oss << __FILE__ << ":" << __LINE__                                 \
                << ": ASSERT_NEAR failed: " << aa << " vs " << bb              \
                << " (tol=" << (tol) << ")";                                   \
            throw TestFailure(oss.str());                                      \
        }                                                                      \
    } while (0)

static void run(const std::string& name, std::function<void()> body) {
    try {
        body();
        ++g_passed;
        std::cout << "  [PASS] " << name << "\n";
    } catch (const TestFailure& e) {
        ++g_failed;
        std::cout << "  [FAIL] " << name << " — " << e.what() << "\n";
    } catch (const std::exception& e) {
        ++g_failed;
        std::cout << "  [FAIL] " << name << " — unexpected: "
                  << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// SparseSupport helpers
// ---------------------------------------------------------------------------

void test_sparse_colwise_sum_matches_dense() {
    Eigen::MatrixXd Xd(4, 3);
    Xd << 1, 0, 2,
          0, 3, 0,
          4, 0, 5,
          0, 0, 1;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();
    Eigen::RowVectorXd s = Skigen::internal::sparse_colwise_sum(Xs);
    Eigen::RowVectorXd expected = Xd.colwise().sum();
    for (int j = 0; j < 3; ++j) ASSERT_NEAR(s(j), expected(j), 1e-12);
}

void test_sparse_colwise_mean_matches_dense() {
    Eigen::MatrixXd Xd(5, 2);
    Xd << 1, 0,
          0, 4,
          2, 0,
          0, 6,
          3, 0;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();
    Eigen::RowVectorXd m = Skigen::internal::sparse_colwise_mean(Xs);
    Eigen::RowVectorXd expected = Xd.colwise().mean();
    for (int j = 0; j < 2; ++j) ASSERT_NEAR(m(j), expected(j), 1e-12);
}

void test_sparse_colwise_squared_norm_matches_dense() {
    Eigen::MatrixXd Xd(4, 3);
    Xd << 1, -2, 0,
          0,  1, 3,
          2,  0, 0,
         -1,  0, 1;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();
    Eigen::VectorXd sn = Skigen::internal::sparse_colwise_squared_norm(Xs);
    for (int j = 0; j < 3; ++j) {
        ASSERT_NEAR(sn(j), Xd.col(j).squaredNorm(), 1e-12);
    }
}

void test_sparse_colwise_variance_matches_dense() {
    Eigen::MatrixXd Xd(6, 2);
    Xd << 1, 0,
          2, 4,
          0, 4,
          3, 0,
          0, 4,
          4, 0;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();
    Eigen::RowVectorXd v = Skigen::internal::sparse_colwise_variance(Xs);
    for (int j = 0; j < 2; ++j) {
        const double mean = Xd.col(j).mean();
        const double expected =
            (Xd.col(j).array() - mean).matrix().squaredNorm() /
            static_cast<double>(Xd.rows());
        ASSERT_NEAR(v(j), expected, 1e-12);
    }
}

void test_sparse_centered_squared_norm_identity() {
    // Verify ||X[:, j] - mean(X[:, j])||^2 = sn(j) - n * mean(j)^2
    Eigen::MatrixXd Xd(5, 3);
    Xd << 1, 0, 0,
          0, 2, 3,
          3, 0, 0,
          0, 0, 1,
          2, 1, 0;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();
    Eigen::VectorXd csn =
        Skigen::internal::sparse_centered_squared_norm(Xs);
    for (int j = 0; j < 3; ++j) {
        const double mean = Xd.col(j).mean();
        const double expected =
            (Xd.col(j).array() - mean).matrix().squaredNorm();
        ASSERT_NEAR(csn(j), expected, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// is_eigen_sparse trait
// ---------------------------------------------------------------------------

void test_is_eigen_sparse_trait() {
    static_assert(Skigen::internal::is_eigen_sparse_v<
                  Eigen::SparseMatrix<double>>);
    static_assert(Skigen::internal::is_eigen_sparse_v<
                  Eigen::SparseMatrix<float, Eigen::RowMajor>>);
    static_assert(!Skigen::internal::is_eigen_sparse_v<Eigen::MatrixXd>);
    static_assert(!Skigen::internal::is_eigen_sparse_v<int>);
    ASSERT_TRUE(true);
}

// ---------------------------------------------------------------------------
// Concepts
// ---------------------------------------------------------------------------

void test_incremental_concepts() {
    // Unsupervised partial_fit(X) — StandardScaler.
    static_assert(Skigen::IncrementalUnsupervised<
                  Skigen::StandardScaler<double>>);
    static_assert(Skigen::IncrementalLike<
                  Skigen::StandardScaler<double>>);
    // MiniBatchKMeans is also unsupervised IncrementalLike.
    static_assert(Skigen::IncrementalUnsupervised<
                  Skigen::MiniBatchKMeans<double>>);
    // Supervised partial_fit(X, y) — SGDRegressor.
    static_assert(Skigen::IncrementalSupervised<
                  Skigen::SGDRegressor<double>>);
    static_assert(Skigen::IncrementalLike<
                  Skigen::SGDRegressor<double>>);
    // LinearRegression has no partial_fit — must NOT satisfy the concept.
    static_assert(!Skigen::IncrementalLike<
                  Skigen::LinearRegression<double>>);
    ASSERT_TRUE(true);
}

void test_multi_output_regressor_concept() {
    static_assert(Skigen::MultiOutputRegressorLike<
                  Skigen::LinearRegression<double>>);
    static_assert(Skigen::MultiOutputRegressorLike<
                  Skigen::Ridge<double>>);
    static_assert(Skigen::MultiOutputRegressorLike<
                  Skigen::Lasso<double>>);
    static_assert(Skigen::MultiOutputRegressorLike<
                  Skigen::ElasticNet<double>>);
    // SGDRegressor doesn't expose fit_multi — must NOT satisfy.
    static_assert(!Skigen::MultiOutputRegressorLike<
                  Skigen::SGDRegressor<double>>);
    ASSERT_TRUE(true);
}

// ---------------------------------------------------------------------------
// SKIGEN_PARAMS reflection layer (Phase A)
// ---------------------------------------------------------------------------

void test_get_params_returns_registered_pairs() {
    Skigen::Ridge<double> r(/*alpha=*/2.5, /*fit_intercept=*/false);
    Skigen::ParameterDict d = r.get_params();
    ASSERT_TRUE(d.size() == 2);
    ASSERT_TRUE(d.count("alpha") == 1);
    ASSERT_TRUE(d.count("fit_intercept") == 1);
    ASSERT_NEAR(std::get<double>(d["alpha"]), 2.5, 1e-12);
    ASSERT_TRUE(std::get<bool>(d["fit_intercept"]) == false);
}

void test_set_param_round_trip() {
    Skigen::Ridge<double> r(1.0, true);
    r.set_param("alpha", Skigen::ParameterValue(0.25));
    r.set_param("fit_intercept", Skigen::ParameterValue(false));
    auto d = r.get_params();
    ASSERT_NEAR(std::get<double>(d["alpha"]), 0.25, 1e-12);
    ASSERT_TRUE(std::get<bool>(d["fit_intercept"]) == false);
    ASSERT_NEAR(r.alpha(), 0.25, 1e-12);
    ASSERT_TRUE(r.fit_intercept() == false);
}

void test_set_param_int_to_double_widening() {
    Skigen::Ridge<double> r;
    // Pass an int where double is expected — widening must succeed.
    r.set_param("alpha", Skigen::ParameterValue(3));
    ASSERT_NEAR(r.alpha(), 3.0, 1e-12);
}

void test_set_param_unknown_throws() {
    Skigen::Ridge<double> r;
    bool threw = false;
    try { r.set_param("nonexistent", Skigen::ParameterValue(1.0)); }
    catch (const Skigen::UnknownParameter&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_set_param_type_mismatch_throws() {
    Skigen::Ridge<double> r;
    // Try to set fit_intercept (bool) with a string — must reject.
    bool threw = false;
    try {
        r.set_param("fit_intercept",
                    Skigen::ParameterValue(std::string("yes")));
    } catch (const Skigen::ParameterTypeMismatch&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_default_get_params_empty_for_unregistered_estimator() {
    // LinearRegression doesn't (yet) register params — its get_params
    // should return an empty dict via the base-class fallback, and
    // set_param on any name should throw UnknownParameter.
    Skigen::LinearRegression<double> lr;
    auto d = lr.get_params();
    ASSERT_TRUE(d.empty());
    bool threw = false;
    try { lr.set_param("alpha", Skigen::ParameterValue(1.0)); }
    catch (const Skigen::UnknownParameter&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_parametrized_like_concept() {
    // Ridge has SKIGEN_PARAMS registered; LinearRegression doesn't.
    static_assert(Skigen::ParametrizedLike<Skigen::Ridge<double>>);
    static_assert(!Skigen::ParametrizedLike<
                  Skigen::LinearRegression<double>>);
    ASSERT_TRUE(true);
}

int main() {
    std::cout << "Skigen Core infrastructure (Phase A) tests\n";
    std::cout << "------------------------------------------\n";

    run("sparse_colwise_sum_matches_dense",         test_sparse_colwise_sum_matches_dense);
    run("sparse_colwise_mean_matches_dense",        test_sparse_colwise_mean_matches_dense);
    run("sparse_colwise_squared_norm_matches_dense",test_sparse_colwise_squared_norm_matches_dense);
    run("sparse_colwise_variance_matches_dense",    test_sparse_colwise_variance_matches_dense);
    run("sparse_centered_squared_norm_identity",    test_sparse_centered_squared_norm_identity);
    run("is_eigen_sparse_trait",                    test_is_eigen_sparse_trait);
    run("incremental_concepts",                     test_incremental_concepts);
    run("multi_output_regressor_concept",           test_multi_output_regressor_concept);
    run("get_params_returns_registered_pairs",      test_get_params_returns_registered_pairs);
    run("set_param_round_trip",                     test_set_param_round_trip);
    run("set_param_int_to_double_widening",         test_set_param_int_to_double_widening);
    run("set_param_unknown_throws",                 test_set_param_unknown_throws);
    run("set_param_type_mismatch_throws",           test_set_param_type_mismatch_throws);
    run("default_get_params_empty_for_unregistered_estimator",
        test_default_get_params_empty_for_unregistered_estimator);
    run("parametrized_like_concept",                test_parametrized_like_concept);

    std::cout << "------------------------------------------\n";
    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
