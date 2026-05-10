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

    std::cout << "------------------------------------------\n";
    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
