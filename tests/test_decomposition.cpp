// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <Eigen/SparseCore>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <optional>
#include <random>

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
// PCA Tests
// ===================================================================

void test_pca_basic() {
    // 2D data with strong first component
    Eigen::MatrixXd X(5, 2);
    X << 1, 2,
         3, 4,
         5, 6,
         7, 8,
         9, 10;

    Skigen::PCA pca(2);
    pca.fit(X);

    ASSERT_TRUE(pca.n_components() == 2);
    ASSERT_TRUE(pca.components().rows() == 2);
    ASSERT_TRUE(pca.components().cols() == 2);
    ASSERT_TRUE(pca.explained_variance().size() == 2);
}

void test_pca_variance_ratio_sums_to_one() {
    Eigen::MatrixXd X(5, 2);
    X << 1, 2,
         3, 4,
         5, 6,
         7, 8,
         9, 10;

    Skigen::PCA pca;
    pca.fit(X);

    double ratio_sum = pca.explained_variance_ratio().sum();
    ASSERT_NEAR(ratio_sum, 1.0, 1e-10);
}

void test_pca_dimensionality_reduction() {
    Eigen::MatrixXd X(5, 3);
    X << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12,
         13, 14, 15;

    Skigen::PCA pca(2);
    pca.fit(X);
    Eigen::MatrixXd Z = pca.transform(X);

    ASSERT_TRUE(Z.rows() == 5);
    ASSERT_TRUE(Z.cols() == 2);
}

void test_pca_inverse_transform() {
    Eigen::MatrixXd X(5, 2);
    X << 1, 2,
         3, 4,
         5, 6,
         7, 8,
         9, 10;

    Skigen::PCA pca(2);
    pca.fit(X);
    Eigen::MatrixXd Z = pca.transform(X);
    Eigen::MatrixXd X_hat = pca.inverse_transform(Z);

    // Perfect reconstruction with all components
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            ASSERT_NEAR(X_hat(i, j), X(i, j), 1e-10);
        }
    }
}

void test_pca_fit_transform() {
    Eigen::MatrixXd X(5, 2);
    X << 1, 2,
         3, 4,
         5, 6,
         7, 8,
         9, 10;

    Skigen::PCA pca1(1);
    pca1.fit(X);
    Eigen::MatrixXd Z1 = pca1.transform(X);

    Skigen::PCA<double> pca2(1);
    Eigen::MatrixXd Z2 = pca2.fit_transform(X);

    for (Eigen::Index i = 0; i < Z1.rows(); ++i) {
        ASSERT_NEAR(std::abs(Z1(i, 0)), std::abs(Z2(i, 0)), 1e-10);
    }
}

void test_pca_not_fitted() {
    Skigen::PCA pca;
    Eigen::MatrixXd X(3, 2);
    X << 1, 2, 3, 4, 5, 6;
    ASSERT_THROW(pca.transform(X), std::runtime_error);
}

// Build a structured low-rank-ish fixture so the randomized solver has a
// clear dominant subspace to recover.
static Eigen::MatrixXd pca_fixture(int n, int p, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    const int rank = 3;
    Eigen::MatrixXd Z(n, rank), W(rank, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < rank; ++j) Z(i, j) = dist(gen);
    for (int i = 0; i < rank; ++i)
        for (int j = 0; j < p; ++j) W(i, j) = dist(gen);
    Eigen::MatrixXd X = Z * W;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j) X(i, j) += 0.01 * dist(gen);
    return X;
}

void test_pca_randomized_matches_full() {
    const Eigen::MatrixXd X = pca_fixture(120, 8, 7);

    Skigen::PCA<double> full(3, "full");
    full.fit(X);
    Skigen::PCA<double> rnd(3, "randomized", 10, 7, std::optional<uint64_t>(0));
    rnd.fit(X);

    // Dominant singular values should agree within a small tolerance.
    for (Eigen::Index i = 0; i < full.singular_values().size(); ++i)
        ASSERT_NEAR(full.singular_values()(i), rnd.singular_values()(i), 1e-2);

    // Explained-variance ratios agree on the captured components.
    for (Eigen::Index i = 0; i < full.explained_variance_ratio().size(); ++i)
        ASSERT_NEAR(full.explained_variance_ratio()(i),
                    rnd.explained_variance_ratio()(i), 1e-2);
}

void test_pca_sparse_matches_dense() {
    const Eigen::MatrixXd Xd = pca_fixture(150, 6, 11);
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::PCA<double> dense(3, "full");
    dense.fit(Xd);
    Skigen::PCA<double> sparse(3, "randomized", 10, 7,
                               std::optional<uint64_t>(0));
    sparse.fit(Xs);

    // The implicitly-centered sparse path must recover the same mean...
    for (Eigen::Index j = 0; j < Xd.cols(); ++j)
        ASSERT_NEAR(dense.mean()(j), sparse.mean()(j), 1e-10);

    // ...and the same dominant singular values / variance ratios.
    for (Eigen::Index i = 0; i < dense.singular_values().size(); ++i)
        ASSERT_NEAR(dense.singular_values()(i), sparse.singular_values()(i),
                    1e-2);
    for (Eigen::Index i = 0; i < dense.explained_variance_ratio().size(); ++i)
        ASSERT_NEAR(dense.explained_variance_ratio()(i),
                    sparse.explained_variance_ratio()(i), 1e-2);
}

void test_pca_sparse_transform_shapes() {
    const Eigen::MatrixXd Xd = pca_fixture(40, 5, 3);
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::PCA<double> pca(2, "randomized", 10, 7, std::optional<uint64_t>(1));
    pca.fit(Xs);

    // transform_impl works on dense input after a sparse fit.
    Eigen::MatrixXd Z = pca.transform(Xd);
    ASSERT_TRUE(Z.rows() == 40);
    ASSERT_TRUE(Z.cols() == 2);

    Eigen::MatrixXd Xhat = pca.inverse_transform(Z);
    ASSERT_TRUE(Xhat.rows() == 40);
    ASSERT_TRUE(Xhat.cols() == 5);
}

void test_pca_unknown_solver_throws() {
    Eigen::MatrixXd X(5, 2);
    X << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    Skigen::PCA<double> pca(2, "arpack");
    ASSERT_THROW(pca.fit(X), std::invalid_argument);
}

void test_pca_empty_sparse_throws() {
    Eigen::SparseMatrix<double> Xs(0, 0);
    Skigen::PCA<double> pca(2, "randomized");
    ASSERT_THROW(pca.fit(Xs), std::invalid_argument);
}

// ===================================================================
// FactorAnalysis Tests
// ===================================================================

void test_fa_basic() {
    // Generate data with known factor structure
    // X = Z * W^T + noise, Z ~ N(0,I), W is (p × k)
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    const int n = 500, p = 5, k = 2;

    Eigen::MatrixXd Z(n, k);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j)
            Z(i, j) = dist(gen);

    Eigen::MatrixXd W_true(p, k);
    W_true << 2.0, 0.0,
              1.5, 0.5,
              0.0, 2.0,
              0.3, 1.0,
              1.0, 0.0;

    Eigen::MatrixXd noise(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            noise(i, j) = dist(gen) * 0.3;

    Eigen::MatrixXd X = Z * W_true.transpose() + noise;

    Skigen::FactorAnalysis<double> fa(k);
    fa.fit(X);

    ASSERT_TRUE(fa.is_fitted());
    ASSERT_TRUE(fa.components().rows() == p);
    ASSERT_TRUE(fa.components().cols() == k);
    ASSERT_TRUE(fa.noise_variance().size() == p);
    ASSERT_TRUE(fa.n_iter() > 0);

    // Covariance should be symmetric and positive semi-definite
    auto cov = fa.covariance();
    ASSERT_TRUE(cov.rows() == p && cov.cols() == p);

    double sym_err = (cov - cov.transpose()).norm();
    ASSERT_NEAR(sym_err, 0.0, 1e-12);
}

void test_fa_noise_recovery() {
    // Noise variances should be close to the true noise level
    std::mt19937 gen(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    const int n = 2000, p = 4, k = 1;

    Eigen::MatrixXd Z(n, k);
    for (int i = 0; i < n; ++i)
        Z(i, 0) = dist(gen);

    Eigen::VectorXd w(p);
    w << 3.0, 2.0, 1.0, 0.5;

    Eigen::VectorXd noise_std(p);
    noise_std << 0.5, 0.3, 0.8, 0.2;

    Eigen::MatrixXd X(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            X(i, j) = Z(i, 0) * w(j) + dist(gen) * noise_std(j);

    Skigen::FactorAnalysis<double> fa(k);
    fa.fit(X);

    // Each noise_variance should be approximately noise_std^2
    for (int j = 0; j < p; ++j) {
        ASSERT_NEAR(fa.noise_variance()(j), noise_std(j) * noise_std(j), 0.15);
    }
}

void test_fa_not_fitted() {
    Skigen::FactorAnalysis<double> fa;
    bool threw = false;
    try { static_cast<void>(fa.components()); } catch (const std::runtime_error&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_fa_ll_improves() {
    // Log-likelihood should be finite
    std::mt19937 gen(99);
    std::normal_distribution<double> dist(0.0, 1.0);
    const int n = 200, p = 5;
    Eigen::MatrixXd X(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            X(i, j) = dist(gen);

    Skigen::FactorAnalysis<double> fa(2);
    fa.fit(X);

    ASSERT_TRUE(std::isfinite(fa.log_likelihood()));
}

// ===================================================================

int main() {
    std::cout << "=== PCA Tests ===\n";
    run_test("pca_basic", test_pca_basic);
    run_test("pca_variance_ratio_sums_to_one", test_pca_variance_ratio_sums_to_one);
    run_test("pca_dimensionality_reduction", test_pca_dimensionality_reduction);
    run_test("pca_inverse_transform", test_pca_inverse_transform);
    run_test("pca_fit_transform", test_pca_fit_transform);
    run_test("pca_not_fitted", test_pca_not_fitted);
    run_test("pca_randomized_matches_full", test_pca_randomized_matches_full);
    run_test("pca_sparse_matches_dense", test_pca_sparse_matches_dense);
    run_test("pca_sparse_transform_shapes", test_pca_sparse_transform_shapes);
    run_test("pca_unknown_solver_throws", test_pca_unknown_solver_throws);
    run_test("pca_empty_sparse_throws", test_pca_empty_sparse_throws);

    std::cout << "\n=== FactorAnalysis Tests ===\n";
    run_test("fa_basic", test_fa_basic);
    run_test("fa_noise_recovery", test_fa_noise_recovery);
    run_test("fa_not_fitted", test_fa_not_fitted);
    run_test("fa_ll_improves", test_fa_ll_improves);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
