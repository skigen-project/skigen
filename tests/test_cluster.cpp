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
// KMeans Tests
// ===================================================================

void test_kmeans_basic() {
    // Two well-separated clusters
    Eigen::MatrixXd X(6, 2);
    X << 1, 1,
         1.5, 1.5,
         1, 1.5,
         10, 10,
         10.5, 10.5,
         10, 10.5;

    Skigen::KMeans<double> km(2);
    km.fit(X);

    ASSERT_TRUE(km.is_fitted());
    ASSERT_TRUE(km.cluster_centers().rows() == 2);
    ASSERT_TRUE(km.labels().size() == 6);

    // First 3 and last 3 should be in different clusters
    ASSERT_TRUE(km.labels()(0) == km.labels()(1));
    ASSERT_TRUE(km.labels()(0) == km.labels()(2));
    ASSERT_TRUE(km.labels()(3) == km.labels()(4));
    ASSERT_TRUE(km.labels()(3) == km.labels()(5));
    ASSERT_TRUE(km.labels()(0) != km.labels()(3));
}

void test_kmeans_predict() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0,
         0, 1,
         10, 10,
         10, 11;

    Skigen::KMeans<double> km(2);
    km.fit(X);

    Eigen::MatrixXd X_new(2, 2);
    X_new << 0.5, 0.5,
             9.5, 10.0;

    auto labels = km.predict(X_new);
    ASSERT_TRUE(labels(0) != labels(1));
}

void test_kmeans_transform() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0, 0, 1, 10, 10, 10, 11;

    Skigen::KMeans<double> km(2);
    km.fit(X);

    auto dists = km.transform(X);
    ASSERT_TRUE(dists.rows() == 4);
    ASSERT_TRUE(dists.cols() == 2);
}

void test_kmeans_inertia() {
    // Perfect clusters
    Eigen::MatrixXd X(2, 2);
    X << 0, 0,
         10, 10;

    Skigen::KMeans<double> km(2);
    km.fit(X);

    // Each point is its own center, inertia should be very small
    ASSERT_NEAR(km.inertia(), 0.0, 1e-10);
}

void test_kmeans_not_fitted() {
    Skigen::KMeans<double> km;
    Eigen::MatrixXd X(3, 2);
    X << 1, 2, 3, 4, 5, 6;
    ASSERT_THROW(km.predict(X), std::runtime_error);
}

void test_kmeans_sparse_separates_two_clusters() {
    // Two well-separated clusters with sparse rows.
    Eigen::MatrixXd Xd(8, 4);
    Xd.setZero();
    // Cluster A — large values in cols 0,1
    Xd(0, 0) = 10; Xd(0, 1) = 10;
    Xd(1, 0) = 11; Xd(1, 1) = 9;
    Xd(2, 0) = 9;  Xd(2, 1) = 11;
    Xd(3, 0) = 10; Xd(3, 1) = 10;
    // Cluster B — large values in cols 2,3
    Xd(4, 2) = 10; Xd(4, 3) = 10;
    Xd(5, 2) = 11; Xd(5, 3) = 9;
    Xd(6, 2) = 9;  Xd(6, 3) = 11;
    Xd(7, 2) = 10; Xd(7, 3) = 10;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::KMeans<double> km(2, 100, 5, 7);
    km.fit(Xs);
    auto labels = km.labels();

    // The two halves should fall in different clusters.
    ASSERT_TRUE(labels.size() == 8);
    int a = labels(0);
    int b = labels(4);
    ASSERT_TRUE(a != b);
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(labels(i) == a);
    for (int i = 4; i < 8; ++i) ASSERT_TRUE(labels(i) == b);
}

void test_kmeans_sparse_inertia_matches_dense() {
    // Same dataset fitted with same seed in dense and sparse paths.
    // Inertia should agree closely (initialisations are deterministic with
    // the same seed; cluster assignments may permute labels, so we compare
    // total inertia rather than per-row labels).
    Eigen::MatrixXd Xd(12, 3);
    Xd.setZero();
    Xd(0, 0) = 1; Xd(0, 2) = 2;
    Xd(1, 0) = 2; Xd(1, 2) = 1;
    Xd(2, 0) = 1; Xd(2, 1) = 1; Xd(2, 2) = 2;
    Xd(3, 0) = 2; Xd(3, 2) = 2;
    Xd(4, 1) = 10; Xd(4, 2) = 10;
    Xd(5, 1) = 11; Xd(5, 2) = 9;
    Xd(6, 1) = 9;  Xd(6, 2) = 11;
    Xd(7, 1) = 10; Xd(7, 2) = 10;
    Xd(8, 0) = 20;
    Xd(9, 0) = 21; Xd(9, 1) = 1;
    Xd(10, 0) = 19; Xd(10, 1) = -1;
    Xd(11, 0) = 20;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::KMeans<double> kd(3, 200, 5, 13);
    kd.fit(Xd);
    Skigen::KMeans<double> ks(3, 200, 5, 13);
    ks.fit(Xs);
    // Same seed/data ⇒ same minimised inertia within FP tolerance.
    ASSERT_NEAR(kd.inertia(), ks.inertia(), 1e-9);
}

void test_kmeans_sparse_predict_matches_assignments() {
    Eigen::MatrixXd Xd(8, 2);
    Xd << 0, 0, 0, 1, 1, 0, 1, 1,
          10, 10, 10, 11, 11, 10, 11, 11;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::KMeans<double> km(2, 100, 5, 0);
    km.fit(Xs);
    auto fit_labels = km.labels();
    auto predict_labels = km.predict(Xs);
    ASSERT_TRUE(fit_labels.size() == predict_labels.size());
    for (int i = 0; i < fit_labels.size(); ++i) {
        ASSERT_TRUE(fit_labels(i) == predict_labels(i));
    }
}

void test_kmeans_sparse_n_samples_too_small_throws() {
    Eigen::SparseMatrix<double> X(2, 3);
    X.insert(0, 0) = 1.0;
    X.makeCompressed();
    Skigen::KMeans<double> km(/*n_clusters=*/3);
    ASSERT_THROW(km.fit(X), std::invalid_argument);
}

// ===================================================================

int main() {
    std::cout << "=== KMeans Tests ===\n";
    run_test("kmeans_basic", test_kmeans_basic);
    run_test("kmeans_predict", test_kmeans_predict);
    run_test("kmeans_transform", test_kmeans_transform);
    run_test("kmeans_inertia", test_kmeans_inertia);
    run_test("kmeans_not_fitted", test_kmeans_not_fitted);
    run_test("kmeans_sparse_separates_two_clusters",   test_kmeans_sparse_separates_two_clusters);
    run_test("kmeans_sparse_inertia_matches_dense",    test_kmeans_sparse_inertia_matches_dense);
    run_test("kmeans_sparse_predict_matches_assignments", test_kmeans_sparse_predict_matches_assignments);
    run_test("kmeans_sparse_n_samples_too_small_throws", test_kmeans_sparse_n_samples_too_small_throws);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
