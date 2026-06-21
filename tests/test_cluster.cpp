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

void test_kmeans_fit_predict_matches_labels() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0, 0, 1, 10, 10, 10, 11;

    Skigen::KMeans<double> km(2, 100, 5, 11);
    auto labels = km.fit_predict(X);

    ASSERT_TRUE(labels.size() == km.labels().size());
    for (int i = 0; i < labels.size(); ++i) {
        ASSERT_TRUE(labels(i) == km.labels()(i));
    }
}

void test_mini_batch_kmeans_fit_predict_matches_labels() {
    Eigen::MatrixXd X(6, 2);
    X << 0, 0,
         0, 1,
         1, 0,
         10, 10,
         10, 11,
         11, 10;

    Skigen::MiniBatchKMeans<double> mbk(2, /*batch_size=*/3, /*max_iter=*/20, /*random_state=*/3);
    auto labels = mbk.fit_predict(X);

    ASSERT_TRUE(labels.size() == mbk.labels().size());
    for (int i = 0; i < labels.size(); ++i) {
        ASSERT_TRUE(labels(i) == mbk.labels()(i));
    }
}

// ===================================================================
// DBSCAN Tests
// ===================================================================

void test_dbscan_two_clusters_with_noise() {
    Eigen::MatrixXd X(7, 2);
    X << 0, 0,
         0.1, 0.0,
         0.0, 0.1,
         5.0, 5.0,
         5.1, 5.0,
         5.0, 5.1,
         10.0, 10.0;

    Skigen::DBSCAN<double> dbscan(/*eps=*/0.25, /*min_samples=*/2);
    auto labels = dbscan.fit_predict(X);

    ASSERT_TRUE(dbscan.is_fitted());
    ASSERT_TRUE(labels(0) == labels(1));
    ASSERT_TRUE(labels(0) == labels(2));
    ASSERT_TRUE(labels(3) == labels(4));
    ASSERT_TRUE(labels(3) == labels(5));
    ASSERT_TRUE(labels(0) != labels(3));
    ASSERT_TRUE(labels(6) == -1);
    ASSERT_TRUE(dbscan.core_sample_indices().size() == 6);
}

void test_dbscan_manhattan_metric() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0,
         0.4, 0.4,
         3.0, 3.0,
         3.3, 3.3;

    Skigen::DBSCAN<double> dbscan(/*eps=*/0.9, /*min_samples=*/2, "manhattan");
    auto labels = dbscan.fit_predict(X);
    ASSERT_TRUE(labels(0) == labels(1));
    ASSERT_TRUE(labels(2) == labels(3));
    ASSERT_TRUE(labels(0) != labels(2));
}

void test_dbscan_invalid_parameters_throw() {
    Eigen::MatrixXd X(2, 2);
    X << 0, 0, 1, 1;
    ASSERT_THROW(Skigen::DBSCAN<double>(0.0).fit(X), std::invalid_argument);
    ASSERT_THROW(Skigen::DBSCAN<double>(0.5, 0).fit(X), std::invalid_argument);
    ASSERT_THROW(Skigen::DBSCAN<double>(0.5, 2, "cosine").fit(X), std::invalid_argument);
}

// ===================================================================
// AgglomerativeClustering Tests
// ===================================================================

void test_agglomerative_two_clusters() {
    Eigen::MatrixXd X(6, 2);
    X << 0, 0,
         0.1, 0.0,
         0.0, 0.1,
         5.0, 5.0,
         5.1, 5.0,
         5.0, 5.1;

    Skigen::AgglomerativeClustering<double> model(2, "ward");
    auto labels = model.fit_predict(X);

    ASSERT_TRUE(model.is_fitted());
    ASSERT_TRUE(labels(0) == labels(1));
    ASSERT_TRUE(labels(0) == labels(2));
    ASSERT_TRUE(labels(3) == labels(4));
    ASSERT_TRUE(labels(3) == labels(5));
    ASSERT_TRUE(labels(0) != labels(3));
    ASSERT_TRUE(model.children().rows() == 5);
    ASSERT_TRUE(model.children().cols() == 2);
}

void test_agglomerative_linkage_variants() {
    Eigen::MatrixXd X(5, 2);
    X << 0, 0,
         0, 1,
         10, 10,
         10, 11,
         20, 0;

    Skigen::AgglomerativeClustering<double> complete(3, "complete", "manhattan");
    Skigen::AgglomerativeClustering<double> average(3, "average", "euclidean");
    Skigen::AgglomerativeClustering<double> single(3, "single", "euclidean");
    ASSERT_TRUE(complete.fit_predict(X).size() == 5);
    ASSERT_TRUE(average.fit_predict(X).size() == 5);
    ASSERT_TRUE(single.fit_predict(X).size() == 5);
}

void test_agglomerative_invalid_parameters_throw() {
    Eigen::MatrixXd X(2, 2);
    X << 0, 0, 1, 1;
    ASSERT_THROW(Skigen::AgglomerativeClustering<double>(0).fit(X), std::invalid_argument);
    ASSERT_THROW(Skigen::AgglomerativeClustering<double>(3).fit(X), std::invalid_argument);
    ASSERT_THROW(Skigen::AgglomerativeClustering<double>(2, "centroid").fit(X), std::invalid_argument);
    ASSERT_THROW(Skigen::AgglomerativeClustering<double>(2, "ward", "manhattan").fit(X), std::invalid_argument);
}

// ===================================================================
// MeanShift Tests
// ===================================================================

void test_mean_shift_discovers_modes() {
    Eigen::MatrixXd X(6, 2);
    X << -2.0, 0.0,
         -2.1, 0.1,
         -1.9, -0.1,
          2.0, 0.0,
          2.1, 0.1,
          1.9, -0.1;

    Skigen::MeanShift<double> model(/*bandwidth=*/0.5, /*max_iter=*/100);
    auto labels = model.fit_predict(X);

    ASSERT_TRUE(model.is_fitted());
    ASSERT_TRUE(model.cluster_centers().rows() == 2);
    ASSERT_TRUE(labels(0) == labels(1));
    ASSERT_TRUE(labels(0) == labels(2));
    ASSERT_TRUE(labels(3) == labels(4));
    ASSERT_TRUE(labels(3) == labels(5));
    ASSERT_TRUE(labels(0) != labels(3));
    ASSERT_TRUE(model.n_iter() > 0);
}

void test_mean_shift_predict_and_orphans() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0,
         0.2, 0.0,
         5.0, 5.0,
         5.2, 5.0;

    Skigen::MeanShift<double> model(/*bandwidth=*/0.4, /*max_iter=*/100, /*cluster_all=*/false);
    model.fit(X);

    Eigen::MatrixXd X_new(3, 2);
    X_new << 0.1, 0.0,
             5.1, 5.0,
             20.0, 20.0;
    auto labels = model.predict(X_new);

    ASSERT_TRUE(labels(0) != labels(1));
    ASSERT_TRUE(labels(2) == -1);
}

void test_mean_shift_invalid_parameters_throw() {
    Eigen::MatrixXd X(2, 2);
    X << 0, 0, 1, 1;
    ASSERT_THROW(Skigen::MeanShift<double>(0.0).fit(X), std::invalid_argument);
    ASSERT_THROW(Skigen::MeanShift<double>(1.0, 0).fit(X), std::invalid_argument);
}

// ===================================================================
// Birch Tests
// ===================================================================

void test_birch_compresses_and_clusters() {
    Eigen::MatrixXd X(9, 2);
    X << -3.0, 0.0,
         -3.1, 0.1,
         -2.9, -0.1,
          3.0, 0.0,
          3.1, 0.1,
          2.9, -0.1,
          0.0, 4.0,
          0.1, 4.1,
         -0.1, 3.9;

    Skigen::Birch<double> birch(/*threshold=*/0.35, /*n_clusters=*/3);
    auto labels = birch.fit_predict(X);

    ASSERT_TRUE(birch.is_fitted());
    ASSERT_TRUE(birch.cluster_centers().rows() == 3);
    ASSERT_TRUE(birch.subcluster_centers().rows() <= X.rows());
    ASSERT_TRUE(labels(0) == labels(1));
    ASSERT_TRUE(labels(0) == labels(2));
    ASSERT_TRUE(labels(3) == labels(4));
    ASSERT_TRUE(labels(3) == labels(5));
    ASSERT_TRUE(labels(6) == labels(7));
    ASSERT_TRUE(labels(6) == labels(8));
    ASSERT_TRUE(labels(0) != labels(3));
    ASSERT_TRUE(labels(0) != labels(6));
    ASSERT_TRUE(labels(3) != labels(6));
}

void test_birch_predict() {
    Eigen::MatrixXd X(6, 2);
    X << 0, 0,
         0.1, 0.0,
         5.0, 5.0,
         5.1, 5.0,
         10.0, 0.0,
         10.1, 0.0;

    Skigen::Birch<double> birch(/*threshold=*/0.25, /*n_clusters=*/3);
    birch.fit(X);

    Eigen::MatrixXd X_new(3, 2);
    X_new << 0.05, 0.0,
             5.05, 5.0,
             10.05, 0.0;
    auto labels = birch.predict(X_new);

    ASSERT_TRUE(labels(0) != labels(1));
    ASSERT_TRUE(labels(0) != labels(2));
    ASSERT_TRUE(labels(1) != labels(2));
}

void test_birch_invalid_parameters_throw() {
    Eigen::MatrixXd X(2, 2);
    X << 0, 0, 1, 1;
    ASSERT_THROW(Skigen::Birch<double>(0.0, 2).fit(X), std::invalid_argument);
    ASSERT_THROW(Skigen::Birch<double>(0.5, 0).fit(X), std::invalid_argument);
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
    run_test("kmeans_fit_predict_matches_labels", test_kmeans_fit_predict_matches_labels);
    run_test("mini_batch_kmeans_fit_predict_matches_labels", test_mini_batch_kmeans_fit_predict_matches_labels);

    std::cout << "\n=== DBSCAN Tests ===\n";
    run_test("dbscan_two_clusters_with_noise", test_dbscan_two_clusters_with_noise);
    run_test("dbscan_manhattan_metric", test_dbscan_manhattan_metric);
    run_test("dbscan_invalid_parameters_throw", test_dbscan_invalid_parameters_throw);

    std::cout << "\n=== AgglomerativeClustering Tests ===\n";
    run_test("agglomerative_two_clusters", test_agglomerative_two_clusters);
    run_test("agglomerative_linkage_variants", test_agglomerative_linkage_variants);
    run_test("agglomerative_invalid_parameters_throw", test_agglomerative_invalid_parameters_throw);

    std::cout << "\n=== MeanShift Tests ===\n";
    run_test("mean_shift_discovers_modes", test_mean_shift_discovers_modes);
    run_test("mean_shift_predict_and_orphans", test_mean_shift_predict_and_orphans);
    run_test("mean_shift_invalid_parameters_throw", test_mean_shift_invalid_parameters_throw);

    std::cout << "\n=== Birch Tests ===\n";
    run_test("birch_compresses_and_clusters", test_birch_compresses_and_clusters);
    run_test("birch_predict", test_birch_predict);
    run_test("birch_invalid_parameters_throw", test_birch_invalid_parameters_throw);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
