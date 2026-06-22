// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <functional>
#include <iostream>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

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
// Helper: generate a small 2D dataset (three blobs)
// ===================================================================

static Eigen::MatrixXd make_blobs(int n = 30) {
    Eigen::MatrixXd X(n, 3);
    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / n;
        X(i, 0) = std::cos(2.0 * std::numbers::pi * t) + 0.1 * (i % 3);
        X(i, 1) = std::sin(2.0 * std::numbers::pi * t) + 0.1 * ((i + 1) % 3);
        X(i, 2) = t * 2.0 - 1.0;
    }
    return X;
}

// ===================================================================
// MDS Tests
// ===================================================================

void test_mds_basic() {
    auto X = make_blobs(20);
    Skigen::MDS<double> mds(2, 50, 1e-3, true, 42);
    auto Y = mds.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 20);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(mds.stress() >= 0.0);
    ASSERT_TRUE(mds.n_iter() > 0);
}

void test_mds_preserves_distances() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0, 1, 0, 0, 1, 1, 1;
    Skigen::MDS<double> mds(2, 300, 1e-6, true, 42);
    auto Y = mds.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 4);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(mds.stress() < 0.5);
}

void test_mds_n_init_keeps_best_stress() {
    auto X = make_blobs(18);
    Skigen::MDS<double> single(2, 100, 1e-4, true, 7, 1);
    Skigen::MDS<double> multi(2, 100, 1e-4, true, 7, 4);
    auto single_embedding = single.fit_transform(X);
    auto multi_embedding = multi.fit_transform(X);
    ASSERT_TRUE(single_embedding.rows() == multi_embedding.rows());
    ASSERT_TRUE(single_embedding.cols() == multi_embedding.cols());
    ASSERT_TRUE(multi.n_init() == 4);
    ASSERT_TRUE(multi.n_iter() > 0);
    ASSERT_TRUE(multi.stress() <= single.stress() + 1e-12);
}

void test_mds_invalid_parameters_throw() {
    auto X = make_blobs(8);
    bool threw = false;
    try { Skigen::MDS<double>(2, 100, 1e-3, false, 42).fit(X); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);

    threw = false;
    try { Skigen::MDS<double>(2, 100, 1e-3, true, 42, 0).fit(X); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ===================================================================
// Isomap Tests
// ===================================================================

void test_isomap_basic() {
    auto X = make_blobs(25);
    Skigen::Isomap<double> iso(2, 5);
    auto Y = iso.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 25);
    ASSERT_TRUE(Y.cols() == 2);
}

void test_isomap_dist_matrix() {
    auto X = make_blobs(15);
    Skigen::Isomap<double> iso(2, 4);
    iso.fit(X);
    auto& D = iso.dist_matrix();
    ASSERT_TRUE(D.rows() == 15);
    ASSERT_TRUE(D.cols() == 15);
    for (int i = 0; i < 15; ++i) {
        ASSERT_NEAR(D(i, i), 0.0, 1e-10);
    }
}

void test_isomap_dense_matches_auto() {
    auto X = make_blobs(25);
    Skigen::Isomap<double> iso_auto(2, 5, "auto");
    Skigen::Isomap<double> iso_dense(2, 5, "dense");
    auto Ya = iso_auto.fit_transform(X);
    auto Yd = iso_dense.fit_transform(X);
    for (int i = 0; i < Ya.rows(); ++i)
        for (int j = 0; j < Ya.cols(); ++j)
            ASSERT_NEAR(std::abs(Ya(i, j)), std::abs(Yd(i, j)), 1e-9);
}

void test_isomap_invalid_solver_throws() {
    auto X = make_blobs(12);
    Skigen::Isomap<double> iso(2, 4, "lanczos");
    bool threw = false;
    try { auto Y = iso.fit_transform(X); (void)Y; }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_lle_dense_matches_auto() {
    auto X = make_blobs(25);
    Skigen::LocallyLinearEmbedding<double> lle_auto(2, 6, 1e-3, 100, 1e-6,
                                                    "standard", "auto");
    Skigen::LocallyLinearEmbedding<double> lle_dense(2, 6, 1e-3, 100, 1e-6,
                                                     "standard", "dense");
    auto Ya = lle_auto.fit_transform(X);
    auto Yd = lle_dense.fit_transform(X);
    for (int i = 0; i < Ya.rows(); ++i)
        for (int j = 0; j < Ya.cols(); ++j)
            ASSERT_NEAR(std::abs(Ya(i, j)), std::abs(Yd(i, j)), 1e-9);
}

void test_lle_invalid_solver_throws() {
    auto X = make_blobs(15);
    Skigen::LocallyLinearEmbedding<double> lle(2, 5, 1e-3, 100, 1e-6,
                                               "standard", "lanczos");
    bool threw = false;
    try { auto Y = lle.fit_transform(X); (void)Y; }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ===================================================================
// TSNE Tests
// ===================================================================

void test_tsne_basic() {
    auto X = make_blobs(20);
    Skigen::TSNE<double> tsne(2, 5.0, 200.0, 300, "exact", 0.5, 12.0,
                              std::optional<uint64_t>(42));
    auto Y = tsne.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 20);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(tsne.kl_divergence() >= 0.0);
}

void test_tsne_barnes_hut_shape() {
    auto X = make_blobs(40);
    Skigen::TSNE<double> tsne(2, 8.0, 200.0, 300, "barnes_hut", 0.5, 12.0,
                              std::optional<uint64_t>(0));
    auto Y = tsne.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 40);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(tsne.kl_divergence() >= 0.0);
    ASSERT_TRUE(tsne.method() == std::string("barnes_hut"));
}

void test_tsne_barnes_hut_separates_clusters() {
    // Two well-separated 5-D blobs should remain separated in the 2-D
    // Barnes-Hut embedding (cross-cluster distance > within-cluster).
    constexpr int n = 60;
    Eigen::MatrixXd X(n, 5);
    std::mt19937_64 rng(1);
    std::normal_distribution<double> ns(0.0, 0.3);
    for (int i = 0; i < n; ++i) {
        const double c = (i < n / 2) ? -5.0 : 5.0;
        for (int j = 0; j < 5; ++j) X(i, j) = c + ns(rng);
    }
    Skigen::TSNE<double> tsne(2, 10.0, 200.0, 500, "barnes_hut", 0.5, 12.0,
                              std::optional<uint64_t>(0));
    Eigen::MatrixXd Y = tsne.fit_transform(X);
    Eigen::RowVector2d c0 = Y.topRows(n / 2).colwise().mean();
    Eigen::RowVector2d c1 = Y.bottomRows(n / 2).colwise().mean();
    const double between = (c0 - c1).norm();
    double within = 0.0;
    for (int i = 0; i < n / 2; ++i) within += (Y.row(i) - c0).norm();
    within /= (n / 2);
    ASSERT_TRUE(between > within);
}

void test_tsne_invalid_method_throws() {
    auto X = make_blobs(10);
    Skigen::TSNE<double> tsne(2, 5.0, 200.0, 100, "tree", 0.5, 12.0,
                              std::optional<uint64_t>(0));
    bool threw = false;
    try { auto Y = tsne.fit_transform(X); (void)Y; }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_tsne_3d_falls_back_to_exact() {
    auto X = make_blobs(20);
    // n_components != 2 ⇒ barnes_hut downgrades to exact.
    Skigen::TSNE<double> tsne(3, 5.0, 200.0, 150, "barnes_hut", 0.5, 12.0,
                              std::optional<uint64_t>(0));
    auto Y = tsne.fit_transform(X);
    ASSERT_TRUE(Y.cols() == 3);
    ASSERT_TRUE(tsne.method() == std::string("exact"));
}

// ===================================================================
// LLE Tests
// ===================================================================

void test_lle_basic() {
    auto X = make_blobs(20);
    Skigen::LocallyLinearEmbedding<double> lle(2, 5);
    auto Y = lle.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 20);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(lle.reconstruction_error() >= 0.0);
}

void test_lle_modified_method() {
    auto X = make_blobs(25);
    Skigen::LocallyLinearEmbedding<double> lle(2, 6, 1e-3, 100, 1e-6, "modified");
    auto Y = lle.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 25);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(lle.method() == "modified");
    bool has_nan = false;
    for (int i = 0; i < Y.rows(); ++i)
        for (int j = 0; j < Y.cols(); ++j)
            if (std::isnan(Y(i, j))) has_nan = true;
    ASSERT_TRUE(!has_nan);
}

void test_lle_hessian_and_ltsa_methods() {
    auto X = make_blobs(30);
    for (const char* method : {"hessian", "ltsa"}) {
        Skigen::LocallyLinearEmbedding<double> lle(2, 8, 1e-3, 100, 1e-6, method);
        auto Y = lle.fit_transform(X);
        ASSERT_TRUE(Y.rows() == 30);
        ASSERT_TRUE(Y.cols() == 2);
        ASSERT_TRUE(lle.method() == method);
        bool has_nan = false;
        for (int i = 0; i < Y.rows(); ++i)
            for (int j = 0; j < Y.cols(); ++j)
                if (std::isnan(Y(i, j))) has_nan = true;
        ASSERT_TRUE(!has_nan);
    }
}

void test_lle_invalid_method_and_neighbors() {
    auto X = make_blobs(12);
    bool threw = false;
    try { Skigen::LocallyLinearEmbedding<double>(2, 5, 1e-3, 100, 1e-6, "bogus").fit(X); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);

    threw = false;
    // modified LLE requires n_neighbors > n_components.
    try { Skigen::LocallyLinearEmbedding<double>(2, 2, 1e-3, 100, 1e-6, "modified").fit(X); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);

    threw = false;
    // hessian LLE requires n_neighbors >= 1 + d + d(d+1)/2 = 6 for d=2.
    try { Skigen::LocallyLinearEmbedding<double>(2, 4, 1e-3, 100, 1e-6, "hessian").fit(X); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ===================================================================
// SpectralEmbedding Tests
// ===================================================================

void test_spectral_embedding_basic() {
    auto X = make_blobs(20);
    Skigen::SpectralEmbedding<double> se(2, 5);
    auto Y = se.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 20);
    ASSERT_TRUE(Y.cols() == 2);
    auto& A = se.affinity_matrix();
    ASSERT_TRUE(A.rows() == 20);
    ASSERT_TRUE(A.cols() == 20);
    ASSERT_TRUE(se.eigen_solver() == std::string("auto"));
}

// The "dense" eigen_solver must reproduce the default ("auto") embedding
// exactly when the Spectra backend is not compiled in (sign-invariant: an
// eigenvector and its negation are both valid, so compare |Y|).
void test_spectral_embedding_dense_matches_auto() {
    auto X = make_blobs(25);
    Skigen::SpectralEmbedding<double> se_auto(2, 5, "auto");
    Skigen::SpectralEmbedding<double> se_dense(2, 5, "dense");
    auto Ya = se_auto.fit_transform(X);
    auto Yd = se_dense.fit_transform(X);
    ASSERT_TRUE(Ya.rows() == Yd.rows());
    ASSERT_TRUE(Ya.cols() == Yd.cols());
    for (int i = 0; i < Ya.rows(); ++i)
        for (int j = 0; j < Ya.cols(); ++j)
            ASSERT_NEAR(std::abs(Ya(i, j)), std::abs(Yd(i, j)), 1e-9);
}

void test_spectral_embedding_invalid_solver_throws() {
    auto X = make_blobs(15);
    Skigen::SpectralEmbedding<double> se(2, 5, "lanczos");
    bool threw = false;
    try { auto Y = se.fit_transform(X); (void)Y; }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ===================================================================
// UMAP Tests
// ===================================================================

void test_umap_basic() {
    auto X = make_blobs(30);
    Skigen::UMAP<double> umap(2, 5, 0.1, 1.0, 100, 5, 42);
    auto Y = umap.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 30);
    ASSERT_TRUE(Y.cols() == 2);
    bool has_nan = false;
    for (int i = 0; i < Y.rows(); ++i) {
        for (int j = 0; j < Y.cols(); ++j) {
            if (std::isnan(Y(i, j))) has_nan = true;
        }
    }
    ASSERT_TRUE(!has_nan);
}

// ===================================================================
// SKIGEN_PARAMS Tests
// ===================================================================

void test_manifold_params() {
    Skigen::MDS<double> mds;
    auto p = mds.get_params();
    ASSERT_TRUE(p.size() >= 3);
    ASSERT_TRUE(std::get<int>(p.at("n_components")) == 2);
    ASSERT_TRUE(std::get<int>(p.at("max_iter")) == 300);
    ASSERT_TRUE(std::get<int>(p.at("n_init")) == 4);

    Skigen::UMAP<double> umap;
    auto pu = umap.get_params();
    ASSERT_TRUE(std::get<int>(pu.at("n_neighbors")) == 15);
    ASSERT_NEAR(std::get<double>(pu.at("min_dist")), 0.1, 1e-10);
}

// ===================================================================

int main() {
    std::cout << "=== MDS Tests ===\n";
    run_test("mds_basic", test_mds_basic);
    run_test("mds_preserves_distances", test_mds_preserves_distances);
    run_test("mds_n_init_keeps_best_stress", test_mds_n_init_keeps_best_stress);
    run_test("mds_invalid_parameters_throw", test_mds_invalid_parameters_throw);

    std::cout << "\n=== Isomap Tests ===\n";
    run_test("isomap_basic", test_isomap_basic);
    run_test("isomap_dist_matrix", test_isomap_dist_matrix);
    run_test("isomap_dense_matches_auto", test_isomap_dense_matches_auto);
    run_test("isomap_invalid_solver_throws",
             test_isomap_invalid_solver_throws);

    std::cout << "\n=== TSNE Tests ===\n";
    run_test("tsne_basic", test_tsne_basic);
    run_test("tsne_barnes_hut_shape", test_tsne_barnes_hut_shape);
    run_test("tsne_barnes_hut_separates_clusters",
             test_tsne_barnes_hut_separates_clusters);
    run_test("tsne_invalid_method_throws", test_tsne_invalid_method_throws);
    run_test("tsne_3d_falls_back_to_exact", test_tsne_3d_falls_back_to_exact);

    std::cout << "\n=== LLE Tests ===\n";
    run_test("lle_basic", test_lle_basic);
    run_test("lle_modified_method", test_lle_modified_method);
    run_test("lle_hessian_and_ltsa_methods", test_lle_hessian_and_ltsa_methods);
    run_test("lle_invalid_method_and_neighbors", test_lle_invalid_method_and_neighbors);
    run_test("lle_dense_matches_auto", test_lle_dense_matches_auto);
    run_test("lle_invalid_solver_throws", test_lle_invalid_solver_throws);

    std::cout << "\n=== SpectralEmbedding Tests ===\n";
    run_test("spectral_embedding_basic", test_spectral_embedding_basic);
    run_test("spectral_embedding_dense_matches_auto",
             test_spectral_embedding_dense_matches_auto);
    run_test("spectral_embedding_invalid_solver_throws",
             test_spectral_embedding_invalid_solver_throws);

    std::cout << "\n=== UMAP Tests ===\n";
    run_test("umap_basic", test_umap_basic);

    std::cout << "\n=== SKIGEN_PARAMS Tests ===\n";
    run_test("manifold_params", test_manifold_params);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
